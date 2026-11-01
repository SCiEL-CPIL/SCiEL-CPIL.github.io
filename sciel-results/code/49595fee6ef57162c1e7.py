"""Structural change: gap-permutation CSP + SLS with cross-row gap swaps."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    counting_bound = total_diffs

    def row_diffs_from_gaps(gaps):
        """Given k gaps (positive ints), return the set of all C(k+1,2) differences."""
        diffs = set()
        # Build prefix sums
        pos = [0]
        for g in gaps:
            pos.append(pos[-1] + g)
        # Compute all pairwise differences
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diffs.add(pos[j] - pos[i])
        return diffs

    def validate_gaps(all_gaps):
        """Check all differences are distinct across rows. Returns (valid, max_entry)."""
        used = set()
        max_entry = 0
        for gaps in all_gaps:
            pos = [0]
            for g in gaps:
                pos.append(pos[-1] + g)
            max_entry = max(max_entry, pos[-1])
            # Compute all diffs
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = pos[j] - pos[i]
                    if d in used:
                        return False, max_entry
                    used.add(d)
        return True, max_entry

    def compute_max_entry(all_gaps):
        max_e = 0
        for gaps in all_gaps:
            s = 0
            for g in gaps:
                s += g
            max_e = max(max_e, s)
        return max_e

    # Phase 1: Initialize with greedy construction to get a valid starting point
    def greedy_init(bound):
        """Build valid rows greedily with given upper bound."""
        used = set()
        all_gaps = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                start = row[-1] + 1
                for x in range(start, bound + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            # Convert to gaps
            gaps = [row[i+1] - row[i] for i in range(k)]
            all_gaps.append(gaps)
        return all_gaps

    # Try greedy with increasing bounds
    best_gaps = None
    best_max = float('inf')
    
    for bound in [counting_bound + 20, counting_bound + 40, counting_bound + 60, 
                  counting_bound + 100, counting_bound + 150, counting_bound + 200]:
        if time.time() - start_time > deadline - 10:
            break
        gaps = greedy_init(bound)
        if gaps is not None:
            valid, m = validate_gaps(gaps)
            if valid and m < best_max:
                best_max = m
                best_gaps = [g[:] for g in gaps]
                break

    if best_gaps is None:
        # Fallback: very generous bound
        gaps = greedy_init(counting_bound + 500)
        if gaps is None:
            gaps = greedy_init(counting_bound + 1000)
        best_gaps = gaps
        valid, best_max = validate_gaps(gaps)

    if time.time() - start_time > deadline - 5:
        # Convert back to rows
        result = []
        for gaps in best_gaps:
            row = [0]
            for g in gaps:
                row.append(row[-1] + g)
            result.append(row)
        return result

    # Phase 2: Stochastic Local Search with gap-swapping moves
    # Objective: minimize max entry while maintaining valid (distinct diffs)
    
    def get_diffs_for_row(gaps, row_idx):
        """Compute diffs for one row given its gaps."""
        pos = [0]
        for g in gaps:
            pos.append(pos[-1] + g)
        diffs = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                diffs.add(pos[j] - pos[i])
        return diffs

    def get_all_diffs(all_gaps):
        """Get set of all differences across all rows."""
        all_d = set()
        for gaps in all_gaps:
            pos = [0]
            for g in gaps:
                pos.append(pos[-1] + g)
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    all_d.add(pos[j] - pos[i])
        return all_d

    current_gaps = [g[:] for g in best_gaps]
    current_max = best_max
    best_gaps_sls = [g[:] for g in best_gaps]
    best_max_sls = best_max
    best_gaps_sls = [g[:] for g in best_gaps]
    best_max_sls = best_max

    # Precompute diffs for current state
    current_all_diffs = get_all_diffs(current_gaps)
    
    # SLS loop with gap-swapping and gap-adjustment moves
    no_improve_count = 0
    max_no_improve = 500
    step = 0
    temperature = 10.0
    
    while time.time() - start_time < deadline - 5:
        step += 1
        if no_improve_count > max_no_improve:
            # Restart with perturbation
            if time.time() - start_time > deadline - 5:
                break
            # Perturb: randomly swap some gaps between rows
            for _ in range(random.randint(2, min(6, n_rows * k // 2))):
                r1 = random.randint(0, n_rows - 1)
                r2 = random.randint(0, n_rows - 1)
                if r1 == r2:
                    continue
                i1 = random.randint(0, k - 1)
                i2 = random.randint(0, k - 1)
                current_gaps[r1][i1], current_gaps[r2][i2] = \
                    current_gaps[r2][i2], current_gaps[r1][i1]
            # Check validity after perturbation
            valid, m = validate_gaps(current_gaps)
            if valid:
                current_max = m
                if m < best_max_sls:
                    best_max_sls = m
                    best_gaps_sls = [g[:] for g in current_gaps]
                no_improve_count = 0
                temperature = 10.0
            else:
                # Revert perturbation
                current_gaps = [g[:] for g in best_gaps_sls]
                current_max = best_max_sls
                no_improve_count = 0
                temperature = 10.0
            current_all_diffs = get_all_diffs(current_gaps)
            continue

        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.6:
            # Gap-swap move: swap a gap between two rows at random positions
            if n_rows < 2:
                # Single row: just try adjusting gaps
                r = 0
                i = random.randint(0, k - 1)
                old_val = current_gaps[r][i]
                # Try small adjustments
                for delta in [1, -1, 2, -2, 3, -3]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    current_gaps[r][i] = new_val
                    # Check if this creates conflicts
                    # Compute new diffs for this row
                    pos = [0]
                    for g in current_gaps[r]:
                        pos.append(pos[-1] + g)
                    new_diffs = set()
                    for ii in range(k + 1):
                        for jj in range(ii + 1, k + 1):
                            new_diffs.add(pos[jj] - pos[ii])
                    # Check against other rows' diffs
                    other_diffs = set()
                    for rr in range(n_rows):
                        if rr == r:
                            continue
                        pos2 = [0]
                        for g in current_gaps[rr]:
                            pos2.append(pos2[-1] + g)
                        for ii in range(k + 1):
                            for jj in range(ii + 1, k + 1):
                                other_diffs.add(pos2[jj] - pos2[ii])
                    # Also check within-row uniqueness is automatic since gaps > 0
                    # But we need all diffs distinct globally
                    if not (new_diffs & other_diffs):
                        # Valid move!
                        new_max = compute_max_entry(current_gaps)
                        if new_max < current_max:
                            current_max = new_max
                            if new_max < best_max_sls:
                                best_max_sls = new_max
                                best_gaps_sls = [g[:] for g in current_gaps]
                                no_improve_count = 0
                        elif new_max <= current_max:
                            no_improve_count = 0
                        else:
                            # Accept with probability (simulated annealing)
                            if random.random() < math.exp(-(new_max - current_max) / temperature):
                                current_max = new_max
                            else:
                                current_gaps[r][i] = old_val
                                continue
                        # Update global diffs
                        current_all_diffs = get_all_diffs(current_gaps)
                        break
                    else:
                        current_gaps[r][i] = old_val
            else:
                r1 = random.randint(0, n_rows - 1)
                r2 = random.randint(0, n_rows - 1)
                if r1 == r2:
                    continue
                i1 = random.randint(0, k - 1)
                i2 = random.randint(0, k - 1)
                
                v1 = current_gaps[r1][i1]
                v2 = current_gaps[r2][i2]
                
                # Perform swap
                current_gaps[r1][i1] = v2
                current_gaps[r2][i2] = v1
                
                # Check validity: compute diffs for affected rows and check conflicts
                # Get diffs for all other rows
                other_diffs = set()
                for rr in range(n_rows):
                    if rr == r1 and rr == r2:
                        continue
                    if rr == r1 or rr == r2:
                        continue
                    pos2 = [0]
                    for g in current_gaps[rr]:
                        pos2.append(pos2[-1] + g)
                    for ii in range(k + 1):
                        for jj in range(ii + 1, k + 1):
                            other_diffs.add(pos2[jj] - pos2[ii])
                
                # Compute diffs for row r1
                pos1 = [0]
                for g in current_gaps[r1]:
                    pos1.append(pos1[-1] + g)
                diffs1 = set()
                for ii in range(k + 1):
                    for jj in range(ii + 1, k + 1):
                        diffs1.add(pos1[jj] - pos1[ii])
                
                # Compute diffs for row r2
                pos2 = [0]
                for g in current_gaps[r2]:
                    pos2.append(pos2[-1] + g)
                diffs2 = set()
                for ii in range(k + 1):
                    for jj in range(ii + 1, k + 1):
                        diffs2.add(pos2[jj] - pos2[ii])
                
                # Check: no diff in diffs1 conflicts with other_diffs or diffs2
                # No diff in diffs2 conflicts with other_diffs or diffs1
                conflict = (diffs1 & other_diffs) or (diffs2 & other_diffs) or (diffs1 & diffs2)
                
                if not conflict:
                    new_max = compute_max_entry(current_gaps)
                    if new_max < current_max:
                        current_max = new_max
                        if new_max < best_max_sls:
                            best_max_sls = new_max
                            best_gaps_sls = [g[:] for g in current_gaps]
                            no_improve_count = 0
                    elif new_max <= current_max:
                        no_improve_count = 0
                    else:
                        if random.random() < math.exp(-(new_max - current_max) / temperature):
                            current_max = new_max
                        else:
                            # Revert
                            current_gaps[r1][i1] = v1
                            current_gaps[r2][i2] = v2
                            no_improve_count += 1
                            temperature *= 0.995
                            continue
                    current_all_diffs = get_all_diffs(current_gaps)
                else:
                    # Revert
                    current_gaps[r1][i1] = v1
                    current_gaps[r2][i2] = v2
                    no_improve_count += 1
                    temperature *= 0.995

        elif move_type < 0.85:
            # Gap-adjustment move: change a single gap value
            r = random.randint(0, n_rows - 1)
            i = random.randint(0, k - 1)
            old_val = current_gaps[r][i]
            # Try small changes
            deltas = [1, -1, 2, -2, 3, -3, 5, -5]
            random.shuffle(deltas)
            moved = False
            for delta in deltas:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                current_gaps[r][i] = new_val
                # Check validity
                pos = [0]
                for g in current_gaps[r]:
                    pos.append(pos[-1] + g)
                new_diffs = set()
                for ii in range(k + 1):
                    for jj in range(ii + 1, k + 1):
                        new_diffs.add(pos[jj] - pos[ii])
                # Check against other rows
                other_diffs = set()
                for rr in range(n_rows):
                    if rr == r:
                        continue
                    pos2 = [0]
                    for g in current_gaps[rr]:
                        pos2.append(pos2[-1] + g)
                    for ii in range(k + 1):
                        for jj in range(ii + 1, k + 1):
                            other_diffs.add(pos2[jj] - pos2[ii])
                if not (new_diffs & other_diffs):
                    new_max = compute_max_entry(current_gaps)
                    if new_max < current_max:
                        current_max = new_max
                        if new_max < best_max_sls:
                            best_max_sls = new_max
                            best_gaps_sls = [g[:] for g in current_gaps]
                            no_improve_count = 0
                    elif new_max <= current_max:
                        no_improve_count = 0
                    else:
                        if random.random() < math.exp(-(new_max - current_max) / temperature):
                            current_max = new_max
                        else:
                            current_gaps[r][i] = old_val
                            continue
                    current_all_diffs = get_all_diffs(current_gaps)
                    moved = True
                    break
                else:
                    current_gaps[r][i] = old_val
            if not moved:
                no_improve_count += 1
                temperature *= 0.995
        else:
            # Row-reconstruction move: rebuild one row greedily within current constraints
            r = random.randint(0, n_rows - 1)
            # Get all diffs from other rows
            other_diffs = set()
            for rr in range(n_rows):
                if rr == r:
                    continue
                pos2 = [0]
                for g in current_gaps[rr]:
                    pos2.append(pos2[-1] + g)
                for ii in range(k + 1):
                    for jj in range(ii + 1, k + 1):
                        other_diffs.add(pos2[jj] - pos2[ii])
            
            # Try to rebuild row r with a smaller max
            target_max = current_max - random.randint(1, 5)
            if target_max < k:  # minimum possible
                target_max = k
            
            row = [0]
            found = True
            while len(row) < k + 1 and found:
                start = row[-1] + 1
                end = min(target_max, start + 50)
                candidates = list(range(start, end + 1))
                random.shuffle(candidates)
                found = False
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & other_diffs):
                        row.append(x)
                        other_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Try wider range
                    for x in range(start, target_max + 1):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & other_diffs):
                            row.append(x)
                            other_diffs |= new_diffs
                            found = True
                            break
            
            if found and len(row) == k + 1:
                new_gaps = [row[i+1] - row[i] for i in range(k)]
                current_gaps[r] = new_gaps
                new_max = compute_max_entry(current_gaps)
                if new_max < current_max:
                    current_max = new_max
                    if new_max < best_max_sls:
                        best_max_sls = new_max
                        best_gaps_sls = [g[:] for g in current_gaps]
                        no_improve_count = 0
                else:
                    # Revert
                    current_gaps[r] = [g for g in best_gaps_sls[r]]
                    no_improve_count += 1
                    temperature *= 0.995
            else:
                no_improve_count += 1
                temperature *= 0.995

        # Cool down temperature slowly
        if step % 100 == 0:
            temperature *= 0.95
            if temperature < 0.5:
                temperature = 0.5

    # Phase 3: High-precision polish pass
    # Try to reduce each gap individually
    polish_gaps = [g[:] for g in best_gaps_sls]
    polish_max = best_max_sls
    
    for _ in range(3):  # Multiple polish sweeps
        if time.time() - start_time > deadline - 2:
            break
        improved = True
        while improved:
            if time.time() - start_time > deadline - 2:
                break
            improved = False
            for r in range(n_rows):
                for i in range(k):
                    if time.time() - start_time > deadline - 2:
                        break
                    old_val = polish_gaps[r][i]
                    # Try reducing
                    for new_val in range(old_val - 1, 0, -1):
                        polish_gaps[r][i] = new_val
                        # Check validity
                        pos = [0]
                        for g in polish_gaps[r]:
                            pos.append(pos[-1] + g)
                        new_diffs = set()
                        for ii in range(k + 1):
                            for jj in range(ii + 1, k + 1):
                                new_diffs.add(pos[jj] - pos[ii])
                        other_diffs = set()
                        for rr in range(n_rows):
                            if rr == r:
                                continue
                            pos2 = [0]
                            for g in polish_gaps[rr]:
                                pos2.append(pos2[-1] + g)
                            for ii in range(k + 1):
                                for jj in range(ii + 1, k + 1):
                                    other_diffs.add(pos2[jj] - pos2[ii])
                        if not (new_diffs & other_diffs):
                            new_max = compute_max_entry(polish_gaps)
                            if new_max < polish_max:
                                polish_max = new_max
                                improved = True
                            break
                        else:
                            continue
                    else:
                        polish_gaps[r][i] = old_val
                        continue
                    break  # Found a reduction, restart sweep

    if polish_max < best_max_sls:
        best_gaps_sls = [g[:] for g in polish_gaps]
        best_max_sls = polish_max

    # Convert best gaps back to rows
    result = []
    for gaps in best_gaps_sls:
        row = [0]
        for g in gaps:
            row.append(row[-1] + g)
        result.append(row)

    return result
# EVOLVE-BLOCK-END
