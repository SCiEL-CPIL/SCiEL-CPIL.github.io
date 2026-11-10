import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Difference triangle set construction combining:
    - Global MRV backtracking (from inspiration) for tight initializations
    - Simulated annealing with multi-element moves (from current) for refinement
    - Local search for final polish
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_initial(n_rows, k, rng=None, perturb=0):
        """
        Greedy initialization with optional randomization.
        """
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                candidates = []
                search_limit = x + 60 + (perturb * 15)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 25:
                        return None

                if not candidates:
                    x = search_limit
                    continue

                if rng is not None and len(candidates) > 1:
                    top = min(4, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def mrv_backtracking(n_rows, k, random_seed, max_time=30.0):
        """
        MRV (Minimum Remaining Values) backtracking search.
        At each step, pick the row with the smallest next valid value.
        This is the structural idea from the inspiration.
        """
        start = time.time()
        rng = random.Random(random_seed)
        
        # State
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        total_steps = n_rows * k
        steps_done = 0
        
        # For MRV, we track for each row the smallest valid next value
        # We use a priority queue approach: at each step, find the row
        # with the smallest feasible next value and place it there.
        
        # To add diversity, we sometimes pick from top few candidates
        diversity = 0.3  # probability of picking non-minimal candidate
        
        while steps_done < total_steps:
            if time.time() - start > max_time:
                return None
            
            # Find the next valid value for each incomplete row
            row_candidates = []
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                
                current_row = rows[r]
                last_val = current_row[-1]
                
                # Find the smallest few valid x
                valid_xs = []
                x = last_val + 1
                checks = 0
                while len(valid_xs) < 5 and checks < 200:
                    # Quick check: x itself must not be in used_diffs
                    if x in used_diffs:
                        x += 1
                        checks += 1
                        continue
                    
                    # Check all differences
                    conflict = False
                    for a in current_row[1:]:
                        if (x - a) in used_diffs:
                            conflict = True
                            break
                    
                    if not conflict:
                        valid_xs.append(x)
                    
                    x += 1
                    checks += 1
                
                if not valid_xs:
                    # Dead end - try to find any valid x by expanding search
                    x = last_val + 1
                    checks = 0
                    while checks < 500:
                        if x in used_diffs:
                            x += 1
                            checks += 1
                            continue
                        conflict = False
                        for a in current_row[1:]:
                            if (x - a) in used_diffs:
                                conflict = True
                                break
                        if not conflict:
                            valid_xs.append(x)
                            break
                        x += 1
                        checks += 1
                    if not valid_xs:
                        return None
                
                row_candidates.append((valid_xs[0], r, valid_xs))
            
            if not row_candidates:
                break
            
            # Sort by smallest valid value (MRV heuristic)
            row_candidates.sort(key=lambda x: x[0])
            
            # Pick a row: usually the most constrained (smallest min value)
            # With some probability, pick from top few for diversity
            if rng.random() < diversity and len(row_candidates) > 1:
                top_n = min(3, len(row_candidates))
                idx = rng.randint(0, top_n - 1)
            else:
                idx = 0
            
            min_x, r, valid_xs = row_candidates[idx]
            
            # Choose which value to place
            if rng.random() < diversity and len(valid_xs) > 1:
                chosen_x = valid_xs[rng.randint(0, min(2, len(valid_xs) - 1))]
            else:
                chosen_x = valid_xs[0]
            
            # Place the value
            rows[r].append(chosen_x)
            # Update used_diffs
            for a in rows[r][:-1]:
                used_diffs.add(chosen_x - a)
            
            steps_done += 1
        
        # Validate
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return None
                    all_diffs.add(d)
        
        return rows

    def try_move_element(rows, ri, ci, new_val, used_diffs):
        """
        Try to move element (ri, ci) to new_val.
        Returns (success, new_used_diffs) if successful, (False, None) otherwise.
        """
        row = rows[ri]
        current = row[ci]

        if ci > 0 and new_val <= row[ci - 1]:
            return False, None
        if ci < len(row) - 1 and new_val >= row[ci + 1]:
            return False, None

        old_diffs = set()
        for i in range(ci):
            old_diffs.add(row[ci] - row[i])
        for j in range(ci + 1, len(row)):
            old_diffs.add(row[j] - row[ci])

        temp_used = used_diffs - old_diffs

        new_diffs = set()
        valid = True
        for i in range(ci):
            d = new_val - row[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci + 1, len(row)):
                d = row[j] - new_val
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)

        if not valid:
            return False, None

        new_used = temp_used | new_diffs
        return True, new_used

    def try_swap_elements(rows, ri1, ci1, ri2, ci2, used_diffs):
        """
        Try to swap elements (ri1, ci1) and (ri2, ci2).
        Returns (success, new_used_diffs) if successful.
        """
        if ri1 == ri2:
            if ci1 > ci2:
                ri1, ri2 = ri2, ri1
                ci1, ci2 = ci2, ci1
            
            row = rows[ri1]
            v1 = row[ci1]
            v2 = row[ci2]
            
            if v1 == v2:
                return True, used_diffs
            
            new_row = row[:]
            new_row[ci1] = v2
            new_row[ci2] = v1
            
            if new_row != sorted(new_row):
                return False, None
            
            old_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if i == ci1 or i == ci2 or j == ci1 or j == ci2:
                        old_diffs.add(row[j] - row[i])
            
            temp_used = used_diffs - old_diffs
            
            new_diffs = set()
            valid = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    if i == ci1 or i == ci2 or j == ci1 or j == ci2:
                        d = new_row[j] - new_row[i]
                        if d <= 0 or d in temp_used:
                            valid = False
                            break
                        new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                return False, None
            
            new_used = temp_used | new_diffs
            return True, new_used
        
        row1 = rows[ri1]
        row2 = rows[ri2]
        v1 = row1[ci1]
        v2 = row2[ci2]
        
        if v1 == v2:
            return True, used_diffs
        
        if ci1 > 0 and v2 <= row1[ci1 - 1]:
            return False, None
        if ci1 < len(row1) - 1 and v2 >= row1[ci1 + 1]:
            return False, None
        if ci2 > 0 and v1 <= row2[ci2 - 1]:
            return False, None
        if ci2 < len(row2) - 1 and v1 >= row2[ci2 + 1]:
            return False, None
        
        old_diffs = set()
        for i in range(ci1):
            old_diffs.add(v1 - row1[i])
        for j in range(ci1 + 1, len(row1)):
            old_diffs.add(row1[j] - v1)
        
        for i in range(ci2):
            old_diffs.add(v2 - row2[i])
        for j in range(ci2 + 1, len(row2)):
            old_diffs.add(row2[j] - v2)
        
        temp_used = used_diffs - old_diffs
        
        new_diffs = set()
        valid = True
        for i in range(ci1):
            d = v2 - row1[i]
            if d <= 0 or d in temp_used:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            for j in range(ci1 + 1, len(row1)):
                d = row1[j] - v2
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        
        if valid:
            for i in range(ci2):
                d = v1 - row2[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        if valid:
            for j in range(ci2 + 1, len(row2)):
                d = row2[j] - v1
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
        
        if not valid:
            return False, None
        
        new_used = temp_used | new_diffs
        return True, new_used

    def simulated_annealing(rows, n_rows, k, max_iters=8000, t_start=50.0, t_end=1.0):
        """
        Simulated annealing with multiple move types.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_rows = [row[:] for row in rows]
        best_score = max(row[-1] for row in rows)
        current_score = best_score
        
        t = t_start
        cooling = (t_start - t_end) / max(1, max_iters)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            t = max(t_end, t - cooling)
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Single element move: try to reduce a random element
                ri = random.randint(0, n_rows - 1)
                ci = random.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                
                for delta in [1, 2, 3, 5, 8, 13]:
                    new_val = current_val - delta
                    if new_val <= 0:
                        continue
                    if ci > 0 and new_val <= rows[ri][ci - 1]:
                        continue
                    if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                        continue
                    
                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if success:
                        new_score = max(r[-1] for r in rows)
                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / t):
                            rows[ri][ci] = new_val
                            used_diffs = new_used
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [row[:] for row in rows]
                        break
            elif move_type < 0.6:
                # Single element move: try to increase
                ri = random.randint(0, n_rows - 1)
                ci = random.randint(1, len(rows[ri]) - 1)
                current_val = rows[ri][ci]
                
                for delta in [1, 2, 3, 5]:
                    new_val = current_val + delta
                    if ci < len(rows[ri]) - 1 and new_val >= rows[ri][ci + 1]:
                        continue
                    
                    success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                    if success:
                        new_score = max(r[-1] for r in rows)
                        delta_score = new_score - current_score
                        if delta_score <= 0 or random.random() < math.exp(-delta_score / t):
                            rows[ri][ci] = new_val
                            used_diffs = new_used
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_rows = [row[:] for row in rows]
                        break
            elif move_type < 0.8:
                # Swap two elements in different rows
                ri1 = random.randint(0, n_rows - 1)
                ri2 = random.randint(0, n_rows - 1)
                if ri1 == ri2:
                    continue
                ci1 = random.randint(1, len(rows[ri1]) - 1)
                ci2 = random.randint(1, len(rows[ri2]) - 1)
                
                success, new_used = try_swap_elements(rows, ri1, ci1, ri2, ci2, used_diffs)
                if success:
                    v1 = rows[ri1][ci1]
                    v2 = rows[ri2][ci2]
                    rows[ri1][ci1] = v2
                    rows[ri2][ci2] = v1
                    
                    new_score = max(r[-1] for r in rows)
                    delta_score = new_score - current_score
                    if delta_score <= 0 or random.random() < math.exp(-delta_score / t):
                        used_diffs = new_used
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [row[:] for row in rows]
                    else:
                        rows[ri1][ci1] = v1
                        rows[ri2][ci2] = v2
            else:
                # Shift a row by a small offset
                ri = random.randint(0, n_rows - 1)
                row = rows[ri]
                if len(row) <= 1:
                    continue
                
                offset = random.choice([-3, -2, -1, 1, 2, 3])
                new_row = [row[0]] + [v + offset for v in row[1:]]
                
                if new_row != sorted(new_row):
                    continue
                if any(v <= 0 for v in new_row[1:]):
                    continue
                
                old_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        old_diffs.add(row[j] - row[i])
                
                temp_used = used_diffs - old_diffs
                
                new_diffs = set()
                valid = True
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d <= 0 or d in temp_used:
                            valid = False
                            break
                        new_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_used = temp_used | new_diffs
                temp_rows = [r[:] for r in rows]
                temp_rows[ri] = new_row
                new_score = max(r[-1] for r in temp_rows)
                
                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / t):
                    rows[ri] = new_row
                    used_diffs = new_used
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_rows = [row[:] for row in rows]
        
        return best_rows, best_score

    def local_search_reduce_max(rows, n_rows, k, max_iters=3000):
        """
        Local search to reduce the maximum entry.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        best_score = max(row[-1] for row in rows)
        no_improve_count = 0
        max_no_improve = 100

        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            if no_improve_count > max_no_improve:
                break

            max_row_idx = max(range(n_rows), key=lambda r: rows[r][-1])
            max_val = rows[max_row_idx][-1]
            row = rows[max_row_idx]
            
            found_improvement = False

            ci = len(row) - 1
            for new_val in range(max_val - 1, row[ci - 1], -1):
                success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                if success:
                    rows[max_row_idx][ci] = new_val
                    used_diffs = new_used
                    new_score = max(r[-1] for r in rows)
                    if new_score < best_score:
                        best_score = new_score
                    found_improvement = True
                    break

            if not found_improvement:
                for ci in range(len(row) - 2, 0, -1):
                    current = row[ci]
                    for new_val in range(current - 1, row[ci - 1], -1):
                        success, new_used = try_move_element(rows, max_row_idx, ci, new_val, used_diffs)
                        if success:
                            rows[max_row_idx][ci] = new_val
                            used_diffs = new_used
                            new_score = max(r[-1] for r in rows)
                            if new_score < best_score:
                                best_score = new_score
                            found_improvement = True
                            break
                    if found_improvement:
                        break

            if not found_improvement:
                for ri in range(n_rows):
                    if ri == max_row_idx:
                        continue
                    row = rows[ri]
                    for ci in range(len(row) - 1, 0, -1):
                        current = row[ci]
                        for new_val in range(current - 1, row[ci - 1], -1):
                            success, new_used = try_move_element(rows, ri, ci, new_val, used_diffs)
                            if success:
                                rows[ri][ci] = new_val
                                used_diffs = new_used
                                new_score = max(r[-1] for r in rows)
                                if new_score < best_score:
                                    best_score = new_score
                                found_improvement = True
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break

            if found_improvement:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return rows, best_score

    # Phase 1: Multiple initializations
    best_rows = None
    best_score = float('inf')
    
    # Deterministic greedy
    rows = greedy_initial(n_rows, k, rng=None)
    if rows is not None:
        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]
    
    # MRV backtracking with multiple seeds (structural idea from inspiration)
    n_mrv_restarts = 20
    for restart in range(n_mrv_restarts):
        if time_remaining() < 50:
            break
        
        rows = mrv_backtracking(n_rows, k, random_seed=1000 + restart, max_time=5.0)
        if rows is None:
            continue
        
        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]
    
    # Randomized greedy with multiple seeds
    n_restarts = 20
    for restart in range(n_restarts):
        if time_remaining() < 40:
            break
        
        rng = random.Random(2000 + restart)
        rows = greedy_initial(n_rows, k, rng=rng, perturb=restart % 5)
        if rows is None:
            continue
        
        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]
    
    if best_rows is None:
        # Fallback
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows

    # Phase 2: Local search on best solution
    improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=3000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    # Phase 3: Simulated annealing
    if time_remaining() > 30:
        sa_rows, sa_score = simulated_annealing(best_rows, n_rows, k, max_iters=5000, t_start=40.0, t_end=1.0)
        if sa_score < best_score:
            best_score = sa_score
            best_rows = sa_rows

    # Phase 4: More local search
    if time_remaining() > 15:
        improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=2000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    # Phase 5: Another SA round with different parameters
    if time_remaining() > 20:
        sa_rows, sa_score = simulated_annealing(best_rows, n_rows, k, max_iters=3000, t_start=20.0, t_end=0.5)
        if sa_score < best_score:
            best_score = sa_score
            best_rows = sa_rows

    # Phase 6: Final polish
    if time_remaining() > 10:
        improved_rows, improved_score = local_search_reduce_max(best_rows, n_rows, k, max_iters=1000)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = improved_rows

    return best_rows
# EVOLVE-BLOCK-END
