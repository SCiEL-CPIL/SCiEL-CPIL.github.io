import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # 1. Greedy Initialization
    # We need n_rows rows, each of length k+1, starting at 0.
    # Greedy: pick smallest available integers that don't create duplicate differences.
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def count_conflicts(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        c = Counter(all_diffs)
        conflicts = 0
        for d, cnt in c.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 2. Tightest-Fit Row Constructor
    # Given a set of used differences (from other rows), find the "best" row 
    # that minimizes its own max element while avoiding conflicts with external diffs.
    # This is a greedy algorithm that always picks the smallest valid x.
    
    def construct_tightest_row(used_diffs_from_others, target_max=None):
        if target_max is None:
            # Estimate a reasonable bound based on counting arguments or previous best
            # For n=4, k=7, scope ~134. Let's allow up to 200 initially.
            target_max = 200 
            
        row = [0]
        current_max = 0
        current_row_diffs = set()
        
        while len(row) < k + 1:
            found = False
            x = current_max + 1
            while x <= target_max:
                # Check against external used diffs
                conflict_ext = False
                # Check against internal diffs
                conflict_int = False
                
                for a in row:
                    d = x - a
                    if d in used_diffs_from_others:
                        conflict_ext = True
                        break
                    if d in current_row_diffs:
                        conflict_int = True
                        break
                
                if not conflict_ext and not conflict_int:
                    row.append(x)
                    current_row_diffs.update(x - a for a in row[:-1])
                    current_max = x
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        
        return row

    # 3. Conflict-Driven Local Search (CDLS)
    # Strategy:
    # 1. Initialize with greedy.
    # 2. Identify all conflicts (differences appearing > 1 time).
    # 3. Pick the "most frequent" or "most problematic" conflict.
    # 4. Identify a row involved in this conflict that has the largest scope or is "loosest".
    # 5. Reconstruct that row using construct_tightest_row, treating all other rows' diffs as forbidden.
    # 6. If the new row is valid (no new conflicts introduced with other rows, or fewer conflicts), accept.
    # 7. To escape local minima, occasionally allow "soft" conflicts or randomize the selection of which row to fix.
    # 8. Since we want to minimize scope, we prioritize rows with large max elements for reconstruction.
    
    pop_rows = greedy_init()
    best_rows = [r[:] for r in pop_rows]
    best_conflicts = count_conflicts(best_rows)
    best_scope = get_scope(best_rows)
    
    # Track history to avoid cycling
    history = set()
    
    def state_key(rows):
        return tuple(tuple(r) for r in rows)
    
    def evaluate(rows):
        conf = count_conflicts(rows)
        scope = get_scope(rows)
        return conf, scope, conf * 10000 + scope

    current_conf, current_scope, current_score = evaluate(pop_rows)
    
    # Main Loop
    while time.time() < deadline:
        # 1. Analyze Conflicts
        all_diffs = []
        diff_locations = {} # diff -> list of (row_idx, i, j)
        
        for r_idx, r in enumerate(pop_rows):
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    all_diffs.append(d)
                    if d not in diff_locations:
                        diff_locations[d] = []
                    diff_locations[d].append((r_idx, i, j))
        
        conflicts = [d for d, locs in diff_locations.items() if len(locs) > 1]
        
        if not conflicts:
            # No conflicts! We have a valid solution.
            # Now we just want to minimize scope.
            # Try to tighten any row by reconstructing it with a tighter bound.
            # Or just stop if we are at the counting bound?
            # For now, try to reduce scope by reconstructing the row with the largest max.
            
            # Find row with largest max
            max_row_idx = max(range(n_rows), key=lambda i: pop_rows[i][-1])
            
            # Try to reconstruct this row with a slightly smaller target
            current_max = pop_rows[max_row_idx][-1]
            
            # Calculate used diffs from other rows
            used_diffs_from_others = set()
            for j in range(n_rows):
                if j == max_row_idx:
                    continue
                used_diffs_from_others.update(get_diffs(pop_rows[j]))
            
            # Try to construct a tighter row
            # We can try target_max = current_max - 1, or just let it be tightest fit
            # Since it was already tightest fit against the *previous* state, 
            # and the other rows haven't changed, it might not get smaller.
            # However, if we change *another* row first, this row might become tighter.
            
            # Let's try to improve scope by random perturbation + reconstruction of a random row
            # to break symmetry and potentially allow a tighter packing.
            
            if current_scope <= 112: # Counting bound
                break
                
            # Pick a random row to perturb and reconstruct
            r_idx = random.randint(0, n_rows - 1)
            
            # Perturb the row slightly to escape local optima
            row = pop_rows[r_idx][:]
            # Randomly change one element
            pos = random.randint(1, k-1)
            old_val = row[pos]
            # Try a few random values
            improved = False
            for _ in range(10):
                new_val = random.randint(row[pos-1]+1, row[pos+1]-1 if pos < k else row[pos]+5)
                if new_val == old_val: continue
                row[pos] = new_val
                # Fix structure
                for j in range(pos+1, len(row)):
                    if row[j] <= row[j-1]:
                        row[j] = row[j-1] + 1
                
                # Reconstruct this row tightly
                used_diffs_from_others = set()
                for j in range(n_rows):
                    if j == r_idx:
                        continue
                    used_diffs_from_others.update(get_diffs(pop_rows[j]))
                
                new_row = construct_tightest_row(used_diffs_from_others, target_max=current_scope)
                
                if new_row:
                    # Check if this reduces scope or keeps it same but reduces "potential"
                    test_rows = [r[:] for r in pop_rows]
                    test_rows[r_idx] = new_row
                    test_conf, test_scope, test_score = evaluate(test_rows)
                    
                    if test_conf <= current_conf and test_scope <= current_scope:
                        if test_conf < current_conf or test_scope < current_scope:
                            pop_rows = test_rows
                            current_conf, current_scope, current_score = evaluate(pop_rows)
                            if current_conf < best_conflicts or (current_conf == best_conflicts and current_scope < best_scope):
                                best_rows = [r[:] for r in pop_rows]
                                best_conflicts = current_conf
                                best_scope = current_scope
                            improved = True
                            break
            
            if not improved:
                # If no improvement, try a different strategy:
                # Reconstruct the "worst" row (largest max) tightly
                max_row_idx = max(range(n_rows), key=lambda i: pop_rows[i][-1])
                used_diffs_from_others = set()
                for j in range(n_rows):
                    if j == max_row_idx:
                        continue
                    used_diffs_from_others.update(get_diffs(pop_rows[j]))
                
                new_row = construct_tightest_row(used_diffs_from_others, target_max=current_scope)
                if new_row:
                    test_rows = [r[:] for r in pop_rows]
                    test_rows[max_row_idx] = new_row
                    test_conf, test_scope, test_score = evaluate(test_rows)
                    if test_conf <= current_conf and test_scope <= current_scope:
                        pop_rows = test_rows
                        current_conf, current_scope, current_score = evaluate(pop_rows)
                        if current_conf < best_conflicts or (current_conf == best_conflicts and current_scope < best_scope):
                            best_rows = [r[:] for r in pop_rows]
                            best_conflicts = current_conf
                            best_scope = current_scope
                            
            continue

        # 2. Select a Conflict to Resolve
        # Prioritize conflicts with the highest frequency
        conflict_freq = Counter(conflicts)
        # Sort by frequency, then by value (smaller diffs are often more constrained?)
        # Actually, let's pick the conflict that involves the row with the largest max,
        # as fixing that row might reduce scope significantly.
        
        # Group conflicts by row
        rows_involved = {}
        for d in conflicts:
            for loc in diff_locations[d]:
                r_idx = loc[0]
                if r_idx not in rows_involved:
                    rows_involved[r_idx] = 0
                rows_involved[r_idx] += 1
        
        # Pick the row involved in the most conflicts, tie-break by largest max
        target_row_idx = max(rows_involved.keys(), key=lambda i: (rows_involved[i], pop_rows[i][-1]))
        
        # 3. Reconstruct the Target Row
        used_diffs_from_others = set()
        for j in range(n_rows):
            if j == target_row_idx:
                continue
            used_diffs_from_others.update(get_diffs(pop_rows[j]))
        
        # To allow escape from local minima, we can sometimes ignore a few "soft" conflicts
        # But for now, let's try strict reconstruction first.
        # If strict fails or doesn't improve, we might need to perturb other rows.
        
        new_row = construct_tightest_row(used_diffs_from_others, target_max=current_scope)
        
        if new_row:
            test_rows = [r[:] for r in pop_rows]
            test_rows[target_row_idx] = new_row
            test_conf, test_scope, test_score = evaluate(test_rows)
            
            # Accept if it reduces conflicts, or keeps conflicts same but reduces scope
            if test_conf < current_conf:
                pop_rows = test_rows
                current_conf, current_scope, current_score = evaluate(pop_rows)
                if current_conf < best_conflicts or (current_conf == best_conflicts and current_scope < best_scope):
                    best_rows = [r[:] for r in pop_rows]
                    best_conflicts = current_conf
                    best_scope = current_scope
            elif test_conf == current_conf and test_scope < current_scope:
                pop_rows = test_rows
                current_conf, current_scope, current_score = evaluate(pop_rows)
                if current_conf < best_conflicts or (current_conf == best_conflicts and current_scope < best_scope):
                    best_rows = [r[:] for r in pop_rows]
                    best_conflicts = current_conf
                    best_scope = current_scope
            else:
                # No improvement. Try a randomized perturbation of the target row before reconstruction.
                # This is a "kick" to escape local minima.
                row = pop_rows[target_row_idx][:]
                pos = random.randint(1, k-1)
                old_val = row[pos]
                
                # Try a few random perturbations
                found_improvement = False
                for _ in range(20):
                    new_val = random.randint(row[pos-1]+1, row[pos+1]-1 if pos < k else row[pos]+10)
                    if new_val == old_val: continue
                    row[pos] = new_val
                    # Fix structure
                    for j in range(pos+1, len(row)):
                        if row[j] <= row[j-1]:
                            row[j] = row[j-1] + 1
                    
                    # Reconstruct this perturbed row tightly
                    # Note: We use the same used_diffs_from_others, which is fine for the "others"
                    # But the perturbation might have created internal conflicts or conflicts with others.
                    # construct_tightest_row ensures no internal conflicts and no conflicts with "others".
                    # So it effectively "heals" the perturbation.
                    
                    new_row = construct_tightest_row(used_diffs_from_others, target_max=current_scope)
                    if new_row:
                        test_rows = [r[:] for r in pop_rows]
                        test_rows[target_row_idx] = new_row
                        test_conf, test_scope, test_score = evaluate(test_rows)
                        
                        if test_conf < current_conf or (test_conf == current_conf and test_scope < current_scope):
                            pop_rows = test_rows
                            current_conf, current_scope, current_score = evaluate(pop_rows)
                            if current_conf < best_conflicts or (current_conf == best_conflicts and current_scope < best_scope):
                                best_rows = [r[:] for r in pop_rows]
                                best_conflicts = current_conf
                                best_scope = current_scope
                            found_improvement = True
                            break
        
    return best_rows
