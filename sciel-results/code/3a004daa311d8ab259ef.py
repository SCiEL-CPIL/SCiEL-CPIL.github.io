import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Total number of distinct differences needed
    total_diffs = n_rows * (k + 1) * k // 2
    
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    if r[j] <= r[i]:
                        return False
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def greedy_row(used_diffs, max_val, rng, skip_prob=0.0):
        """
        Build a single row of length k+1 starting at 0, with all differences
        not in used_diffs, and max value <= max_val.
        """
        row = [0]
        local_diffs = set()
        while len(row) < k + 1:
            x = row[-1] + 1
            found = False
            while x <= max_val:
                if rng.random() < skip_prob and x < max_val:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    local_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        return row
    
    def greedy_construct(M_limit, seed_val=0, skip_prob=0.0):
        """
        Greedy construction: build rows one by one.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = greedy_row(used_diffs, M_limit, rng, skip_prob)
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        return rows
    
    def interleave_greedy(M_limit, seed_val=0, skip_prob=0.0):
        """
        Interleaved greedy: extend the row with the smallest current maximum at each step.
        Maintains local diffs incrementally for speed.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[0] for _ in range(n_rows)]
        local_diffs = [set() for _ in range(n_rows)]
        
        for _ in range(n_rows * k):
            candidates = [i for i in range(n_rows) if len(rows[i]) < k + 1]
            if not candidates:
                break
            
            target_row_idx = min(candidates, key=lambda idx: rows[idx][-1])
            row = rows[target_row_idx]
            ld = local_diffs[target_row_idx]
            
            x = row[-1] + 1
            found = False
            while x <= M_limit:
                if rng.random() < skip_prob and x < max_val if False else x < M_limit:
                    x += 1
                    continue
                
                bad = False
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs or d in ld:
                        bad = True
                        break
                    new_diffs.append(d)
                
                if not bad:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    ld.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
        
        for r in rows:
            if len(r) != k + 1:
                return None
        
        return rows
    
    def simulated_annealing(rows, time_budget, seed_val=0):
        """
        Simulated annealing to reduce the scope.
        Moves: replace one row with a new valid row that fits in the current scope.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used = get_used_diffs(current)
        deadline_sa = time.time() + time_budget
        T = 30.0
        cooling = 0.995
        step = 0
        
        while time.time() < deadline_sa and T > 0.1:
            step += 1
            
            # Pick a random row to replace
            ri = rng.randint(0, n_rows - 1)
            old_row = current[ri]
            old_diffs = get_row_diffs(old_row)
            
            used_without = used - old_diffs
            
            # Try to build a new row within current_scope
            max_mark = current_scope
            new_row = None
            
            # Try several randomizations of greedy
            for attempt in range(30):
                new_row = greedy_row(used_without, max_mark, rng, skip_prob=0.3)
                if new_row is not None:
                    break
            
            if new_row is None:
                T *= cooling
                continue
            
            new_diffs = get_row_diffs(new_row)
            
            if new_diffs & used_without:
                T *= cooling
                continue
            
            # Compute new scope
            other_max = 0
            for i, r in enumerate(current):
                if i != ri:
                    other_max = max(other_max, max(r))
            new_scope = max(other_max, max(new_row))
            
            delta = new_scope - current_scope
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used = used_without | new_diffs
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
            
            T *= cooling
        
        return best, best_scope
    
    def local_search(rows, time_budget, seed_val=0):
        """
        Local search: try to reduce the scope by adjusting individual elements.
        Focus on reducing the largest element.
        """
        if rows is None:
            return None, None
        
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = max(max(r) for r in best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used = get_used_diffs(current)
        deadline_ls = time.time() + time_budget
        
        while time.time() < deadline_ls:
            # Find the row with the maximum element
            max_val = -1
            max_row_idx = -1
            for i, r in enumerate(current):
                if max(r) > max_val:
                    max_val = max(r)
                    max_row_idx = i
            
            if max_val < current_scope:
                break
            
            ri = max_row_idx
            # Find the position of the maximum in this row
            pi = current[ri].index(max_val)
            
            old_val = current[ri][pi]
            old_diffs = get_row_diffs(current[ri])
            
            used_without = used - old_diffs
            
            # Try to reduce this value
            found = False
            for new_val in range(old_val - 1, current[ri][pi - 1], -1):
                # Check validity
                bad = False
                new_diffs = set()
                for i in range(len(current[ri])):
                    if i == pi:
                        continue
                    if i < pi:
                        d = new_val - current[ri][i]
                        if d <= 0 or d in used_without:
                            bad = True
                            break
                        new_diffs.add(d)
                    else:
                        d = current[ri][i] - new_val
                        if d <= 0 or d in used_without:
                            bad = True
                            break
                        new_diffs.add(d)
                
                if not bad:
                    # Check against other elements in the same row
                    for i in range(len(current[ri])):
                        if i == pi:
                            continue
                        if i < pi:
                            d = new_val - current[ri][i]
                            # This difference should not already be in new_diffs from a different pair
                            # Actually, we need to check that all pairwise differences in the new row are distinct
                            # and not in used_without
                            pass
                    
                    # Full check: build the new row and verify all differences
                    new_row = current[ri][:]
                    new_row[pi] = new_val
                    all_diffs = get_row_diffs(new_row)
                    
                    # Check that all differences are distinct and not in used_without
                    if len(all_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                        if not (all_diffs & used_without):
                            # Accept
                            current[ri][pi] = new_val
                            used = used_without | all_diffs
                            current_scope = max(max(r) for r in current)
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best = [r[:] for r in current]
                            found = True
                            break
                # continue to try smaller values
        
            if not found:
                # Try to reduce a non-maximum element to free up space
                # Pick a random position in the row with the maximum
                pi2 = rng.randint(1, k - 1)
                old_val2 = current[ri][pi2]
                old_diffs2 = get_row_diffs(current[ri])
                used_without2 = used - old_diffs2
                
                found2 = False
                for new_val2 in range(old_val2 - 1, current[ri][pi2 - 1], -1):
                    bad = False
                    new_diffs2 = set()
                    for i in range(len(current[ri])):
                        if i == pi2:
                            continue
                        if i < pi2:
                            d = new_val2 - current[ri][i]
                            if d <= 0 or d in used_without2:
                                bad = True
                                break
                            new_diffs2.add(d)
                        else:
                            d = current[ri][i] - new_val2
                            if d <= 0 or d in used_without2:
                                bad = True
                                break
                            new_diffs2.add(d)
                    
                    if not bad:
                        new_row = current[ri][:]
                        new_row[pi2] = new_val2
                        all_diffs2 = get_row_diffs(new_row)
                        
                        if len(all_diffs2) == len(new_row) * (len(new_row) - 1) // 2:
                            if not (all_diffs2 & used_without2):
                                current[ri][pi2] = new_val2
                                used = used_without2 | all_diffs2
                                current_scope = max(max(r) for r in current)
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best = [r[:] for r in current]
                                found2 = True
                                break
                
                if not found2:
                    # If we can't reduce any element, try a random move
                    # to escape local minimum
                    ri_rand = rng.randint(0, n_rows - 1)
                    pi_rand = rng.randint(1, k)
                    old_val_rand = current[ri_rand][pi_rand]
                    old_diffs_rand = get_row_diffs(current[ri_rand])
                    used_without_rand = used - old_diffs_rand
                    
                    # Try a small random change
                    for delta_rand in [-2, -1, 1, 2]:
                        new_val_rand = old_val_rand + delta_rand
                        if new_val_rand <= current[ri_rand][pi_rand - 1] or new_val_rand >= current[ri_rand][pi_rand + 1]:
                            continue
                        
                        bad = False
                        new_row = current[ri_rand][:]
                        new_row[pi_rand] = new_val_rand
                        all_diffs_rand = get_row_diffs(new_row)
                        
                        if len(all_diffs_rand) == len(new_row) * (len(new_row) - 1) // 2:
                            if not (all_diffs_rand & used_without_rand):
                                current[ri_rand][pi_rand] = new_val_rand
                                used = used_without_rand | all_diffs_rand
                                current_scope = max(max(r) for r in current)
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best = [r[:] for r in current]
                                break
        
        return best, best_scope
    
    # Main logic
    best_rows = None
    best_M = float('inf')
    
    # Phase 1: Get a baseline with greedy
    start_M = total_diffs
    end_M = total_diffs + 60
    
    for M in range(start_M, end_M):
        if time.time() > deadline - 40:
            break
        
        for seed in range(8):
            if time.time() > deadline - 35:
                break
            
            if seed % 3 == 0:
                rows = greedy_construct(M, seed, skip_prob=0.0)
            elif seed % 3 == 1:
                rows = interleave_greedy(M, seed, skip_prob=0.0)
            else:
                rows = greedy_construct(M, seed, skip_prob=0.1)
            
            if rows is not None and is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                break
        
        if best_rows is not None and best_M <= M:
            break
    
    # If no baseline found, try larger M
    if best_rows is None:
        for M in range(end_M, end_M + 150):
            if time.time() > deadline - 10:
                break
            rows = greedy_construct(M, 0, skip_prob=0.0)
            if rows is not None and is_valid(rows):
                best_M = max(max(r) for r in rows)
                best_rows = [r[:] for r in rows]
                break
    
    # Emergency fallback
    if best_rows is None:
        M_huge = 2000
        rows = greedy_construct(M_huge, 0, skip_prob=0.0)
        if rows is not None:
            best_rows = rows
            best_M = max(max(r) for r in rows)
        else:
            # Construct a valid solution with large gaps
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 500
                    row.append(base)
                best_rows.append(row)
            best_M = max(max(r) for r in best_rows)
    
    # Phase 2: Improve with Simulated Annealing
    time_left = deadline - time.time()
    if time_left > 10:
        sa_budget = min(time_left - 10, 50)
        if sa_budget > 0:
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=42)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows
        
        # If time remains, try another SA run
        time_left = deadline - time.time()
        if time_left > 10:
            sa_budget2 = min(time_left - 10, 30)
            if sa_budget2 > 0:
                improved_rows2, improved_M2 = simulated_annealing(best_rows, sa_budget2, seed_val=123)
                if improved_rows2 is not None and improved_M2 < best_M:
                    best_M = improved_M2
                    best_rows = improved_rows2
    
    # Phase 3: Local search for final refinement
    time_left = deadline - time.time()
    if time_left > 5:
        ls_budget = min(time_left - 5, 15)
        if ls_budget > 0:
            improved_rows3, improved_M3 = local_search(best_rows, ls_budget, seed_val=777)
            if improved_rows3 is not None and improved_M3 < best_M:
                best_M = improved_M3
                best_rows = improved_rows3
    
    return best_rows

# Test / Main guard
if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", max(max(r) for r in rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
