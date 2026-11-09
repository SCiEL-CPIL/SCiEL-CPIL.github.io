import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Total number of distinct differences needed
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs  # Simple counting bound

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        used = set()
        for r in rows:
            if len(r) != k + 1 or r[0] != 0:
                return False
            # Check strictly increasing
            for i in range(1, k + 1):
                if r[i] <= r[i-1]:
                    return False
            # Check distinct differences
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def get_used_diffs(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used.add(r[j] - r[i])
        return used

    def greedy_construct(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                next_val = row[-1] + 1
                valid_found = False
                while next_val <= M_limit:
                    new_diffs = set()
                    valid = True
                    for prev in row:
                        d = next_val - prev
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Check internal consistency
                        curr_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                curr_diffs.add(row[j] - row[i])
                        
                        if new_diffs & curr_diffs:
                            valid = False
                        
                        if valid:
                            row.append(next_val)
                            used_diffs.update(new_diffs)
                            valid_found = True
                            break
                    next_val += 1
                
                if not valid_found:
                    return None
            rows.append(row)
        return rows

    def simulated_annealing(rows, time_budget, seed_val=0):
        if rows is None:
            return None, None
            
        rng = random.Random(seed_val)
        best = [r[:] for r in rows]
        best_scope = get_scope(best)
        current = [r[:] for r in best]
        current_scope = best_scope
        
        used_diffs = get_used_diffs(current)
        
        T = 10.0
        cooling = 0.995
        deadline = time.time() + time_budget
        
        def compute_row_diffs(row):
            diffs = set()
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.add(row[j] - row[i])
            return diffs

        step = 0
        while time.time() < deadline and T > 0.1:
            step += 1
            
            # Pick a random row
            ri = rng.randint(0, n_rows - 1)
            row = current[ri]
            old_diffs = compute_row_diffs(row)
            used_without = used_diffs - old_diffs
            
            # Pick a random element index to change (1 to k)
            # Or try to change the last element to reduce scope
            # Let's try changing a random element
            if rng.random() < 0.5:
                # Target the last element to reduce scope
                idx = k
            else:
                idx = rng.randint(1, k)
            
            old_val = row[idx]
            
            # Determine range for new value
            lower_bound_val = row[idx-1] + 1
            upper_bound_val = row[idx+1] - 1 if idx < k else current_scope
            
            # If we are trying to reduce scope, bias towards lower values
            # But we must maintain validity.
            
            new_val = None
            # Try a few random candidates in the valid range
            for _ in range(50):
                if lower_bound_val > upper_bound_val:
                    break
                # Sample from range
                # Bias towards smaller values if we want to reduce scope
                if rng.random() < 0.3:
                    new_val = rng.randint(lower_bound_val, min(upper_bound_val, old_val - 1)) if old_val > lower_bound_val else None
                else:
                    new_val = rng.randint(lower_bound_val, upper_bound_val)
                
                if new_val is None or new_val == old_val:
                    continue
                    
                # Check validity
                new_row = row[:]
                new_row[idx] = new_val
                new_diffs = compute_row_diffs(new_row)
                
                # Check if new diffs are in used_without
                if new_diffs & used_without:
                    continue
                
                # Check if new diffs are distinct within the row (implied by set size?)
                # compute_row_diffs returns a set, so internal duplicates are merged.
                # We need to ensure that the *count* of differences matches.
                # A set of differences from a valid row of length k+1 has size k(k+1)/2.
                # If there are internal duplicates, the set size will be smaller.
                if len(new_diffs) != diffs_per_row:
                    continue
                
                # Valid move
                # Calculate new scope
                other_max = 0
                for i, r in enumerate(current):
                    if i != ri:
                        other_max = max(other_max, max(r))
                new_scope = max(other_max, max(new_row))
                
                delta = new_scope - current_scope
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    # Accept
                    current[ri] = new_row
                    used_diffs = used_without | new_diffs
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                    break
            
            T *= cooling
            
        return best, best_scope

    def backtrack_row(used_diffs, M_limit, k, seed_val=0):
        """
        Finds a valid row of length k+1 starting at 0, within [0, M_limit],
        using only differences not in used_diffs.
        Returns (row, new_diffs) or (None, None).
        """
        rng = random.Random(seed_val)
        
        def search(idx, current_row, local_diffs):
            if idx == k + 1:
                return current_row, local_diffs
            
            # Try values from current_row[-1] + 1 to M_limit
            # To improve speed, we can shuffle or use heuristics
            start_val = current_row[-1] + 1
            end_val = M_limit
            
            # Heuristic: try smaller values first to keep scope low?
            # Or random?
            # Let's try a mix: mostly sequential, some random
            candidates = list(range(start_val, end_val + 1))
            # Shuffle a bit to avoid getting stuck in one path
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check if val - prev is a valid diff
                valid = True
                new_local_diffs = set(local_diffs)
                
                for prev in current_row:
                    d = val - prev
                    if d in used_diffs or d in new_local_diffs:
                        valid = False
                        break
                    new_local_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check if we can still complete the row
                # Simple heuristic: not implemented deeply, rely on recursion
                
                res_row, res_diffs = search(idx + 1, current_row + [val], new_local_diffs)
                if res_row is not None:
                    return res_row, res_diffs
            
            return None, None
        
        return search(1, [0], set())

    def solve_backtrack(M_limit, time_limit, seed_val=0):
        """
        Attempts to construct n_rows rows within M_limit using backtracking.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        # Order rows? Maybe try to fill "harder" rows first?
        # For now, fixed order.
        
        for r_idx in range(n_rows):
            if time.time() > time_limit:
                return None
            
            # Try to find a row
            # Use different seeds for each row attempt to diversify
            row, new_diffs = backtrack_row(used_diffs, M_limit, k, seed_val=seed_val + r_idx)
            
            if row is None:
                return None
            
            rows.append(row)
            used_diffs.update(new_diffs)
            
        return rows

    # --- Main Logic ---
    
    # Phase 1: Get a baseline
    # Try greedy with increasing M
    best_rows = None
    best_M = float('inf')
    
    # Start from lower bound
    start_M = lower_bound
    # Cap the initial search to avoid wasting time if greedy is too slow for small M
    end_M = lower_bound + 30 
    
    for M in range(start_M, end_M):
        if time.time() > deadline - 30:
            break
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 25:
                break
            rows = greedy_construct(M, seed)
            if rows is not None:
                if is_valid(rows):
                    current_max = get_scope(rows)
                    if current_max < best_M:
                        best_M = current_max
                        best_rows = [r[:] for r in rows]
                    break
        if best_rows is not None and best_M <= M:
            break
            
    # Fallback if greedy failed
    if best_rows is None:
        M_huge = 300
        rows = greedy_construct(M_huge, 0)
        if rows:
            best_rows = rows
            best_M = get_scope(rows)
        else:
            # Construct a trivial solution with large gaps
            best_rows = []
            base = 0
            for i in range(n_rows):
                row = [0]
                for j in range(1, k + 1):
                    base += 100
                    row.append(base)
                best_rows.append(row)
            best_M = base

    # Phase 2: Improve with SA
    time_left = deadline - time.time()
    if time_left > 5:
        sa_budget = min(time_left - 5, 60)
        if sa_budget > 0:
            improved_rows, improved_M = simulated_annealing(best_rows, sa_budget, seed_val=42)
            if improved_rows is not None and improved_M < best_M:
                best_M = improved_M
                best_rows = improved_rows
            
            # Try another SA run if time permits
            time_left = deadline - time.time()
            if time_left > 5:
                sa_budget2 = min(time_left - 5, 30)
                if sa_budget2 > 0:
                    improved_rows2, improved_M2 = simulated_annealing(best_rows, sa_budget2, seed_val=123)
                    if improved_rows2 is not None and improved_M2 < best_M:
                        best_M = improved_M2
                        best_rows = improved_rows2

    # Phase 3: Backtracking for optimal solution
    # Try to find a solution for M from lower_bound to best_M - 1
    time_left = deadline - time.time()
    if time_left > 2:
        # Allocate time for backtracking
        bt_time_per_M = 2.0  # seconds per M attempt
        max_M_to_try = best_M - 1
        
        for M in range(lower_bound, max_M_to_try + 1):
            if time.time() > deadline - 2:
                break
            
            # Try to solve for M
            # Use a short time limit for each M
            limit = time.time() + min(bt_time_per_M, deadline - time.time() - 2)
            
            # Try a few seeds
            found = False
            for seed in range(5):
                if time.time() > limit:
                    break
                rows = solve_backtrack(M, limit, seed)
                if rows is not None:
                    if is_valid(rows):
                        best_M = M
                        best_rows = rows
                        found = True
                        break
            
            if found:
                # If we found a solution at M, we can stop if M == lower_bound
                if M == lower_bound:
                    break
                # If M > lower_bound, we can continue to see if we can go lower?
                # No, we are iterating upwards. If we found a solution at M, 
                # we should check if we missed a solution at lower M?
                # We iterate from lower_bound upwards. So the first success is the minimum.
                break

    return best_rows

if __name__ == "__main__":
    rows = construct(7, 5)
    if rows:
        print("Solution found:")
        for r in rows:
            print(r)
        print("Scope:", get_scope(rows))
        print("Valid:", is_valid(rows))
    else:
        print("No solution found")
