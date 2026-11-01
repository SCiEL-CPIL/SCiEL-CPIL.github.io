import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Hard deadline: evaluator kills at 150s. Leave 5s margin for safety.
    deadline = start_time + 145.0
    
    # Lower bound for M (scope): n_rows * k * (k+1) / 2
    min_M = n_rows * k * (k + 1) // 2
    
    # We will search for the minimal M starting from min_M.
    # For (7,5), min_M = 105. Best known is 112.
    # We'll try up to min_M + 20 to be safe, but prioritize lower M.
    
    best_solution = None
    current_M = min_M
    
    # Max M to try in the optimization loop
    # If we can't find a solution quickly, we increase M.
    # A greedy fallback will handle larger M.
    max_M_to_try = min_M + 20

    while current_M <= max_M_to_try and time.time() < deadline - 10.0:
        
        # Try to find a valid packing of n_rows Sidon sets of size k+1 in [0, current_M]
        # Strategy: Branch and Bound with Bitset Optimization and Heuristics
        
        success = False
        # Try multiple random restarts for this M
        # If M is small, we need more restarts. If M is larger, fewer might suffice.
        num_restarts = 30 if current_M == min_M else 10
        if time.time() > deadline - 15.0:
            num_restarts = 1
            
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            used_diffs = set()
            rows = []
            
            valid = True
            # We build rows one by one.
            # To improve success rate, we can shuffle the order of candidates in the backtracking.
            
            def gen_row(current_row, used_diffs_local, deadline_local):
                """
                Recursively build a row starting with current_row (which starts with [0]).
                used_diffs_local is the set of all differences used so far (global + local).
                Returns the completed row list or None.
                """
                if time.time() > deadline_local:
                    return None
                
                if len(current_row) == k + 1:
                    return list(current_row)
                
                last = current_row[-1]
                # Candidates must be > last and <= current_M
                # And their diffs to all existing elements in current_row must be unused in used_diffs_local
                
                candidates = []
                for x in range(last + 1, current_M + 1):
                    new_diffs = []
                    valid_candidate = True
                    for y in current_row:
                        d = x - y
                        if d in used_diffs_local:
                            valid_candidate = False
                            break
                        # Check if d is already in new_diffs (internal collision)
                        if d in new_diffs:
                            valid_candidate = False
                            break
                        new_diffs.append(d)
                    
                    if valid_candidate:
                        # Heuristic: Score candidates to prefer "tight" or "diverse" choices?
                        # For now, just collect.
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    return None
                
                # Randomize the order of candidates to escape local minima
                random.shuffle(candidates)
                
                for x, new_diffs in candidates:
                    new_used = used_diffs_local.union(new_diffs)
                    res = gen_row(current_row + [x], new_used, deadline_local)
                    if res:
                        return res
                
                return None

            # Call gen_row for each row
            for r in range(n_rows):
                if time.time() > deadline - 5.0:
                    valid = False
                    break
                
                # Start a new row with [0]
                # Note: We pass a copy of used_diffs because gen_row is recursive and modifies it locally via union,
                # but we need to update the global used_diffs only if the row is successful.
                # However, gen_row as written creates new sets. This is correct for backtracking within a row.
                # We need to update the global 'used_diffs' only after a full row is found.
                
                row_res = gen_row([0], set(used_diffs), deadline - 5.0)
                
                if row_res is None:
                    valid = False
                    break
                
                rows.append(row_res)
                # Update global used_diffs with the differences of the new row
                for i in range(len(row_res)):
                    for j in range(i + 1, len(row_res)):
                        used_diffs.add(row_res[j] - row_res[i])
            
            if valid:
                success = True
                # We found a valid solution for this M.
                # Since we are iterating M upwards, the first M that yields a solution is the minimal M found by this method.
                # Any solution for M has max entry <= M.
                # We can return immediately.
                return rows
        
        # If no solution found for this M, increase M
        current_M += 1

    # Fallback: If we didn't find a solution in the loop (timeout or M exceeded),
    # we must return a valid solution.
    # We use a greedy construction with a larger M to ensure validity.
    # The greedy approach picks the smallest available number for each position.
    # This might result in a larger max entry, but it is valid.
    
    if best_solution is not None:
        return best_solution
        
    # Greedy construction
    # We need to find n_rows rows.
    # To ensure we don't run out of space, we pick a large M.
    # A safe M is min_M * 2 or so. Let's just let the greedy algorithm find the max.
    # Actually, to be safe and fast, we can just use a large bound.
    # For (7,5), 105 is LB. 112 is known. Let's use 200 as a safe upper bound for greedy.
    # Greedy is O(n_rows * k^2 * M).
    
    safe_M = min_M + 100
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            valid = True
            new_diffs = []
            for y in row:
                d = x - y
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(x)
                used_diffs.update(new_diffs)
            
            x += 1
            # Safety check: if x exceeds safe_M, we might have a problem, but greedy usually works for reasonable M.
            # If it fails, we'd need a more sophisticated fallback, but for this task, greedy with M=205 should work.
        
        rows.append(row)
        
    return rows
