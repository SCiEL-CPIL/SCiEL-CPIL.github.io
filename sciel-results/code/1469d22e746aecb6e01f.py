import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_extension(row, new_val, used_diffs):
        if new_val <= row[-1]:
            return False
        new_diffs = {new_val - v for v in row}
        if new_diffs & used_diffs:
            return False
        return True

    def get_candidates(row, used_diffs, limit_val):
        candidates = []
        x = row[-1] + 1
        # We don't want to search too far for each step in beam search
        # limit_val can be current_max + some_margin
        while x <= limit_val:
            if is_valid_extension(row, x, used_diffs):
                candidates.append(x)
            x += 1
        return candidates

    # Beam Search Parameters
    beam_width = 100
    max_depth = n_rows * row_len
    # We build rows one point at a time. Total points to add: n_rows * (k)
    # State: list of rows.
    # We use a priority queue based on cost (max entry).
    
    start_time = time.time()
    timeout = 110
    
    # Initial state
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = set()
    initial_cost = 0
    
    # Beam: list of (cost, rows)
    # We need to track used_diffs efficiently. 
    # Since beam width is small, recomputing used_diffs or storing it is okay.
    # Storing used_diffs in the state tuple makes comparison hard.
    # We'll store (cost, rows) and recompute used_diffs when expanding? 
    # That's O(N*K^2) per state. With beam 100, depth 30, it's 100*30*100 = 300k ops. Fine.
    
    # To speed up candidate generation, we can cache used_diffs for each state in the beam.
    # State: (cost, rows, used_diffs)
    
    beam = [(0, initial_rows, initial_used)]
    
    # We need to expand until all rows have length row_len
    # We track the number of points added to each row.
    # Actually, just check len(row) == row_len.
    
    best_solution = None
    best_cost = float('inf')
    
    # We expand level by level? Or just take the best from the beam?
    # Beam search usually expands all states in the beam, generates children, keeps best W.
    # Here, a "move" is adding one point to one row.
    # Total moves: n_rows * k.
    
    # We will perform n_rows * k steps of beam search.
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            # For each row, try to add a point
            for i in range(n_rows):
                if len(rows[i]) >= row_len:
                    continue
                    
                current_max = max(r[-1] for r in rows if r) # Actually max of all entries
                # To limit search space, we look for candidates up to current_global_max + margin
                # current_global_max is the max of all rows in this state
                state_max = max(max(r) for r in rows)
                
                # Heuristic: "Largest remaining differences first" -> 
                # Maybe we should prioritize rows that are "behind"?
                # Let's try: Prioritize rows with the SMALLEST length (most work to do)
                # OR rows with the LARGEST max (bottleneck).
                # Let's stick to trying all rows, but limit candidates.
                
                # Find valid candidates
                # We want small candidates to minimize cost.
                # Limit search to state_max + 10? Or just a fixed window?
                # If we are too restrictive, we might miss valid solutions.
                # Let's search up to state_max + 20.
                limit_val = state_max + 20
                
                candidates = []
                x = rows[i][-1] + 1
                while x <= limit_val:
                    if is_valid_extension(rows[i], x, used_diffs):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # If no candidates in window, expand window?
                    # Or skip this row for this state?
                    # If we skip, we might fail to complete.
                    # Let's expand window a bit more if empty.
                    x = rows[i][-1] + 1
                    while x <= limit_val + 50:
                        if is_valid_extension(rows[i], x, used_diffs):
                            candidates.append(x)
                        x += 1
                        
                if not candidates:
                    continue

                # Select top few candidates to keep beam size manageable
                # Sort by value (smallest first)
                candidates.sort()
                # Take first 5 candidates to diversify
                for new_val in candidates[:5]:
                    # Create new state
                    new_rows = [r[:] for r in rows]
                    new_rows[i].append(new_val)
                    
                    # Update used_diffs
                    new_used_diffs = set(used_diffs)
                    for v in new_rows[i]:
                        if v != new_val:
                            new_used_diffs.add(abs(new_val - v))
                            
                    # Calculate new cost
                    new_cost = max(max(r) for r in new_rows)
                    
                    # Only add if it's not worse than current best found?
                    # No, beam search keeps the best W states regardless of global best,
                    # but we can prune if new_cost >= best_cost found so far?
                    # Yes, if we already have a solution with cost C, we don't need states with cost >= C.
                    if new_cost < best_cost:
                        next_beam.append((new_cost, new_rows, new_used_diffs))
        
        # Keep only the best beam_width states
        if not next_beam:
            # If beam is empty, we might have to relax constraints or break
            break
            
        # Sort by cost
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        # Check if any state in beam is a complete solution
        for cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                
        # Optional: Early termination if we have a very good solution?
        # The task asks for the best possible. We should run until timeout or beam empty.
        
    if best_solution is None:
        # Fallback to a simple greedy if beam search failed
        # This is unlikely for small N, K, but good for safety
        rows = [[0] for _ in range(n_rows)]
        used = set()
        for _ in range(n_rows * k):
            # Find row with smallest length
            i = min(range(n_rows), key=lambda idx: len(rows[idx]))
            if len(rows[i]) >= row_len:
                continue
            x = rows[i][-1] + 1
            while True:
                new_diffs = {x - v for v in rows[i]}
                if not (new_diffs & used):
                    rows[i].append(x)
                    used.update(new_diffs)
                    break
                x += 1
        best_solution = rows
        best_cost = max(max(r) for r in rows)

    return best_solution
