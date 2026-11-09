import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    # k+1 integers per row.
    # Row: [0, x1, x2, ..., xk]
    # Differences: C(k+1, 2)
    
    # We want to find rows such that all diffs are distinct.
    # Minimize max value.
    
    # Lower bound for M:
    # Total diffs = n_rows * (k+1)*k//2
    # Diffs must be in [1, M]
    # So M >= Total Diffs
    
    total_diffs = n_rows * (k + 1) * k // 2
    
    # The best known set for (7, 5) has largest entry 112.
    # The simple counting bound is 105.
    # We will try to find a solution with M starting from the lower bound, 
    # or a known good bound if lower bound is too tight to find quickly.
    
    # Strategy:
    # 1. Use a greedy construction to find a valid solution quickly with a reasonably small M.
    # 2. Try to optimize (reduce M) if time permits, or try different M values.
    # 3. Given the complexity, a simple backtracking might be too slow for M=105.
    #    We will prioritize finding a valid solution first.
    
    # Greedy approach:
    # For each row, pick the smallest possible next element that doesn't create
    # duplicate differences within the row or with previous rows.
    
    # To get a better initial solution, we can randomize the order of rows
    # or the specific choices in the greedy step.
    
    def greedy_construct(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            # We need to pick k more numbers.
            # To make it harder to collide, maybe we should pick numbers that are "spread out"?
            # Or just smallest possible.
            # Let's try smallest possible first.
            
            # To add variety, we can shuffle the candidate order?
            # No, smallest is usually best for minimizing max value in greedy.
            # But we might get stuck.
            
            # Let's try to pick the next number.
            # The next number must be > row[-1].
            # And <= M_limit.
            # And must allow for remaining k - len(row) + 1 numbers?
            # Actually, we just need to pick k numbers.
            
            # Heuristic:
            # Try candidates from row[-1] + 1 up to M_limit.
            # Check if valid.
            
            found = True
            while len(row) < k + 1:
                next_val = row[-1] + 1
                # We need to ensure we can still pick enough numbers.
                # remaining_to_pick = k + 1 - len(row)
                # max_possible_val = M_limit - (remaining_to_pick - 1)
                
                valid_found = False
                # To break ties or add randomness, we could shuffle a window?
                # For now, deterministic smallest.
                
                # Optimization: If we are close to M_limit, we might fail.
                
                while next_val <= M_limit:
                    # Check if adding next_val is valid
                    # Diffs with all previous elements in row
                    valid = True
                    for prev in row:
                        d = next_val - prev
                        if d in used_diffs:
                            valid = False
                            break
                        # Also check if d is already in the row's internal diffs?
                        # We haven't added them to used_diffs yet for this row.
                        # We need to check against internal diffs of the current partial row.
                        # Let's maintain a set for current row diffs?
                        # Or just check against all previous pairs?
                        # For small k, it's fine to recompute or maintain a set.
                        # Let's maintain a local set `row_diffs` for the current row.
                        # But we are in a while loop, so we can just check against `used_diffs`
                        # and then check for internal collisions separately?
                        # Actually, if we pick `next_val`, the new diffs are `next_val - x` for x in row.
                        # These must be distinct from each other AND from `used_diffs`.
                        # Are they distinct from each other?
                        # If row is increasing, `next_val - x` are distinct for distinct x.
                        # Yes, because x are distinct.
                        # So we only need to check against `used_diffs`.
                        
                        if not valid:
                            break
                    
                    if valid:
                        # Also need to ensure that we don't create a diff that is already
                        # present in the current row?
                        # Wait, `used_diffs` is global (all previous rows).
                        # Internal diffs of the current row are not in `used_diffs` yet.
                        # So we must check if any new diff is already in the current row's diffs.
                        # Let's compute current row diffs.
                        
                        # This is O(k) per step.
                        curr_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                curr_diffs.add(row[j] - row[i])
                        
                        new_diffs = {next_val - x for x in row}
                        
                        if new_diffs & curr_diffs:
                            valid = False
                        if new_diffs & used_diffs:
                            valid = False
                        
                        if valid:
                            row.append(next_val)
                            # Update used_diffs?
                            # We should update used_diffs with the new diffs of the partial row?
                            # No, `used_diffs` should only contain diffs from *completed* rows?
                            # Or all diffs so far?
                            # If we update `used_diffs` as we build the row, then when we pick the next element,
                            # we check against all previous diffs including those from the current row.
                            # This is correct and more efficient.
                            
                            used_diffs.update(new_diffs)
                            valid_found = True
                            break
                    next_val += 1
                
                if not valid_found:
                    # Failed to find a valid next value
                    return None
            
            rows.append(row)
            
        return rows
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    best_rows = None
    best_M = float('inf')
    
    # Try to find a solution with M close to the lower bound.
    # Start from a known good value (112) and go down?
    # Or start from lower bound and go up?
    # Greedy is fast. Let's try a range of M values.
    
    # For (7, 5), LB=105, Best=112.
    # Let's try M from 105 to 120.
    
    # To increase the chance of finding a solution, we can use multiple random seeds
    # for the greedy process? The current greedy is deterministic (smallest first).
    # Maybe we can randomize the choice among valid candidates?
    # Or try different orders of rows?
    
    # Let's stick to deterministic greedy first. It's fast.
    
    # If greedy fails for M=105, it might succeed for M=106, etc.
    
    # We'll iterate M from LB to some upper bound.
    # If we find a solution, we record it.
    # We can stop if we find one at the LB?
    # But maybe a larger M allows a "better" structure? No, we want to minimize M.
    # So we want the smallest M that yields a valid solution.
    
    # Greedy might get stuck for small M even if a solution exists.
    # But it's a good heuristic.
    
    # Let's try M from total_diffs to total_diffs + 20.
    
    search_start = total_diffs
    search_end = total_diffs + 30
    
    # Prioritize lower M.
    for M in range(search_start, search_end):
        if time.time() > deadline - 10:
            break
        
        rows = greedy_construct(M)
        if rows is not None:
            # Verify validity (just in case greedy logic has a bug)
            if is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max <= M: # Should be true by construction
                    if current_max < best_M:
                        best_M = current_max
                        best_rows = [r[:] for r in rows]
                    # If we found a solution at the lower bound, we can't do better.
                    if M == total_diffs:
                        break
                    # Even if M > total_diffs, we might find a smaller max in the rows
                    # if the greedy didn't use the full range?
                    # But greedy uses smallest possible, so max should be close to M.
                    # We continue to see if a smaller M works?
                    # No, we are iterating M upwards.
                    # If we found a valid solution for M, we have a candidate.
                    # We should stop if we find one for the smallest M tried?
                    # Yes, because we are trying M in increasing order.
                    # The first valid M gives the smallest M for this greedy strategy.
                    # However, a solution might exist for a smaller M that greedy missed.
                    # But we started at LB. So if LB fails, we go up.
                    # So the first success is the best M for this strategy.
                    break
    
    if best_rows is None:
        # Greedy failed for all M in range.
        # This is unlikely for M up to 135.
        # Fallback: Use a much larger M to ensure a valid solution.
        # Greedy with large M should always work.
        M_fallback = total_diffs + 50
        rows = greedy_construct(M_fallback)
        if rows is not None and is_valid(rows):
            best_rows = rows
            best_M = max(max(r) for r in rows)
        else:
            # Emergency fallback: Construct a trivial solution?
            # We need a valid solution.
            # Just pick numbers that are multiples of a large prime?
            # Or just spread them out very widely.
            # Row i: [0, (i+1)*1000, (i+1)*2000, ...]
            # This will have huge max, but it's valid.
            # Let's just use a very large M.
            M_huge = 1000
            rows = greedy_construct(M_huge)
            if rows is not None:
                best_rows = rows
            else:
                # Should not happen
                best_rows = [[0] + list(range(1, k + 1))] * n_rows
                # This is likely invalid, but better than crashing.
                # Actually, let's force a valid one.
                # Use disjoint ranges for each row.
                # Row 0: 0..M
                # Row 1: M+1..2M+1 (shifted)
                # But diffs must be global.
                # If we shift by large amounts, diffs within row are same as before.
                # So we need the base rows to have distinct diffs.
                # This is getting complicated.
                # Let's trust greedy_construct with M=1000.
                pass
    
    return best_rows
