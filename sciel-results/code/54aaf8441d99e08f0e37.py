import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    total_diffs_per_row = k * (k + 1) // 2
    
    # Lower bound on scope
    # We need n_rows * total_diffs_per_row distinct differences in [1, M]
    # So M >= n_rows * total_diffs_per_row
    lb = n_rows * total_diffs_per_row
    
    # Upper bound: Greedy construction
    def greedy_construct(n, k_val, rng):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            for pos in range(k_val):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        for a in row:
                            used.add(x - a)
                        row.append(x)
                        break
                    x += 1
            rows.append(row)
        return rows

    rng = random.Random(42)
    current_best_rows = greedy_construct(n_rows, k, rng)
    current_best_scope = max(max(r) for r in current_best_rows)
    
    # Helper to get differences of a row
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        diffs = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for d in get_diffs(row):
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Strategy:
    # 1. Determine a starting upper bound (current_best_scope).
    # 2. Iterate M downwards from current_best_scope - 1 to lb.
    # 3. For each M, generate a pool of candidate rows that fit in [0, M].
    # 4. Use randomized greedy set-packing to find n_rows disjoint rows from the pool.
    # 5. If found, update best and continue. If not found after some tries, break (assuming M is too small).
    
    best_scope = current_best_scope
    best_rows = current_best_rows
    
    # We will try to improve.
    # Since the gap is small (134 vs 112), we can afford to try each M.
    
    for M in range(current_best_scope - 1, lb - 1, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Number of restarts for this M
        # If M is close to lb, it's harder. If M is close to best, it's easier.
        # Let's use a fixed number of restarts, say 50, with a time limit per M.
        max_time_for_M = 3.0
        start_time = time.time()
        
        for restart in range(100):
            if time.time() - start_time > max_time_for_M:
                break
            if time.time() > deadline - 5:
                break
                
            # Generate candidate rows for this M
            # We need rows of length k+1, starting at 0, strictly increasing, max <= M.
            # And their internal differences must be unique within the row.
            
            candidates = []
            # How many candidates? Maybe 50-100.
            # We can generate them by random sampling or greedy with randomization.
            
            # Method: Randomized greedy to generate valid rows within scope M
            # We need to ensure the row itself is valid (internal diffs unique).
            # Note: The problem statement says "all differences between two entries of the same row are distinct over all rows".
            # This implies internal diffs must be unique within the row AND across rows.
            # So a candidate row must have unique internal diffs.
            
            for _ in range(50):
                row = [0]
                valid_row = True
                for pos in range(k):
                    # Try to pick a value that keeps internal diffs unique and <= M
                    # We can do a mini-backtracking here or just try random values
                    # Since M is small, we can try values from row[-1]+1 to M
                    # But we need to ensure internal uniqueness.
                    
                    # Let's try a few random permutations of the remaining "slots" or just greedy with random jumps
                    # Simpler: Generate a random increasing sequence of length k from 1..M, check validity.
                    # If valid, add to candidates.
                    
                    # Efficient generation:
                    # Pick k distinct values from 1..M, sort them, prepend 0.
                    # Check if internal diffs are unique.
                    # This is fast.
                    
                    # But we need to ensure the max is <= M.
                    # If we pick from 1..M, max is <= M.
                    
                    # Number of ways to pick k from M is C(M, k).
                    # For M=134, k=7, C(134, 7) is huge.
                    # We can just sample.
                    
                    vals = rng.sample(range(1, M + 1), k)
                    vals.sort()
                    row_candidate = [0] + vals
                    
                    # Check internal uniqueness
                    diffs = set()
                    ok = True
                    for i in range(len(row_candidate)):
                        for j in range(i + 1, len(row_candidate)):
                            d = row_candidate[j] - row_candidate[i]
                            if d in diffs:
                                ok = False
                                break
                            diffs.add(d)
                    if ok:
                        candidates.append((row_candidate, diffs))
                        
                # If we generated enough candidates, break
                if len(candidates) >= 20:
                    break
            
            if len(candidates) < n_rows:
                continue
            
            # Now, select n_rows disjoint rows from candidates
            # Greedy with randomization:
            # Sort candidates by some heuristic? 
            # Or just pick randomly, check disjointness.
            # Backtracking on the set of candidates is better.
            
            # Let's do a randomized backtracking on the candidate list.
            # State: selected_rows, used_diffs
            # Pick a candidate, if disjoint, recurse.
            
            # To make it efficient, we can shuffle candidates at each level.
            
            def backtrack_set_packing(idx, selected, used_diffs):
                if time.time() > deadline - 5:
                    return None
                if len(selected) == n_rows:
                    return selected
                if idx == len(candidates):
                    return None
                
                # Try candidates from idx to end
                # Randomize order
                # Instead of idx, we can just pick a random unused candidate each time?
                # No, that might miss combinations.
                # Let's use a simple recursive search with pruning.
                
                # Optimization: If the number of remaining candidates is less than needed, fail.
                remaining_needed = n_rows - len(selected)
                remaining_candidates = len(candidates) - idx
                if remaining_candidates < remaining_needed:
                    return None
                
                # Heuristic: Try candidates that use "small" differences first?
                # Or just random.
                
                # To speed up, we can pre-sort or just iterate.
                # Let's iterate and pick randomly among valid ones?
                # Actually, simple DFS with random restarts on the candidate order is robust.
                
                # Let's just iterate through candidates starting from idx.
                # We can shuffle the list of candidates once per restart.
                
                for i in range(idx, len(candidates)):
                    row, diffs = candidates[i]
                    if not diffs.isdisjoint(used_diffs):
                        continue
                    
                    new_selected = selected + [row]
                    new_used = used_diffs | diffs
                    res = backtrack_set_packing(i + 1, new_selected, new_used)
                    if res is not None:
                        return res
                return None
            
            # Shuffle candidates to explore different branches
            rng.shuffle(candidates)
            res = backtrack_set_packing(0, [], set())
            if res is not None:
                # Verify
                if is_valid(res):
                    best_rows = res
                    best_scope = M
                    found = True
                    break
            
        if not found:
            # If we couldn't find a solution for M, it's likely that M is too small
            # (or we just got unlucky). 
            # Given the small gap, if we fail for M, M-1 is even harder.
            # But maybe we just didn't sample enough candidates.
            # Let's allow a few "failed" M's before breaking?
            # No, the set of valid solutions for M-1 is a subset of M.
            # If we exhaust the search space for M (which we don't, we sample), we can't be sure.
            # However, if M is the counting bound, it's very tight.
            # Let's just break to be safe, or continue if we have time?
            # If we break, we might miss a solution at M-1 if our candidate pool was bad.
            # But generating candidates for M-1 is similar.
            # Let's break to save time for lower M's if we think M is infeasible.
            # Actually, if we fail at M, M-1 is harder. So breaking is reasonable.
            break
            
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
