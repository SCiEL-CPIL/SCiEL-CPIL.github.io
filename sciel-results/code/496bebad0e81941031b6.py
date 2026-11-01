import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit
    
    n = k + 1  # Number of elements per row
    num_diffs_per_row = n * (n - 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # 1. Greedy to establish an upper bound (scope)
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    initial_rows = greedy_construct()
    best_scope = max(max(r) for r in initial_rows)
    
    # We want to find a solution with a smaller scope.
    # Let's try to fit rows within a scope slightly smaller than the greedy bound.
    # The counting bound is 105 for (7,5). Greedy is likely higher.
    # We will search for a valid set of rows where max(entry) <= target_scope.
    
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def check_validity(rows):
        used = set()
        for row in rows:
            for d in get_row_diffs(row):
                if d in used:
                    return False
                used.add(d)
        return True

    # 2. Generate candidate rows
    # A row is a set of n integers {0, a1, ..., a_{n-1}} with 0 < a1 < ... < a_{n-1} <= scope.
    # We need to select n_rows such rows such that their difference sets are disjoint.
    
    # We will use a backtracking search with constraint propagation.
    # To make this efficient, we pre-generate a pool of "good" candidate rows.
    # A good row is one that uses small differences, as small differences are more "expensive" 
    # (they block more other rows).
    
    def generate_candidate_rows(scope, num_candidates):
        candidates = []
        # We generate random rows that fit in [0, scope]
        # We bias towards smaller numbers to increase the chance of packing them tightly.
        
        # Strategy: Generate rows by picking random increasing sequences.
        # To ensure we have enough variety, we use random sampling.
        
        seen_diffs = set()
        # We want unique difference sets to avoid trivial redundancy, but not strictly required.
        # However, if two rows have the same diff set, they are interchangeable.
        
        attempts = 0
        while len(candidates) < num_candidates and attempts < 200000:
            attempts += 1
            # Pick n-1 random distinct integers from 1 to scope
            if scope < n:
                break
            row_vals = [0]
            try:
                # Use random.sample to get distinct values
                # Weight towards smaller values? 
                # Simple approach: just random sample.
                # To bias towards smaller, we can sample from 1..scope//2 + scope//4 etc.
                # Let's try sampling from 1..scope but with a geometric-ish distribution?
                # Or just standard uniform.
                
                # Actually, for difference triangles, rows with small entries are more valuable.
                # Let's generate rows by picking random steps.
                # Start at 0.
                # Next point: random in [1, scope]
                # But must be increasing.
                
                # Better: Pick n-1 distinct values from 1..scope.
                # To bias towards small values, we can pick from 1..scope//2 + scope//4?
                # Let's stick to uniform first, but maybe limit the range for generation.
                # If we only generate rows with max entry <= scope, we are safe.
                
                # Optimization: If we want to find a solution with max entry M,
                # we only need rows with max entry <= M.
                
                vals = random.sample(range(1, scope + 1), n - 1)
                vals.sort()
                row = [0] + vals
                
                diffs = get_row_diffs(row)
                
                # Check if this row is "unique" in terms of diff set?
                # If diff set is in seen_diffs, skip.
                if diffs in seen_diffs:
                    continue
                seen_diffs.add(diffs)
                
                candidates.append((row, diffs))
                
            except ValueError:
                continue
                
        return candidates

    # Backtracking solver
    # State: 
    # - used_diffs: set of all differences used so far
    # - available_rows: list of (row, diffs) that don't conflict with used_diffs
    # - rows_to_place: n_rows
    
    def solve_with_backtracking(target_scope, num_candidates_to_generate):
        # Generate candidates
        candidates = generate_candidate_rows(target_scope, num_candidates_to_generate)
        if not candidates:
            return None
            
        # Sort candidates by "density" or something?
        # Rows with smaller max entry are generally better for packing.
        candidates.sort(key=lambda x: max(x[0]))
        
        used_diffs = set()
        current_rows = []
        
        # Pre-compute conflict matrix? Too big.
        # Instead, at each step, filter candidates.
        
        # Dynamic Variable Ordering: 
        # Which row to place next?
        # All rows are symmetric in terms of constraints (just disjoint diffs).
        # So the order of placement doesn't matter for correctness, only for search space pruning.
        # However, if we place a row with "common" differences first, it prunes more.
        # Differences that appear in many candidate rows are "hot".
        
        # Let's compute frequency of each difference in the candidate pool.
        diff_freq = {}
        for _, diffs in candidates:
            for d in diffs:
                diff_freq[d] = diff_freq.get(d, 0) + 1
        
        # Score a candidate row by the sum of frequencies of its differences.
        # High score = uses common differences = more constraining = prune more.
        # We want to place highly constraining rows first? 
        # Or least constraining?
        # In exact cover, MRV (Minimum Remaining Values) is for variables.
        # Here, all variables are "place a row".
        # If we place a row that uses rare differences, it blocks fewer other rows.
        # If we place a row that uses common differences, it blocks many other rows.
        # Placing common-diff rows first might lead to dead ends faster?
        # Or it might resolve conflicts faster?
        # Let's try placing rows with HIGHEST frequency sum first (most constraining).
        
        def score_candidate(diffs):
            return sum(diff_freq.get(d, 0) for d in diffs)
        
        # We will use a recursive backtracker.
        # To avoid deep recursion limits, we might need to be careful, but n_rows is small (7).
        
        def backtrack(used_diffs, current_rows, remaining):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                return current_rows[:]
                
            # Filter candidates that are compatible with used_diffs
            # We don't want to iterate over ALL candidates every time if we can avoid it.
            # But n_candidates is ~1000-5000. n_rows is 7.
            # 7 * 5000 = 35000 checks. Fast enough.
            
            compatible = []
            for row, diffs in candidates:
                if not (diffs & used_diffs):
                    compatible.append((row, diffs))
            
            if not compatible:
                return None
                
            # Order compatible candidates
            # Heuristic: Sort by score (highest first) to prune early?
            # Or randomize to avoid getting stuck in a specific bad order?
            # Let's sort by score descending.
            compatible.sort(key=lambda x: score_candidate(x[1]), reverse=True)
            
            # To diversify, we can shuffle the top K and pick from there?
            # Or just iterate.
            
            for row, diffs in compatible:
                new_used = used_diffs | diffs
                result = backtrack(new_used, current_rows + [row], remaining - 1)
                if result is not None:
                    return result
                    
            return None
            
        return backtrack(set(), [], n_rows)

    # Strategy:
    # 1. Try to solve for scope = best_scope - 1
    # 2. If fail, try scope = best_scope - 2, etc.
    # 3. If all fail within time, return greedy.
    
    # We need to be careful with the number of candidates generated.
    # If target_scope is small, there are fewer possible rows.
    # If target_scope is large, there are many.
    
    # Let's iterate downwards from best_scope - 1.
    
    best_found = initial_rows
    best_score = best_scope
    
    # We'll try a few scopes.
    # The gap between greedy and optimal might be small.
    # Greedy for (7,5) is likely around 110-120?
    # Optimal is 112.
    # If greedy is 115, we try 114, 113, 112.
    
    # To save time, we'll generate candidates once for the highest scope we try?
    # No, candidates depend on scope (max entry <= scope).
    # But a row valid for scope S is also valid for scope S+1.
    # So we can generate for the largest scope and filter?
    # No, generate_candidate_rows generates rows with entries up to scope.
    # If we generate for scope 115, we have rows with max entry up to 115.
    # If we want to find a solution with max entry <= 112, we can just filter the pool
    # to only include rows with max(entry) <= 112.
    
    # So:
    # 1. Generate a large pool of candidates with max entry <= best_scope.
    # 2. Try to solve for target_scope = best_scope - 1 using only rows with max <= target_scope.
    # 3. If fail, target_scope = best_scope - 2, etc.
    
    # How many candidates?
    # Let's generate 5000 candidates.
    
    pool = generate_candidate_rows(best_scope, 5000)
    
    # Sort pool by max entry
    pool.sort(key=lambda x: max(x[0]))
    
    # Pre-compute diff frequencies for the whole pool?
    # Frequencies change as we filter by scope.
    # But for heuristic ordering, global frequency is a decent proxy.
    
    global_diff_freq = {}
    for _, diffs in pool:
        for d in diffs:
            global_diff_freq[d] = global_diff_freq.get(d, 0) + 1
            
    def score_candidate_global(diffs):
        return sum(global_diff_freq.get(d, 0) for d in diffs)
        
    def solve_with_pool(pool, target_scope, n_rows):
        # Filter pool to rows with max entry <= target_scope
        filtered_pool = [(row, diffs) for row, diffs in pool if max(row) <= target_scope]
        
        if len(filtered_pool) < n_rows:
            return None
            
        # We need to select n_rows disjoint rows from filtered_pool.
        
        used_diffs = set()
        current_rows = []
        
        def backtrack(used_diffs, current_rows, remaining, pool_idx):
            # pool_idx is an optimization to avoid re-checking rows we already considered
            # in this branch? No, because we can pick any row from the remaining pool.
            # Since rows are interchangeable, we can just iterate over the whole filtered_pool
            # but we must avoid picking the same row twice.
            # We can track used_row_indices.
            
            if time.time() > deadline:
                return None
                
            if remaining == 0:
                return current_rows[:]
                
            # Find compatible rows
            # To avoid symmetry (rows are identical in structure, just different values),
            # we don't need to worry about permuting the order of rows.
            # But we must ensure we don't pick the same candidate row twice.
            
            # Iterate through filtered_pool
            for i, (row, diffs) in enumerate(filtered_pool):
                if diffs & used_diffs:
                    continue
                    
                # Optimization: If we've already tried a row with the same diff set, skip?
                # We ensured unique diff sets in generation.
                
                # Heuristic: Order by score?
                # We can't sort inside the loop easily without breaking the "pick any" logic
                # unless we pre-sort.
                
                # Let's just pick the first compatible one? 
                # No, that's greedy and might fail.
                # We need to try different ones.
                
                # To improve search, we can order the candidates by score (descending)
                # so we try the most "constraining" ones first.
                
                # But re-sorting every time is expensive.
                # Let's pre-sort filtered_pool by score.
                
                # Actually, let's just do a simple randomization or fixed order.
                # Fixed order by score is good.
                
                # We'll assume filtered_pool is sorted by score descending.
                
                new_used = used_diffs | diffs
                result = backtrack(new_used, current_rows + [row], remaining - 1, i + 1)
                if result is not None:
                    return result
                    
            return None
            
        # Sort filtered_pool by score descending
        filtered_pool.sort(key=lambda x: score_candidate_global(x[1]), reverse=True)
        
        return backtrack(set(), [], n_rows, 0)

    # Try scopes
    for scope in range(best_scope - 1, max(best_scope - 10, 100), -1):
        if time.time() > deadline - 5:
            break
            
        solution = solve_with_pool(pool, scope, n_rows)
        
        if solution is not None:
            # Validate
            if check_validity(solution):
                new_score = max(max(r) for r in solution)
                if new_score < best_score:
                    best_score = new_score
                    best_found = solution
                    # We found a better solution. Can we go lower?
                    # Yes, continue the loop to try even lower scopes.
                    # But we used the pool generated for best_scope.
                    # Rows in solution have max <= scope.
                    # We can continue trying lower scopes.
            else:
                # Should not happen if backtracking is correct
                pass
        else:
            # No solution found for this scope.
            # Try next lower scope.
            continue
            
    # If we didn't find anything better, return greedy
    return best_found
