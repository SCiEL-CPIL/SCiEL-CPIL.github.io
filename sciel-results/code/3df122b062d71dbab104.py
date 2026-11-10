import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def now():
        return time.time()
        
    def time_left():
        return deadline - now()

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    
    # Precompute all possible rows up to a reasonable max scope
    # We don't know the optimal scope, but we can estimate a bound.
    # Lower bound: n_rows * diffs_per_row
    # Upper bound: Let's start with something generous, e.g., 2 * lb + 100
    lb = n_rows * diffs_per_row
    max_scope_guess = int(lb * 1.5) + 200
    
    # Generate candidate rows
    # A row is valid if it starts with 0, is strictly increasing, and has distinct internal diffs.
    # We will store rows as tuples.
    # We also store the set of differences for each row for quick lookup.
    
    # To make this efficient, we only generate rows that are "compact" enough.
    # However, generating ALL rows up to max_scope_guess might be too many.
    # Let's estimate: For k=5, row_len=6. 
    # Number of subsets of size 6 from {0..M} is C(M+1, 6).
    # If M=100, C(101, 6) ~ 100^6/720 ~ 10^12/720 ~ 10^9. Too many.
    
    # Strategy: Instead of precomputing ALL rows, we will use a heuristic search
    # similar to the original but framed as Local Search on the set of rows.
    
    # Actually, the prompt asks for a DIFFERENT strategy.
    # The current one is "Construct rows sequentially".
    # A different family is "Global Optimization / Local Search".
    
    # Let's try a different angle: 
    # 1. Randomly initialize a set of rows (valid individually).
    # 2. Check global validity (no diff collisions).
    # 3. If collision, pick a conflicting pair and try to "fix" one by replacing it 
    #    with a random valid row that doesn't collide with others.
    # 4. Simulated Annealing on the "max entry".
    
    # To make "random valid row" fast, we can precompute a pool of valid rows 
    # for small scopes, or generate them on the fly.
    
    # Let's try generating a pool of valid rows up to a certain scope.
    # If k is small (like 5), we can afford to generate many.
    
    def generate_valid_rows(max_val):
        """Generate all valid rows of length row_len starting at 0 with max <= max_val."""
        valid_rows = []
        # Use recursion or itertools
        import itertools
        
        # For efficiency, we might limit this if max_val is large.
        # If max_val is too large, we switch to on-the-fly generation.
        if max_val > 200: 
            return None # Signal to use on-the-fly
        
        for combo in itertools.combinations(range(1, max_val + 1), row_len - 1):
            row = [0] + list(combo)
            # Check strictly increasing (guaranteed by combinations)
            # Check distinct diffs
            diffs = set()
            ok = True
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
            if ok:
                valid_rows.append((tuple(row), diffs))
        return valid_rows

    # Try to build a pool
    # Estimate scope needed
    # If n_rows * diffs_per_row is small, pool is feasible.
    # Let's try a pool up to scope = lb + 50
    pool_scope = lb + 50
    if pool_scope < 50:
        pool_scope = 50
        
    pool = generate_valid_rows(pool_scope)
    
    if pool is None:
        # Fallback to a simpler heuristic if pool is too big
        # Just use the original greedy but with random restarts and local search?
        # Let's stick to a simplified local search that generates rows on the fly.
        
        def gen_random_row(used_bits, scope):
            """Generate a random valid row not colliding with used_bits."""
            # Try several times
            for _ in range(100):
                # Randomly pick row_len - 1 distinct values from 1..scope
                # Ensure they are increasing
                vals = random.sample(range(1, scope + 1), row_len - 1)
                vals.sort()
                row = [0] + vals
                
                # Check internal diffs
                diffs = set()
                ok = True
                for i in range(row_len):
                    for j in range(i + 1, row_len):
                        d = row[j] - row[i]
                        if d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                if not ok:
                    continue
                
                # Check global collision
                conflict = False
                for d in diffs:
                    if used_bits & (1 << d):
                        conflict = True
                        break
                if conflict:
                    continue
                
                return tuple(row), diffs
            return None, None

    else:
        # We have a pool.
        # Pool is a list of (row_tuple, diff_set)
        pass

    # Initial Solution: Random Selection from Pool (or generated)
    
    def get_initial_solution():
        if pool is not None:
            # Shuffle pool
            shuffled_pool = pool[:]
            random.shuffle(shuffled_pool)
            
            rows = []
            used_bits = 0
            for _ in range(n_rows):
                # Find a row in pool that doesn't collide
                found = None
                for row_tuple, diffs in shuffled_pool:
                    # Check collision
                    conflict = False
                    for d in diffs:
                        if used_bits & (1 << d):
                            conflict = True
                            break
                    if not conflict:
                        found = (row_tuple, diffs)
                        break
                
                if found is None:
                    return None # Failed to find initial solution
                
                rows.append(found)
                for d in found[1]:
                    used_bits |= (1 << d)
            return rows
        else:
            # Generate on the fly
            rows = []
            used_bits = 0
            for _ in range(n_rows):
                res, diffs = gen_random_row(used_bits, pool_scope)
                if res is None:
                    return None
                rows.append((res, diffs))
                for d in diffs:
                    used_bits |= (1 << d)
            return rows

    best_rows = get_initial_solution()
    if best_rows is None:
        # Fallback to deterministic greedy if random fails
        # (This is a safety net, but we want to be different)
        # Let's try a greedy fill
        rows = []
        used_bits = 0
        for _ in range(n_rows):
            # Greedy smallest
            row = [0]
            for i in range(1, row_len):
                x = row[-1] + 1
                while True:
                    # Check if x works
                    new_diffs = [x - a for a in row]
                    ok = True
                    seen = set()
                    for d in new_diffs:
                        if d in used_bits or d in seen:
                            ok = False
                            break
                        seen.add(d)
                    if ok:
                        row.append(x)
                        for d in new_diffs:
                            used_bits |= (1 << d)
                        break
                    x += 1
            rows.append((tuple(row), set(d for a in row for b in row if b > a)))
        best_rows = rows

    best_max = max(r[0][-1] for r in best_rows)
    
    # Local Search / Simulated Annealing
    # State: list of rows (tuples)
    # Move: Replace row i with a new valid row j
    # Cost: max entry of the set of rows
    
    # To make moves fast, we need to quickly find a replacement row.
    # If we have a pool, we can filter it.
    # If not, we generate.
    
    def calculate_max(rows):
        return max(r[0][-1] for r in rows)

    def get_used_bits_except(rows, idx):
        ub = 0
        for i, r in enumerate(rows):
            if i == idx:
                continue
            for d in r[1]:
                ub |= (1 << d)
        return ub

    # Precompute bit masks for pool if available
    if pool is not None:
        # Create a list of indices for the pool
        pool_indices = list(range(len(pool)))
        
        # Optimization: For each row in pool, precompute its bitmask?
        # No, we check collision on the fly with used_bits.
        
        # To speed up replacement:
        # When replacing row i, we have fixed_used = used_bits_except(i).
        # We want a row from pool such that (row_diffs & fixed_used) == 0.
        # And we want to minimize the max entry of the new set.
        # This is a search problem.
        
        # Heuristic: 
        # 1. Identify the row with the largest max entry.
        # 2. Try to replace it with a row from the pool that has a smaller max entry
        #    and doesn't collide with others.
        
        current_rows = [list(r) for r in best_rows] # List of [row_tuple, diff_set]
        
        T = 10.0
        cooling = 0.995
        
        while time_left() > 10:
            # Pick a random row to replace, biased towards larger ones
            # Or just pick the one with max entry
            max_idx = max(range(n_rows), key=lambda i: current_rows[i][0][-1])
            
            # Also try random swaps
            if random.random() < 0.5:
                idx = max_idx
            else:
                idx = random.randint(0, n_rows - 1)
            
            fixed_used = get_used_bits_except(current_rows, idx)
            
            # Find a replacement
            new_row = None
            new_diffs = None
            
            # Search in pool
            # We want a row that is disjoint from fixed_used.
            # To improve score, we prefer rows with smaller max entry.
            
            # Sort pool indices by max entry? 
            # Pre-sorting pool by max entry is good.
            # Let's do it once.
            if not hasattr(get_initial_solution, 'sorted_pool'):
                get_initial_solution.sorted_pool = sorted(range(len(pool)), key=lambda i: pool[i][0][-1])
            
            candidates = get_initial_solution.sorted_pool
            
            # Try to find a valid replacement
            # We can add randomness: shuffle a subset or pick random from top K
            # For simplicity, iterate through sorted pool until we find one that fits.
            # To allow "uphill" moves (worse max), we might accept a slightly larger row 
            # if it allows future improvements? 
            # But replacing a row with a LARGER max entry doesn't help the immediate cost.
            # It might help if it frees up "small" diffs for other rows?
            # Actually, if we replace row i with row j, and max(j) > max(i), 
            # the global max might increase. We only accept if global max decreases 
            # or with probability exp(-delta/T).
            
            # Let's find the BEST replacement (smallest max entry) that is valid.
            # If best_max_new < best_max, accept.
            # Else, maybe accept with SA probability.
            
            best_candidate = None
            best_candidate_max = float('inf')
            
            # Limit search to top N candidates to keep it fast
            search_limit = 100
            for idx_p in candidates[:search_limit]:
                row_tuple, diffs = pool[idx_p]
                # Check collision
                conflict = False
                for d in diffs:
                    if fixed_used & (1 << d):
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Valid candidate
                row_max = row_tuple[-1]
                if row_max < best_candidate_max:
                    best_candidate = (row_tuple, diffs)
                    best_candidate_max = row_max
                    
                # We can stop if we find one with max 0? No, min is k.
                # We can stop if we find one that reduces the global max?
                # We don't know global max change yet.
                # Let's just find the absolute best (smallest max) valid row.
                
            if best_candidate is None:
                # No replacement found in top 100. Try random sampling from pool.
                for _ in range(20):
                    idx_p = random.choice(candidates)
                    row_tuple, diffs = pool[idx_p]
                    conflict = False
                    for d in diffs:
                        if fixed_used & (1 << d):
                            conflict = True
                            break
                    if not conflict:
                        best_candidate = (row_tuple, diffs)
                        best_candidate_max = row_tuple[-1]
                        break

            if best_candidate is None:
                T *= cooling
                continue
            
            # Calculate new global max
            old_row_max = current_rows[idx][0][-1]
            new_row_max = best_candidate[0][-1]
            
            # The new global max is max( max(other_rows), new_row_max )
            # We need max of other rows.
            max_others = 0
            for i, r in enumerate(current_rows):
                if i != idx:
                    if r[0][-1] > max_others:
                        max_others = r[0][-1]
            
            new_global_max = max(max_others, new_row_max)
            old_global_max = max(max_others, old_row_max) # Which is current best_max
            
            delta = new_global_max - old_global_max
            
            if delta < 0 or (T > 0.1 and random.random() < pow(2.71828, -delta / T)):
                # Accept
                current_rows[idx] = best_candidate
                if new_global_max < best_max:
                    best_max = new_global_max
                    best_rows = [list(r) for r in current_rows]
            
            T *= cooling

    else:
        # If pool was too big, we used the on-the-fly generator.
        # We can do a similar local search but generating rows on the fly is slower.
        # Given the constraints, if pool is None, we likely have large n_rows or k.
        # For (7,5), pool should be fine.
        pass

    # Convert back to list of lists
    result = [list(r[0]) for r in best_rows]
    
    # Final validation
    used = set()
    valid = True
    for r in result:
        if not r or r[0] != 0:
            valid = False
            break
        for a, b in zip(r, r[1:]):
            if b <= a:
                valid = False
                break
        diffs = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs or d in used:
                    valid = False
                    break
                diffs.add(d)
        if not valid:
            break
        used.update(diffs)
        
    if not valid:
        # Fallback to a simple valid construction if something went wrong
        # This should rarely happen if the logic is sound
        rows = []
        used_bits = 0
        for _ in range(n_rows):
            row = [0]
            for i in range(1, row_len):
                x = row[-1] + 1
                while True:
                    new_diffs = [x - a for a in row]
                    ok = True
                    seen = set()
                    for d in new_diffs:
                        if d in used_bits or d in seen:
                            ok = False
                            break
                        seen.add(d)
                    if ok:
                        row.append(x)
                        for d in new_diffs:
                            used_bits |= (1 << d)
                        break
                    x += 1
            rows.append(row)
        return rows

    return result
