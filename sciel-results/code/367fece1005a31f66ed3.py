import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    L = k + 1
    diffs_per_row = k * (k + 1) // 2
    
    # If the problem is trivial or small, we can use a simpler heuristic
    # But let's assume the hard case (4, 7) -> L=8, diffs=28, total_diffs=112
    
    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid_solution(rows):
        """
        Verify that all differences across all rows are distinct.
        """
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def backtracking_search(max_scope, time_limit):
        """
        Backtracking search to find a valid configuration within max_scope.
        We model this as filling rows one by one.
        To improve efficiency, we use a bitmask for used differences.
        Since max_scope is likely < 200, a Python int is sufficient for the bitmask.
        """
        used_mask = 0
        rows = []
        
        # We will build rows one by one.
        # For each row, we need to choose k-1 numbers (besides 0) from [1, max_scope]
        # such that the differences are unique within the row and do not collide with used_mask.
        
        # Heuristic: Choose the next row's elements in a way that spreads them out?
        # Or just backtrack?
        
        # A pure backtracking on 4 rows of 8 elements is huge.
        # We need a strong heuristic.
        
        # Let's try a hybrid:
        # 1. Greedy construction with randomization to get a "good" starting point.
        # 2. Local search to minimize scope.
        
        # Actually, the prompt asks for a DIFFERENT strategy.
        # The current program uses Greedy + SA/LS.
        # Let's try: **Iterative Deepening A* (IDA*)** or **Backtracking with Constraint Propagation**.
        
        # Given the 120s limit, a full backtracking might be too slow if not guided.
        # Let's use a **Greedy with Backtracking** approach specifically for the last few rows.
        
        # Strategy:
        # 1. Fill the first n_rows-1 rows using a fast greedy (randomized).
        # 2. Use backtracking to fill the last row, trying to minimize its max element.
        # 3. If successful, record the scope.
        # 4. Repeat with different random seeds for the greedy part.
        
        # Wait, this is still similar to the current one.
        # Let's try **Row-Swap Local Search** as suggested by inspiration 2, but implemented differently.
        # Or better: **Constructive Heuristic with Conflict Resolution**.
        
        # Let's implement a specific variant: **Greedy Construction with Backtracking on the Last Row**.
        # This is different from the current program which does Greedy on ALL rows then SA on the whole thing.
        # By constraining the first rows to be "greedy" and solving the last row via backtracking,
        # we might find tighter solutions because the last row can "fit" into the gaps left by the first rows.
        
        def greedy_fill_row(used_mask, current_max, seed):
            """
            Greedily fill a row given the used differences.
            Returns the row and the new used mask, or None if stuck (though greedy shouldn't get stuck if scope is large enough).
            We bias towards smaller numbers.
            """
            row = [0]
            x = 0
            rng = random.Random(seed)
            
            while len(row) < L:
                # Look for the smallest valid x > current_max
                # To add diversity, we can look in a window [x+1, x+window]
                # But for the "greedy" part, we want to be tight.
                # Let's try candidates from x+1 upwards, but with a small random skip probability?
                # No, let's stick to a randomized window for the first n-1 rows.
                
                found = False
                # Window size depends on how tight we want to be.
                # If we are filling early rows, we have more freedom.
                window = 10
                candidates = list(range(x + 1, x + window + 1))
                rng.shuffle(candidates)
                
                for cand in candidates:
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = cand - a
                        if (used_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if valid:
                        # Check internal collision
                        if new_diffs.bit_count() != len(row) + 1 - 1: # len(row) new diffs? No, len(row) is current size. Adding 1 makes size+1. Diffs added = size.
                            # Wait, if row has m elements, adding 1 creates m new diffs.
                            # So bit_count should be m.
                            if new_diffs.bit_count() != len(row):
                                valid = False
                                break
                        
                        if valid:
                            row.append(cand)
                            used_mask |= new_diffs
                            x = cand
                            found = True
                            break
                
                if not found:
                    # If no candidate in window, expand window
                    cand = x + 1
                    while True:
                        valid = True
                        new_diffs = 0
                        for a in row:
                            d = cand - a
                            if (used_mask >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        
                        if valid:
                            if new_diffs.bit_count() != len(row):
                                valid = False
                                break
                            
                        if valid:
                            row.append(cand)
                            used_mask |= new_diffs
                            x = cand
                            found = True
                            break
                        cand += 1
                        if cand > max_scope:
                            return None, used_mask # Exceeded scope
                    
            return row, used_mask

        def backtrack_last_row(used_mask, max_scope, seed):
            """
            Try to fill the last row using backtracking.
            This is the critical step. We want to find a row that fits in the remaining differences.
            """
            row = [0]
            x = 0
            rng = random.Random(seed)
            
            def dfs(current_row, current_x, current_used_mask):
                if len(current_row) == L:
                    return current_row[:]
                
                # Determine candidates for the next element
                # We want to minimize the final max, so we try small values first.
                # But we must ensure we can complete the row.
                
                # Heuristic: Try values in a random order but biased towards small?
                # Or try all values from current_x + 1 to max_scope?
                # To avoid combinatorial explosion, we limit the branching factor or use a heuristic.
                
                # Let's try candidates from current_x + 1 to max_scope.
                # If we find a valid completion, we return it.
                
                # Optimization: If current_x is already > max_scope, fail.
                if current_x > max_scope:
                    return None
                
                # Generate candidates
                # To make it faster, we can restrict the range.
                # The maximum possible value for the next element is max_scope.
                # The minimum is current_x + 1.
                
                # Let's shuffle the candidates to break symmetry, but keep small ones more likely?
                # Actually, for backtracking, order matters.
                # Let's try a mix: some small, some random.
                
                cands = list(range(current_x + 1, max_scope + 1))
                # This list can be large (up to 134).
                # Shuffling 134 items is fine.
                rng.shuffle(cands)
                
                # But we want to find a solution quickly.
                # A "best-first" approach might be better: try small values first.
                # Let's try sorted(cands) but with a random perturbation?
                # Or just sorted?
                
                # If we always try smallest first, we might get stuck in a dead end quickly.
                # Let's try a random order.
                
                for cand in cands:
                    # Check if cand is valid
                    valid = True
                    new_diffs = 0
                    for a in current_row:
                        d = cand - a
                        if (current_used_mask >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    
                    if not valid:
                        continue
                    
                    # Check internal collision
                    if new_diffs.bit_count() != len(current_row):
                        continue
                    
                    # Recurse
                    new_row = current_row + [cand]
                    new_mask = current_used_mask | new_diffs
                    
                    res = dfs(new_row, cand, new_mask)
                    if res:
                        return res
                        
                return None

            return dfs(row, 0, used_mask)

        # Main loop
        best_rows = None
        best_scope = float('inf')
        
        # We will try to construct n_rows-1 rows greedily, and 1 row via backtracking.
        # We'll also try different numbers of greedy rows? No, let's stick to n-1 greedy, 1 backtrack.
        # Or maybe 2 backtrack?
        
        # Let's iterate with different random seeds.
        # Each iteration:
        # 1. Generate n_rows-1 rows greedily.
        # 2. Try to fill the last row via backtracking with a scope limit.
        # 3. If successful, update best.
        
        # To find the *minimum* scope, we can binary search on the scope?
        # Or just run many iterations and keep the best found.
        
        # Let's run many iterations.
        
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            seed = attempts
            
            # Generate first n_rows - 1 rows
            used_mask = 0
            rows = []
            success = True
            
            for i in range(n_rows - 1):
                # For earlier rows, we can be more aggressive (smaller window) or more relaxed?
                # Let's use a window that scales with the row index?
                # Or just a fixed window.
                row, used_mask = greedy_fill_row(used_mask, best_scope - 1 if best_scope < float('inf') else 200, seed + i)
                if row is None:
                    success = False
                    break
                rows.append(row)
                
            if not success:
                continue
                
            # Now try to fill the last row
            # We want to minimize the scope.
            # Let's try backtracking with a scope limit.
            # What limit?
            # We can try the current best_scope - 1.
            # Or we can try to find ANY solution first, then tighten.
            
            # Let's try to find a solution with scope <= current_best - 1.
            # If current_best is inf, try a reasonable bound like 150.
            
            limit = best_scope - 1 if best_scope < float('inf') else 150
            
            # Backtracking can be slow. Let's add a timeout to the DFS.
            # We can pass a deadline to the DFS.
            
            last_row = backtrack_last_row(used_mask, limit, seed)
            
            if last_row:
                rows.append(last_row)
                sc = scope_of(rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [r[:] for r in rows]
                    
                    # If we found a very good solution, we might want to try to improve it further
                    # by reducing the scope even more?
                    # The loop will naturally try limit = best_scope - 1 in the next iteration.
                    
                    if best_scope <= 112:
                        break # Theoretical bound reached (or close to it)

        if best_rows:
            return best_rows
        else:
            # Fallback: Pure greedy if backtracking failed
            used_mask = 0
            rows = []
            for i in range(n_rows):
                row, used_mask = greedy_fill_row(used_mask, 200, 12345 + i)
                if row:
                    rows.append(row)
                else:
                    # Should not happen
                    break
            return rows

    return backtracking_search(200, deadline)

# Test locally
if __name__ == "__main__":
    rows = construct(4, 7)
    print(rows)
    print("Scope:", scope_of(rows))
    print("Valid:", is_valid_solution(rows))
