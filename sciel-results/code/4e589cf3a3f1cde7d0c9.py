import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Precompute LCM to bound the search for m
    # LCM of divisors is 30030
    max_m = 30030
    
    best_R = float('inf')
    best_solution = None
    start_time = time.time()
    time_limit = 85.0 # Leave margin for evaluation overhead
    
    # Sort indices by divisor size? 
    # Actually, for backtracking, we want to order by the number of candidates in the interval [m, m+R].
    # The number of candidates for d_i is roughly R/d_i.
    # Smaller d_i -> more candidates. Larger d_i -> fewer candidates.
    # So we should try variables with larger d_i first in backtracking to prune faster?
    # Or variables with fewer candidates.
    # Let's just use the natural order or sort by divisor descending.
    
    # We will iterate m from 1 to max_m
    # For each m, we try R from 9 upwards
    # If we find a solution, we update best_R and best_solution
    # We can stop searching R for this m if R >= best_R
    
    for m in range(1, max_m + 1):
        if time.time() - start_time > time_limit:
            break
            
        # If best_R is 9, we can't do better, so we can break entirely?
        # 9 is the minimum possible range for 10 distinct integers.
        if best_R == 9:
            break
            
        # Try increasing R
        # We can start R from 9
        for R in range(9, best_R):
            # Check if a valid assignment exists for this m and R
            # Candidates for x_i are multiples of d_i in [m, m+R]
            # Let x_i = m + delta_i
            # delta_i must be such that m + delta_i is divisible by d_i
            # delta_i ≡ -m (mod d_i)
            # 0 <= delta_i <= R
            
            candidates = []
            for i, d in enumerate(divisors):
                # Find smallest delta >= 0 such that (m + delta) % d == 0
                # (m + delta) % d == 0 => delta % d == (-m) % d
                r = (-m) % d
                if r > R:
                    # No candidate in range
                    candidates.append([])
                    continue
                # Candidates: r, r+d, r+2d, ... <= R
                deltas = []
                curr = r
                while curr <= R:
                    deltas.append(curr)
                    curr += d
                candidates.append(deltas)
            
            # If any candidate list is empty, no solution for this R
            if any(len(c) == 0 for c in candidates):
                continue
                
            # Backtracking to find distinct deltas
            # Order variables by length of candidate list (ascending) to prune faster
            # But we need to map back to original indices
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            def backtrack(idx, used_deltas):
                if idx == n:
                    return True
                i = order[idx]
                for delta in candidates[i]:
                    if delta not in used_deltas:
                        used_deltas.add(delta)
                        if backtrack(idx + 1, used_deltas):
                            return True
                        used_deltas.remove(delta)
                return False
            
            used = set()
            if backtrack(0, used):
                # Solution found
                best_R = R
                # Construct the solution
                # We need to recover the deltas. The backtrack above doesn't store them.
                # Let's modify backtrack to store the assignment.
                
                # Re-run backtrack with storage
                assignment = [None] * n
                
                def backtrack_store(idx, used_deltas):
                    if idx == n:
                        return True
                    i = order[idx]
                    for delta in candidates[i]:
                        if delta not in used_deltas:
                            used_deltas.add(delta)
                            assignment[i] = delta
                            if backtrack_store(idx + 1, used_deltas):
                                return True
                            used_deltas.remove(delta)
                            assignment[i] = None
                    return False
                
                if backtrack_store(0, set()):
                    solution = [m + assignment[i] for i in range(n)]
                    # Verify distinctness and divisibility
                    if len(set(solution)) == n and all(solution[i] % divisors[i] == 0 for i in range(n)):
                        best_solution = solution
                        # If R is 9, we can stop
                        if R == 9:
                            break
            # End if backtrack succeeded
            
        # End for R
        
    # End for m
    
    if best_solution is None:
        # Fallback: return divisors themselves (valid but not optimal)
        return divisors[:]
        
    return best_solution

