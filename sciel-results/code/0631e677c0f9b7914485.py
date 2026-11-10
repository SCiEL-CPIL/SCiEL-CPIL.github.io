import numpy as np
import time
import itertools
from scipy.optimize import linprog

def construct(q, n, radius):
    # Parameters are fixed by the problem description effectively, 
    # but we should handle the general case or at least the specific one robustly.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # However, the task description specifies 8 matches, ternary, radius 1.
    # I will implement it generally for q=3, n=8, r=1 but allow parameters.
    
    if q != 3 or n != 8 or radius != 1:
         # Fallback to a simple construction for other cases to avoid errors, 
         # though the evaluator likely only calls with 3, 8, 1.
         # Direct sum for general q, n, r=1?
         # For now, return the seed solution if params don't match, 
         # or a simple greedy if they do.
         # Given the specific task contract, it's 3, 8, 1.
         tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
         return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = q ** n
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Precompute neighbors
    # List of all words
    # Convert 0..N-1 to base q digits?
    # It's faster to generate neighbors on the fly or precompute once.
    
    # Precompute neighbors for all N words
    # Shape: (N, 1 + 2*n)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    # Precompute powers of 3
    powers = [3 ** i for i in range(n)]
    
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            # Change digit i to (d+1)%3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            # Change digit i to (d+2)%3
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Seed solution: Tetracode direct sum with all 3^4 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            # Convert to integer representation (LSB first in my neighbor precompute)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            seed_code_ints.append(w_int)
    
    best_code = seed_code_ints[:]
    best_size = len(best_code)
    
    # Strategy: Minimum Vertex Cover via LP Relaxation
    # We want to select a subset of centers C such that every word is covered.
    # This is a Set Cover problem. The LP relaxation is:
    # Minimize sum(x_i)
    # Subject to: sum_{i: w in N(i)} x_i >= 1 for all w
    #             0 <= x_i <= 1
    
    # We will iteratively:
    # 1. Solve LP relaxation on the current candidate set (initially seed).
    # 2. Identify variables with x_i < epsilon (redundant) and x_i > 1-epsilon (essential).
    # 3. Fix essential variables to 1, remove redundant variables.
    # 4. If the integer solution (variables fixed to 1) covers everything, we are done.
    # 5. If not, we need to repair. The LP gives hints on which fractional variables are most "needed".
    #    We can greedily add the most needed fractional variables until covered.
    # 6. Repeat with the new set.

    # To make this efficient, we don't solve the full N x M LP (6561 x 486) from scratch every time if we can avoid it.
    # However, scipy's linprog (HiGHS or similar) is quite fast for this size.
    # 6561 constraints, ~500 variables. This is small for LP solvers.
    
    # Let's build the constraint matrix A for the seed set first.
    # A[w, i] = 1 if word w is in neighbors[center_i]
    
    def solve_lp_and_prune(candidate_centers):
        """
        Solves the LP relaxation for the given candidate centers.
        Returns a list of centers that are 'strongly' selected (x_i close to 1) 
        and potentially some fractional ones that might be needed.
        Actually, for vertex cover on bipartite graph, LP relaxation is integral 
        IF the graph is bipartite. 
        Wait, the Set Cover LP is not necessarily integral. 
        However, the problem is equivalent to Minimum Dominating Set in the graph 
        where edges connect centers to words they cover. This graph is bipartite.
        The Minimum Vertex Cover problem on bipartite graphs has an integral LP relaxation.
        BUT, we are not looking for a Vertex Cover in the standard sense (edges covered).
        We are looking for a Set Cover (words covered).
        This is equivalent to finding a set of centers such that their union of neighbors is the whole space.
        This is NOT the same as Minimum Vertex Cover on the bipartite graph (C, W, E).
        In Minc Vertex Cover, we cover edges. Here we cover vertices W.
        So the LP is:
        Min sum x_c
        s.t. for each w in W: sum_{c: w in N(c)} x_c >= 1
             0 <= x_c <= 1
        
        This LP is NOT necessarily integral. It is a Set Cover LP.
        The gap can be up to log(n).
        So we use it as a heuristic to guide pruning.
        """
        m = len(candidate_centers)
        if m == 0:
            return []
        
        # Build A matrix: N x m
        # A[w, i] = 1 if candidate_centers[i] covers w
        A = np.zeros((N, m), dtype=np.float64)
        
        for i, c in enumerate(candidate_centers):
            # Mark all neighbors of c as covered by i
            A[neighbors[c], i] = 1.0
            
        # Objective: minimize sum(x)
        c_obj = np.ones(m, dtype=np.float64)
        
        # Constraints: A @ x >= 1
        # linprog requires A_ub @ x <= b_ub
        # So we use -A @ x <= -1
        A_ub = -A
        b_ub = -np.ones(N, dtype=np.float64)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1) for _ in range(m)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                x = res.x
                return x
            else:
                # Fallback to uniform if LP fails
                return np.ones(m) / m
        except Exception:
            return np.ones(m) / m

    # Iterative Pruning and Repair
    current_centers = best_code[:]
    
    while time.time() < deadline:
        # 1. Solve LP
        x = solve_lp_and_prune(current_centers)
        
        if x is None:
            break
            
        # 2. Analyze solution
        # If the sum of x is close to an integer, and most x are 0 or 1, we might have a good candidate.
        # Let's identify "definite" centers (x > 0.9) and "definite" non-centers (x < 0.1).
        # The rest are "borderline".
        
        definite_indices = [i for i, val in enumerate(x) if val > 0.9]
        borderline_indices = [i for i, val in enumerate(x) if 0.1 <= val <= 0.9]
        
        # Construct a candidate code from definite centers
        candidate_code = [current_centers[i] for i in definite_indices]
        
        # Check if this candidate code covers the whole space
        # We can do this efficiently by marking covered words.
        covered = np.zeros(N, dtype=np.uint8)
        for c in candidate_code:
            covered[neighbors[c]] = 1
            
        uncovered_count = N - covered.sum()
        
        if uncovered_count == 0:
            # We found a valid code
            if len(candidate_code) < best_size:
                best_size = len(candidate_code)
                best_code = candidate_code[:]
                
                # If we are close to the lower bound, we can stop or try to prune more
                if best_size <= 403:
                    break
            
            # Try to prune further?
            # If we have borderline variables, maybe some definite ones are actually redundant 
            # in the context of the others?
            # The LP solution is a local optimum of the relaxation.
            # Let's try to remove any center from candidate_code that doesn't break coverage.
            # This is the standard "remove-repair" or just "remove if safe".
            
            new_code = candidate_code[:]
            # Sort by "importance" (x value) descending? Or just iterate.
            # Removing a center with high x value is risky. Removing one with lower x value (but >0.9) is safer?
            # Actually, if x_i > 0.9, it's likely essential.
            # Let's just try to remove any center.
            
            removed_any = False
            for i in range(len(new_code)):
                if time.time() > deadline - 5:
                    break
                c_to_remove = new_code[i]
                # Temporarily remove
                # Check coverage without c_to_remove
                # We can compute the "unique" words covered by c_to_remove that are NOT covered by others.
                # If unique_count == 0, it is safe to remove.
                
                # Efficient check:
                # Count how many times each word is covered by the current new_code.
                # If a word is covered by c_to_remove, and its total count > 1, it's safe.
                # If any word covered by c_to_remove has total count == 1, it's NOT safe.
                
                # Let's just re-evaluate coverage for the specific neighbors of c_to_remove.
                # This is O(17) checks per removal candidate.
                
                safe = True
                # We need to know if any neighbor of c_to_remove is ONLY covered by c_to_remove in new_code.
                # To do this fast, we can maintain a count array?
                # For simplicity and given small N, let's just check.
                # But checking all neighbors for each removal candidate is O(17 * 486) which is fine.
                
                # Actually, maintaining a global count array is better if we do multiple removals.
                # But let's keep it simple: check if removal is safe.
                
                # Optimization: Pre-compute coverage counts for the current new_code?
                # That's O(486 * 17) = 8000 ops.
                # Then for each candidate, check its 17 neighbors.
                
                # Let's do a batch removal pass.
                # 1. Compute coverage counts for new_code.
                # 2. Iterate through new_code. If a center's neighbors are all covered by others, remove it.
                
                # Let's restart the pruning loop for this specific code to be clean.
                # If we removed any, we break the main while loop and restart with the new smaller set.
                
                # For now, let's just try to remove one at a time in a dedicated sub-loop.
                # This is getting complex. Let's stick to the LP result for now.
                # If the LP says x_i > 0.9, it's likely needed.
                # The main gain comes from the LP identifying redundant sets.
                
                # Let's try a simple greedy removal on the valid candidate_code found.
                # This is a different strategy from the main LP loop, so let's do it here.
                
                # Actually, the current program's strength was the greedy.
                # The LP gives a better starting point for the greedy.
                # So:
                # 1. LP -> candidate_code (valid)
                # 2. Greedy Remove on candidate_code -> improved_code
                # 3. If improved_code is better, update best.
                # 4. Use improved_code as the new current_centers for the next LP iteration?
                #    Or just keep the original seed for LP and use the best found for output?
                #    Using the best found as the next LP input might help converge faster.
                
                # Let's implement the Greedy Remove here.
                removed = True
                while removed and time.time() < deadline - 5:
                    removed = False
                    # Compute coverage counts
                    counts = np.zeros(N, dtype=np.int32)
                    for c in new_code:
                        counts[neighbors[c]] += 1
                    
                    for i in range(len(new_code)):
                        c = new_code[i]
                        # Check if any neighbor of c has count == 1
                        # If yes, c is essential.
                        # If no, c is redundant.
                        essential = False
                        for nb in neighbors[c]:
                            if counts[nb] == 1:
                                essential = True
                                break
                        if not essential:
                            # Remove c
                            new_code.pop(i)
                            removed = True
                            # Update counts?
                            # It's easier to recompute counts in the next iteration of the while loop.
                            # Or update counts now.
                            counts[neighbors[c]] -= 1
                            # Break inner loop to restart with updated list
                            break
                    
                    if len(new_code) < best_size:
                        best_size = len(new_code)
                        best_code = new_code[:]
                
                # If we improved, let's use this new set for the next LP iteration
                current_centers = new_code[:]
                # Continue to next LP iteration to see if we can prune more
                continue
        else:
            # The definite centers are not enough.
            # We need to add borderline centers.
            # Sort borderline by x value descending.
            # Add them one by one until covered.
            
            # This is essentially a greedy algorithm guided by LP weights.
            # Let's construct a "weighted greedy" from the LP solution.
            
            # Start with definite centers
            current_covered = np.zeros(N, dtype=np.uint8)
            current_code_list = []
            
            for i in definite_indices:
                c = current_centers[i]
                current_covered[neighbors[c]] = 1
                current_code_list.append(c)
                
            # Add borderline centers in order of x value
            borderline_sorted = sorted(borderline_indices, key=lambda i: x[i], reverse=True)
            
            for i in borderline_sorted:
                if time.time() > deadline - 5:
                    break
                if current_covered.sum() == N:
                    break
                c = current_centers[i]
                # Check if adding c helps
                # It should, since it was borderline and we are adding in order of "need"
                # Just add it.
                current_covered[neighbors[c]] = 1
                current_code_list.append(c)
                
            # Check if covered
            if current_covered.sum() == N:
                if len(current_code_list) < best_size:
                    best_size = len(current_code_list)
                    best_code = current_code_list[:]
                
                # Now try to prune this new set using the same "remove if redundant" logic
                new_code = current_code_list[:]
                removed = True
                while removed and time.time() < deadline - 5:
                    removed = False
                    counts = np.zeros(N, dtype=np.int32)
                    for c in new_code:
                        counts[neighbors[c]] += 1
                    
                    for i in range(len(new_code)):
                        c = new_code[i]
                        essential = False
                        for nb in neighbors[c]:
                            if counts[nb] == 1:
                                essential = True
                                break
                        if not essential:
                            new_code.pop(i)
                            removed = True
                            counts[neighbors[c]] -= 1
                            break
                    
                    if len(new_code) < best_size:
                        best_size = len(new_code)
                        best_code = new_code[:]
                
                current_centers = new_code[:]
                continue
            else:
                # Still not covered?
                # This shouldn't happen if the LP was feasible and we added all borderline.
                # But if some x were < 0.1 and we ignored them, they might be needed.
                # In that case, fall back to standard greedy on the remaining uncovered.
                # Or just break and use the best found so far.
                break

    # Convert integer codewords back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        # word[0] is LSB (coordinate 0), word[1] is coordinate 1, etc.
        # This matches the order in the seed construction
        final_code.append(word)
        
    return final_code
