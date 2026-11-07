"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges (indices)
    all_edges = list(combinations(range(n), 2))
    
    # Helper to check validity of the final result
    def is_valid(values, edges):
        if len(values) != n: return False
        if len(set(values)) != n: return False
        if not all(isinstance(v, int) and 1 <= v <= max_val for v in values): return False
        if len(edges) != m: return False
        edge_set = set(tuple(sorted(e)) for e in edges)
        if len(edge_set) != m: return False
        # Check no loops
        for e in edges:
            if e[0] == e[1]: return False
        return True

    def get_score(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    # CSP Solver: Find m edges such that distinct sums <= K and distinct prods <= K
    # Returns list of edges or None if not found within time limit
    def solve_csp(values, K, time_limit=2.0):
        start = time.time()
        
        # Precompute sums and products for all edges
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # We will select edges one by one.
        # State: (current_edge_index_in_selection, set_of_chosen_edges, set_of_sums, set_of_prods)
        # To make it fast, we use recursion with backtracking.
        # Optimization: Sort edges? Or just iterate.
        # Given m=35 out of 91, branching factor is high. 
        # However, K is small (likely < 20). This prunes heavily.
        
        best_solution = [None]
        
        def backtrack(start_idx, chosen_edges, chosen_sums, chosen_prods):
            if time.time() > time_limit:
                return False
            
            if len(chosen_edges) == m:
                best_solution[0] = chosen_edges[:]
                return True
            
            # Heuristic: Try edges that add fewest new sums/prods first?
            # Or just iterate all available edges.
            # To speed up, we can filter edges that are "safe" (sum/prod already in set OR set size < K)
            
            # Generate candidates
            candidates = []
            for i in range(start_idx, len(all_edges)):
                if all_edges[i] in chosen_edges:
                    continue
                
                s = edge_sums[i]
                p = edge_prods[i]
                
                # Check if adding this edge violates K
                new_sums_count = len(chosen_sums) + (0 if s in chosen_sums else 1)
                new_prods_count = len(chosen_prods) + (0 if p in chosen_prods else 1)
                
                if new_sums_count <= K and new_prods_count <= K:
                    # Score for ordering: prefer edges that don't add new values (more "tight")
                    # Or maybe prefer edges that add values to keep options open?
                    # Usually, filling with existing values is safer to stay within K.
                    cost = (0 if s in chosen_sums else 1) + (0 if p in chosen_prods else 1)
                    candidates.append((cost, i))
            
            # Sort by cost (prefer re-using sums/prods)
            candidates.sort(key=lambda x: x[0])
            
            for cost, i in candidates:
                u, v = all_edges[i]
                chosen_edges.append((u, v))
                chosen_sums.add(edge_sums[i])
                chosen_prods.add(edge_prods[i])
                
                if backtrack(i + 1, chosen_edges, chosen_sums, chosen_prods):
                    return True
                    
                chosen_edges.pop()
                chosen_sums.remove(edge_sums[i])
                chosen_prods.remove(edge_prods[i])
                
            return False

        backtrack(0, [], set(), set())
        return best_solution[0]

    # Main Search Strategy:
    # 1. Generate value sets.
    # 2. For each value set, try increasing K until a solution is found or time runs out.
    # 3. Keep the best (lowest K) solution.
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate a pool of candidate value sets
    candidate_values_list = []
    
    # 1. Small integers
    candidate_values_list.append(list(range(1, n+1)))
    
    # 2. Random small integers (likely to have collisions)
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        vals = [random.randint(1, 15) for _ in range(n)]
        if len(set(vals)) == n:
            candidate_values_list.append(vals)
            
    # 3. Structured sets (e.g., powers, or specific patterns)
    # e.g., 1, 2, 4, 8... (products grow fast, sums collide less)
    # e.g., 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 (already tried)
    # Try a set with many repeated sum structures?
    # e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    
    # 4. Local search on the best value set found so far
    # We will iterate through candidates, solve CSP, and if we find a solution, 
    # we might try to improve values slightly.
    
    for vals in candidate_values_list:
        if time.time() > deadline - 5:
            break
            
        # Try K from a low bound up to a reasonable max
        # Theoretical min K is around sqrt(m) or so? 
        # With 35 edges, if all sums are distinct, K=35. 
        # We want K < 35.
        # Let's try K from 10 to 30.
        
        found_solution = False
        for K in range(10, 31):
            if time.time() > deadline - 5:
                break
                
            edges = solve_csp(vals, K, time_limit=1.0)
            
            if edges:
                score = get_score(vals, edges)
                # The score is max(distinct_sums, distinct_prods). 
                # Since we enforced <= K, score <= K.
                # It might be less than K if we didn't fill all slots? No, we fill m edges.
                # But distinct counts might be less than K if we didn't use all allowed slots.
                
                if score < best_score:
                    best_score = score
                    best_values = vals[:]
                    best_edges = edges[:]
                    found_solution = True
                    break # Found a solution for this K, try next value set or higher K? 
                    # Actually, if we found a solution for K, score <= K. 
                    # We want to minimize score. 
                    # If we found a solution for K=15, score is <= 15.
                    # We should check if we can get a better score with a different value set 
                    # or by trying lower K? No, we started from low K. 
                    # So if we found solution for K=15, the score is exactly the max distinct counts.
                    # We should continue to other value sets to see if they allow a lower K.
                    
        # If we found a solution, we might want to do a quick local search on values to see if we can lower K further?
        # For now, just collect candidates.

    # If we didn't find anything good, fallback
    if best_edges is None:
        # Fallback to a simple heuristic result
        vals = list(range(1, n+1))
        # Just pick 35 edges that seem "dense"
        # Greedy fill
        edges = []
        sums = set()
        prods = set()
        for e in all_edges:
            s = vals[e[0]] + vals[e[1]]
            p = vals[e[0]] * vals[e[1]]
            if s in sums and p in prods:
                edges.append(list(e))
            elif len(sums) < 35 and len(prods) < 35:
                # Add if it doesn't blow up too much?
                # Just add to ensure we have 35
                edges.append(list(e))
                sums.add(s)
                prods.add(p)
            if len(edges) == 35:
                break
        # If still not 35, fill with random
        while len(edges) < 35:
            e = random.choice(all_edges)
            if e not in edges:
                edges.append(list(e))
        
        best_values = vals
        best_edges = edges
        best_score = get_score(best_values, best_edges)
        
    # Final Verification and Type Conversion
    final_values = [int(v) for v in best_values]
    final_edges = [list(map(int, e)) for e in best_edges]
    
    if not is_valid(final_values, final_edges):
        # Emergency fallback
        return {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:35]]}
        
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
