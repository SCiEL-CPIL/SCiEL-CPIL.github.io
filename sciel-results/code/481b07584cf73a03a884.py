"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    n = 14
    m = 35
    maximum = 10000
    
    # Helper to compute the objective
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    # Base solution: K9 on vertices 0-8 (values 1-9), excluding edge (7,8)
    # Values: 1, 2, ..., 14
    base_values = list(range(1, 15))
    base_edges = []
    for i in range(9):
        for j in range(i+1, 9):
            if i == 7 and j == 8:
                continue
            base_edges.append([i, j])
            
    best_score = _raw_value(base_values, base_edges)
    best_values = base_values[:]
    best_edges = [e[:] for e in base_edges]
    
    # We have 35 edges. K9 has 36. We removed one.
    # The structure is fixed to the first 9 vertices.
    # Can we improve by changing values?
    # The values 1..9 give specific sums and products.
    # Sums in K9\{(7,8)}:
    # Products in K9\{(7,8)}:
    
    # Let's try to search for better value sets.
    # Since the graph is fixed (indices 0..8), we only need to assign values to these 9 indices.
    # The other 5 indices (9..13) are isolated, so their values don't affect the score,
    # as long as they are distinct from the first 9 and <= 10000.
    
    # Strategy:
    # 1. Try random permutations of 1..9 for the active vertices.
    # 2. Try random subsets of size 9 from 1..100.
    # 3. Try random subsets of size 9 from 1..1000.
    
    # Optimization: Pre-compute the edge pairs (0..8)
    active_edges = [(i, j) for i in range(9) for j in range(i+1, 9) if not (i==7 and j==8)]
    
    def evaluate_active(active_vals):
        # active_vals is a list of 9 values for indices 0..8
        sums = set()
        prods = set()
        for i, j in active_edges:
            v = active_vals[i]
            w = active_vals[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    # Initial evaluation
    current_active = base_values[:9]
    current_score = evaluate_active(current_active)
    
    # Search 1: Permutations of 1..9
    # There are 9! = 362,880 permutations.
    # We can sample some.
    # Actually, since the graph is symmetric under vertex relabeling?
    # No, the graph is not complete, so vertex degrees matter.
    # Degrees in K9\{(7,8)}:
    # Vertices 0..6: degree 8 (connected to all other 8 in 0..8)
    # Vertex 7: degree 7 (connected to 0..6)
    # Vertex 8: degree 7 (connected to 0..6)
    # So vertices 7 and 8 are symmetric. Vertices 0..6 are symmetric.
    # So we only need to assign values to two groups: {7,8} and {0..6}.
    
    # Let's try to optimize the values assigned to the two "low degree" vertices (7,8)
    # vs the "high degree" vertices (0..6).
    
    # Let's try random sets of 9 distinct integers from 1..1000
    random.seed(42)
    
    # We will try a few hundred random sets
    for _ in range(2000):
        if time.time() > deadline - 5:
            break
            
        # Random set of 9 distinct values
        # Try small values first, as they tend to have more collisions in sums/products?
        # Or maybe large values are better to avoid collisions?
        # Actually, for products, large values might produce unique products easily.
        # For sums, small values might collide more.
        # Let's try a mix.
        
        # Sample from 1..50
        vals = random.sample(range(1, 51), 9)
        
        # Assign to vertices.
        # Due to symmetry, we can fix the assignment of the two largest/smallest to 7,8?
        # Let's just randomize the permutation of the 9 values onto the 9 vertices.
        # But since 0..6 are symmetric, we can sort them and assign to 0..6.
        # And assign the remaining two to 7,8 (order doesn't matter for 7,8 as they are symmetric).
        
        # So:
        # Pick 2 values for {7,8} and 7 for {0..6}.
        # To reduce search space, we can assume wlog that the values at 7,8 are the two smallest?
        # Or two largest?
        # Let's just try all 9 choose 2 ways to pick which values go to 7,8.
        # That's 36 combinations.
        
        # Actually, let's just do a random permutation of the 9 values onto the 9 vertices.
        # It's fast enough.
        perm = vals[:]
        random.shuffle(perm)
        
        score = evaluate_active(perm)
        if score < current_score:
            current_score = score
            current_active = perm[:]
            
            # Update best_values and best_edges
            # We need to fill in the remaining 5 values.
            # They must be distinct from current_active and <= 10000.
            used = set(current_active)
            remaining = []
            # Pick 5 random values not in used
            candidates = [x for x in range(1, 10001) if x not in used]
            # This is slow if done naively.
            # Better: just pick 5 specific values, e.g., 101..105 if not used.
            # Or simply:
            for val in range(1, 10001):
                if val not in used:
                    remaining.append(val)
                    if len(remaining) == 5:
                        break
            
            new_vals = current_active + remaining
            # Check distinctness
            if len(set(new_vals)) == 14:
                best_score = score
                best_values = new_vals
                # Edges remain the same indices
                # best_edges is already correct for indices 0..8
                
    # One more try: specific heuristic
    # Try values 1..9 but permuted such that small values are at high degree nodes?
    # High degree nodes (0..6) participate in more sums/products.
    # If we put small values there, sums might be smaller and collide more?
    # If we put large values there, sums are larger.
    
    # Let's try the base case again to ensure we have a valid fallback.
    # base_score was 29.
    
    # If we didn't find anything better, return base.
    if current_score < best_score:
         # This logic is slightly off, we should just return the best found.
         # Let's re-evaluate.
         pass
         
    # Actually, let's just return the best_values and best_edges we tracked.
    # Ensure types are correct.
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
