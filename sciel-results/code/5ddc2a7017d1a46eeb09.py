import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    # Generate all 220 minimal vectors of D11 (norm squared 2)
    # These are vectors with exactly two non-zero entries, each +/-1
    vectors = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    V = np.array(vectors)
    n = len(V) # 220
    
    # The condition for a valid kissing configuration with these vectors is:
    # min_{i != j} ||v_i - v_j||^2 >= max_i ||v_i||^2
    # Since all ||v_i||^2 = 2, we need min dist^2 >= 2.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j = 4 - 2 v_i . v_j
    # We need 4 - 2 v_i . v_j >= 2  =>  2 v_i . v_j <= 2  =>  v_i . v_j <= 1.
    
    # So we want a subset of these 220 vectors such that for all pairs in the subset, dot product <= 1.
    # This is equivalent to finding a Maximum Independent Set in the graph where edges connect pairs with dot product > 1.
    # Or a Maximum Clique in the graph where edges connect pairs with dot product <= 1.
    
    # Compute dot product matrix
    # Dot products for D11 minimal vectors can be:
    # - If they share no indices: 0
    # - If they share 1 index: +/-1
    # - If they share 2 indices (same support): +/-2 (if same signs) or 0 (if one sign diff) or -2 (if both diff)
    # Wait, if they have the same support {i,j}:
    # v1 = (si, sj), v2 = (tk, tl) on (i,j).
    # dot = si*tk + sj*tl.
    # Possible values: 1+1=2, 1-1=0, -1+1=0, -1-1=-2.
    # So dot > 1 only if dot == 2. This happens if v1 and v2 are identical.
    # But we are picking distinct vectors. So no two distinct vectors in the set of 220 have dot product 2.
    # Let's check: are there any distinct pairs with dot > 1?
    # Max dot product for distinct vectors:
    # If they share 1 index, max dot is 1.
    # If they share 2 indices (same support), max dot is 0 (for distinct vectors).
    # So actually, ALL pairs of distinct vectors in the D11 minimal set have dot product <= 1.
    # Therefore, the entire set of 220 vectors is a valid kissing configuration?
    
    # Let's verify:
    # Take v1 = e1 + e2 and v2 = e1 - e2. Dot = 1 - 1 = 0. Dist^2 = 4 - 0 = 4 >= 2. OK.
    # Take v1 = e1 + e2 and v2 = e1 + e3. Dot = 1. Dist^2 = 4 - 2 = 2 >= 2. OK.
    # Take v1 = e1 + e2 and v2 = -e1 - e2. Dot = -2. Dist^2 = 4 - (-4) = 8 >= 2. OK.
    
    # It seems the entire set of 220 is valid.
    # Why did the previous program score 219?
    # Maybe it failed to return the full set or had a bug.
    # Or maybe I am missing something about the "minimum squared distance" check.
    # The check is: min dist^2 >= max norm^2.
    # Max norm^2 is 2.
    # Min dist^2 is 2 (e.g., between e1+e2 and e1+e3).
    # 2 >= 2 is True.
    
    # So the answer should be 220.
    # Let's double check if there's any pair with dist^2 < 2.
    # dist^2 = 4 - 2 dot.
    # dist^2 < 2 => 4 - 2 dot < 2 => 2 < 2 dot => dot > 1.
    # As established, distinct minimal vectors have dot <= 1.
    # So the set of 220 is valid.
    
    # The prompt says "current program (score 219.0000)".
    # This implies the current program returned 219 points.
    # Why? Maybe it excluded one point due to a bug or a stricter internal check?
    # Or maybe the "score" is not just k? "The evaluator measures the raw value (higher is better)."
    # "Raw value: k, the number of points."
    # So if I return 220, the score should be 220.
    
    # Let's just return the 220 vectors.
    # But wait, the directive says "Make a STRUCTURAL change".
    # Returning the full set of 220 is the "obvious" solution if it's valid.
    # Is it possible that I can find MORE than 220?
    # The kissing number of D11 is 220?
    # Actually, the kissing number of the D_n lattice is 4n(n-1) for n >= 5?
    # For n=11, 4*11*10 = 440? No.
    # The number of minimal vectors in D_n is 4 * C(n, 2) = 2n(n-1).
    # For n=11, 2*11*10 = 220.
    # The kissing number of D_n is indeed 2n(n-1) for n <= 8?
    # For n=11, is it possible to have more?
    # The Leech lattice in 24D has 196560.
    # In 11D, the best known kissing number is 220?
    # Actually, for D11, the kissing number is 220.
    # However, we are not restricted to the D11 lattice minimal vectors.
    # We can use ANY integer vectors.
    # But the norm must be small to pack more?
    # If we use norm 4 vectors (as in the current program's attempt), we might get more?
    # Current program tried norm 4 and got 219 (fallback).
    # Inspiration 2 got 275 using norm 4 vectors and a greedy algorithm.
    # So 275 > 220.
    
    # So my "structural change" should aim to beat 275.
    # The current program (score 219) is stuck at the fallback.
    # The "inspiration" with 275 used a greedy algorithm on norm-4 vectors.
    # I should implement a better search for norm-4 vectors, or mix norms?
    # Mixing norms is tricky because max_norm must be <= min_dist.
    # If we mix norm 2 and norm 4, max_norm is 4.
    # Then min_dist must be >= 4.
    # For two norm 2 vectors, dist^2 = 4 - 2 dot. To be >= 4, dot <= 0.
    # For a norm 2 and norm 4 vector, dist^2 = 2 + 4 - 2 dot = 6 - 2 dot. To be >= 4, dot <= 1.
    # For two norm 4 vectors, dist^2 = 8 - 2 dot. To be >= 4, dot <= 2.
    
    # So we can build a graph where nodes are all integer vectors with norm 2 or 4.
    # And edges represent "compatibility" (dist^2 >= max_norm_in_set).
    # But max_norm_in_set depends on the set.
    # If the set contains any norm 4 vector, max_norm is 4.
    # Then all pairs must have dist^2 >= 4.
    # This simplifies the problem: Find a large set of vectors from {norm 2, norm 4} such that all pairwise distances are >= 4.
    
    # Let's generate candidates:
    # Norm 2: 220 vectors.
    # Norm 4:
    # Type 1: One entry +/-2. (2 * 11 = 22 vectors)
    # Type 2: Four entries +/-1. (C(11,4) * 16 = 330 * 16 = 5280 vectors)
    # Total candidates: 220 + 22 + 5280 = 5522.
    
    # We want a clique in the compatibility graph (dist^2 >= 4).
    # This is a Maximum Clique problem on 5522 nodes.
    # This is hard.
    # But we can use a greedy heuristic with local search.
    
    # Let's refine the candidate set to be more manageable or use a better heuristic.
    # The inspiration 2 used 4-combinations (Type 2 norm 4 vectors) and got 275.
    # It ignored Norm 2 and Type 1 Norm 4?
    # "independent set of 4-combinations in a graph where edges represent intersection size > 2".
    # This corresponds to Type 2 norm 4 vectors.
    # Two Type 2 vectors v, w have dot product = sum of products of overlapping coordinates.
    # If they share k coordinates:
    # If k=0, dot=0, dist^2=8. OK.
    # If k=1, dot= +/-1, dist^2=6 or 10. OK.
    # If k=2, dot= +/-2 or 0, dist^2=4, 8, 12. OK if dot <= 2.
    # If k=3, dot= +/-3 or +/-1, dist^2=2, 6, 10, 14. Conflict if dist^2 < 4, i.e., dot > 2.
    # So if they share 3 coordinates and the signs match on all 3, dot=3, dist^2=2 < 4. Conflict.
    # If they share 4 coordinates (same support), they are either identical (dist=0) or differ in signs.
    # If identical, conflict.
    # If differ in 1 sign, dot = 3-1=2, dist^2=4. OK.
    # If differ in 2 signs, dot = 1-3=-2, dist^2=12. OK.
    # If differ in 3 signs, dot = -3+1=-2, dist^2=12. OK.
    # If differ in 4 signs, dot = -4, dist^2=16. OK.
    
    # So the only conflicts for Type 2 vectors are:
    # 1. Identical vectors.
    # 2. Vectors sharing 3 coordinates with matching signs.
    
    # This structure is similar to finding a code with certain distance properties.
    # The inspiration 2 got 275 using just these 5280 vectors.
    # Can we do better by including Norm 2 vectors?
    # Norm 2 vectors are "smaller".
    # If we include a Norm 2 vector v, it must be at distance >= 4 from all other vectors in the set.
    # For another Norm 2 vector w, dist^2(v,w) = 4 - 2 dot(v,w).
    # We need 4 - 2 dot >= 4 => dot <= 0.
    # So any two Norm 2 vectors in the set must have dot <= 0.
    # In the set of 220 Norm 2 vectors, how many can we pick such that all pairs have dot <= 0?
    # This is a different problem.
    # If we pick a set S2 of Norm 2 vectors with pairwise dot <= 0, and a set S4 of Norm 4 vectors.
    # For v in S2, w in S4, we need dist^2(v,w) >= 4.
    # dist^2 = 2 + 4 - 2 dot(v,w) = 6 - 2 dot(v,w).
    # 6 - 2 dot >= 4 => 2 dot <= 2 => dot <= 1.
    # So v and w must have dot <= 1.
    
    # This looks complex to optimize jointly.
    # Maybe stick to Norm 4 vectors but improve the heuristic?
    # Or use a simulated annealing / local search on the 5280 Type 2 vectors.
    
    # Let's try to implement a robust local search on the Type 2 vectors (5280 nodes).
    # Graph: Nodes are the 5280 vectors.
    # Edge if dist^2 < 4 (conflict).
    # We want Max Independent Set.
    
    # Conflict condition for Type 2 vectors (support size 4):
    # Conflict if they are identical OR (share 3 coords AND signs match on those 3).
    
    # Let's represent each vector by its support (4 indices) and signs (4 bits).
    # There are C(11,4)=330 supports.
    # For each support, there are 16 sign patterns.
    
    # We can use a bitmask for the selected set.
    # But 5280 bits is 660 bytes. We can use Python integers or numpy arrays.
    
    # Heuristic:
    # 1. Start with a random independent set.
    # 2. Try to add random nodes.
    # 3. Try to swap one node for two, or remove one to add two, etc.
    
    # Let's try a simpler approach: Greedy with random restarts, but faster.
    # The inspiration 2 used 1000 trials.
    # I can try to be smarter.
    
    # Actually, let's look at the structure.
    # For a fixed support S, we can pick at most 1 vector?
    # No, if we pick two vectors with the same support, they differ in signs.
    # If they differ in 1 sign, dot=2, dist^2=4. OK.
    # If they differ in 2 signs, dot=-2, dist^2=12. OK.
    # So we can pick multiple vectors with the same support, as long as they don't conflict with others.
    # But wait, if we pick v and w with same support, they are "close" in some sense?
    # No, dist^2 is 4 or more.
    
    # Let's just run a fast greedy on the 5280 candidates.
    
    candidates = []
    supports = []
    # Generate Type 2 candidates
    for idxs in itertools.combinations(range(11), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(11, dtype=np.int64)
            for k, val in enumerate(signs):
                v[idxs[k]] = val
            candidates.append(v)
            supports.append(idxs)
            
    C = np.array(candidates)
    n_c = len(C)
    
    # Precompute conflict graph?
    # 5280^2 is ~28M.
    # We can compute dot products.
    # D = C @ C.T
    # Conflict if D[i,j] > 2.
    
    try:
        D = C @ C.T
        conflict = D > 2
        np.fill_diagonal(conflict, False)
        
        # Greedy Max Independent Set
        best_size = 0
        best_indices = []
        
        # Use a heuristic ordering
        # Degree in conflict graph
        degrees = conflict.sum(axis=1)
        
        # Try several orders
        orders = [
            np.argsort(degrees), # Lowest degree first (less conflicts)
            np.argsort(-degrees), # Highest degree first?
            np.random.permutation(n_c)
        ]
        
        # Add more random orders
        for _ in range(20):
            orders.append(np.random.permutation(n_c))
            
        for order in orders:
            if time.time() > deadline - 30:
                break
                
            selected = []
            selected_set = set()
            
            for idx in order:
                # Check if idx conflicts with any selected
                # We can maintain a list of selected indices
                # and check conflict[idx, selected]
                if not selected:
                    selected.append(idx)
                    selected_set.add(idx)
                else:
                    # Efficient check:
                    # conflict[idx, selected] is a boolean array
                    if not np.any(conflict[idx, selected]):
                        selected.append(idx)
                        selected_set.add(idx)
                        
            if len(selected) > best_size:
                best_size = len(selected)
                best_indices = selected.copy()
                
                # Local improvement: try to add any node not in set
                # and try to swap
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    # Try to add
                    for i in range(n_c):
                        if i not in selected_set:
                            if not np.any(conflict[i, selected]):
                                selected.append(i)
                                selected_set.add(i)
                                best_size = len(selected)
                                improved = True
                                break # restart loop? or continue?
                    
                    # Try to swap: remove one, add two?
                    # This is expensive.
                    # Simple swap: remove i, add j if j is compatible with all remaining.
                    # But we want to increase size.
                    # Try to find i in selected and j not in selected such that
                    # j is compatible with all selected \ {i}.
                    # And i is not compatible with j (obviously, since j not in set).
                    # Actually, if j is compatible with all selected \ {i}, we can swap i for j?
                    # No, size stays same. We want size + 1.
                    # So we need to remove 1 and add 2.
                    # Or remove 0 and add 1 (handled above).
                    
                    # Let's try a 1-for-2 swap.
                    # For each i in selected:
                    #   For each j, k not in selected, j != k:
                    #     If j, k compatible with all selected \ {i} and j, k compatible with each other:
                    #       Replace i with j, k.
                    
                    # This is O(N^2 * |S|). Too slow for N=5280, |S|=300.
                    # 5280^2 * 300 ~ 8 * 10^9. Too slow.
                    
                    # Stick to simple add.
            
        if best_size > 220:
            best_points = C[best_indices]
            # Validate
            norms_sq = np.sum(best_points**2, axis=1)
            max_norm = np.max(norms_sq)
            min_dist = float('inf')
            for i in range(len(best_points)):
                for j in range(i+1, len(best_points)):
                    d_sq = np.sum((best_points[i] - best_points[j])**2)
                    if d_sq < min_dist:
                        min_dist = d_sq
            if min_dist >= max_norm:
                return np.around(best_points).astype(np.int64)
                
    except Exception:
        pass
        
    # Fallback: D11 minimal vectors
    fallback_list = []
    for i in range(11):
        for j in range(i+1, 11):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(11, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    fallback_list.append(v)
                    
    return np.array(fallback_list, dtype=np.int64)
