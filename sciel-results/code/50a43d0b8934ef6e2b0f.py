"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Candidate Pool
    # We focus on weight-4 vectors (4 non-zero coordinates, values +/-1).
    # Total number of weight-4 vectors in R^11 is C(11,4) * 2^4 = 330 * 16 = 5280.
    # This is a manageable size for pairwise distance checks if done efficiently.
    # 
    # Strategy: 
    # The kissing number problem for equal norm vectors (norm 2) in Z^11.
    # Condition: ||v_i - v_j||^2 >= ||v_i||^2 = 4.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j = 8 - 2 v_i.v_j.
    # So 8 - 2 v_i.v_j >= 4  =>  4 >= 2 v_i.v_j  =>  v_i.v_j <= 2.
    # 
    # For weight-4 vectors, the dot product v_i.v_j is the sum of products of overlapping coordinates.
    # If they share k coordinates with same sign and l with opposite sign, dot = k - l.
    # Max possible dot product is 4 (identical vectors).
    # We need dot <= 2.
    # 
    # Let's generate all weight-4 vectors.
    candidates = []
    
    # Indices for 4 non-zero coordinates
    for comb in itertools.combinations(range(D), 4):
        # Signs for these 4 coordinates
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(comb, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Precompute norms squared (all are 4)
    # norms_sq = np.sum(candidates * candidates, axis=1) # All 4
    
    # Precompute Gram matrix? 5280 x 5280 is ~28M entries. 28M * 8 bytes = 224 MB.
    # This might be too large for memory or slow to compute in pure Python loops, 
    # but numpy matrix multiplication is fast.
    # Let's try to compute the Gram matrix G where G[i,j] = candidates[i] @ candidates[j].
    
    # Check memory/time. 5280x11 @ 11x5280.
    # This is a standard BLAS operation. Should take < 1s.
    
    G = candidates @ candidates.T
    
    # Compatibility condition: G[i,j] <= 2 for i != j.
    # Note: G[i,i] = 4.
    
    # We want to select the largest subset of indices such that for all i,j in subset, G[i,j] <= 2.
    # This is equivalent to finding a maximum clique in the graph where an edge exists if G[i,j] <= 2.
    # Or, finding the maximum independent set in the graph where an edge exists if G[i,j] > 2.
    
    # Let's define the "conflict" graph: edge (i,j) if G[i,j] > 2.
    # We want the largest set of nodes with no edges between them.
    # This is the Maximum Independent Set problem, which is NP-hard.
    # However, for this specific geometry, greedy heuristics often work well.
    
    # Greedy Min-Degree Heuristic for Maximum Independent Set:
    # 1. Start with all nodes.
    # 2. Repeatedly pick the node with the minimum degree (fewest conflicts) in the remaining graph.
    # 3. Add it to the independent set.
    # 4. Remove it and its neighbors from the graph.
    
    # To speed this up, we can use a bitset representation or efficient array operations.
    # n_cand = 5280.
    # A boolean adjacency matrix of shape (5280, 5280) is 28M booleans ~ 28 MB. This is fine.
    
    # Build conflict matrix
    # Conflicts if G > 2
    conflict_mat = G > 2
    
    # We need to find the max independent set.
    # Greedy approach:
    # Maintain a set of available nodes.
    # At each step, find the node with the fewest conflicts among available nodes.
    # This is O(N^3) if done naively (N steps, each finding min degree O(N), updating O(N)).
    # N=5280. 5280^3 is 1.5e11. Too slow.
    
    # Optimization:
    # We can use a heap or just iterate.
    # Actually, the structure of weight-4 vectors is highly symmetric.
    # Maybe we can restrict the search space further?
    # 
    # Alternative: 
    # Use the known kissing number bounds or constructions.
    # The kissing number in 11D is known to be at least 556 (from Levenshtein-Rodman bounds or explicit constructions).
    # Wait, the task asks for integer vectors. The "kissing number" usually refers to unit spheres.
    # Here, we normalize to norm 2. So we are looking for the max number of vectors in Z^11 with norm 2 (wait, norm sqrt(4)=2) 
    # such that min distance >= 2.
    # Distance squared >= 4.
    # This is exactly the packing of norm-2 integer vectors.
    
    # Let's try a faster greedy:
    # 1. Sort candidates by some heuristic (e.g., lexicographic).
    # 2. Iterate through them. If compatible with all selected, add.
    # This is O(K * N * D) where K is final size.
    # If K ~ 500, N ~ 5000, D ~ 11. 500 * 5000 * 11 = 27.5M ops. Fast.
    
    # But simple greedy order matters.
    # Let's try multiple random orderings (restarts) and keep the best.
    
    best_set = []
    best_count = 0
    rng = np.random.default_rng(42)
    
    # Precompute candidate norms (all 4, but keep for general logic)
    # norms_sq = np.full(n_cand, 4)
    
    # We can vectorize the compatibility check against the current set.
    # Current set vectors: C (k, D)
    # Candidate vector: v (D,)
    # Dots: C @ v (k,)
    # Condition: dots <= 2 (since norms are 4, min norm is 4, so 2*dots <= 4 -> dots <= 2)
    
    # Let's run several greedy restarts.
    n_restarts = 200  # We have time.
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random permutation of candidates
        perm = rng.permutation(n_cand)
        
        curr_set_indices = []
        curr_vecs = []  # List of vectors in current set
        
        for idx in perm:
            v = candidates[idx]
            
            if not curr_vecs:
                curr_set_indices.append(idx)
                curr_vecs.append(v)
                continue
                
            # Check compatibility with all in curr_vecs
            # Convert curr_vecs to array for fast matmul? 
            # Doing this every iteration is O(K) array creation.
            # Better to maintain an array.
            
            # Let's optimize by maintaining an array of selected vectors.
            # But we are in a loop.
            # Let's just use the list for now, it's small.
            # Actually, let's pre-allocate or use numpy array and track length.
            
            # For speed, let's check against the last few? No, must check all.
            
            # Vectorized check:
            # If curr_vecs is a list, np.array(curr_vecs) is expensive.
            # Let's maintain `curr_set_arr` and `curr_set_len`.
            
            # Refactoring the inner loop for speed:
            # Instead of restarting the whole greedy from scratch with a new permutation,
            # we can just do one greedy pass with a specific ordering.
            # The ordering is the key.
            
            # Let's try a "min-conflict" greedy.
            # Start with empty set.
            # At each step, pick the candidate that has the fewest conflicts with the current set?
            # Or the one that is compatible with the most?
            # Since we need compatibility with ALL, we just need to find ANY compatible one.
            # The "Min-Degree" heuristic for Independent Set is:
            # Pick node with min degree in the *conflict graph of remaining nodes*.
            # This requires updating degrees.
            
            # Let's stick to simple random restarts first, it's robust.
            # To make it faster, we can break early if we know we can't beat best_count?
            # No, we don't know.
            
            # Optimization: Use numpy for the check.
            # Maintain `curr_vecs_np` as a 2D array? 
            # Appending to a 2D array is expensive.
            # Let's just use the list and convert to array for the dot product.
            # K will be ~500. Converting 500 vectors of dim 11 is fast.
            
            if len(curr_vecs) > 0:
                # Check if v is compatible with all curr_vecs
                # dots = np.array(curr_vecs) @ v
                # if np.all(dots <= 2):
                
                # To avoid array creation overhead, we can just loop if K is small?
                # No, numpy is better.
                
                # Let's try a different strategy: 
                # Use the precomputed G matrix.
                # We have G (n_cand, n_cand).
                # We want to select indices.
                # If we maintain a set of selected indices `S`, 
                # a new index `i` is compatible if for all `j` in `S`, G[i,j] <= 2.
                
                # We can check this using G[i, S] <= 2.
                
                # Let's implement the greedy using G.
                
                # This restart loop is getting complex. Let's restart the logic.
                
                pass 
            else:
                curr_set_indices.append(idx)
                curr_vecs.append(v)
                
        # End of permutation loop
        # This simple greedy is likely too slow or yields poor results if not optimized.
        
    # Let's try a more efficient greedy using the G matrix.
    
    # Greedy Independent Set using G:
    # available = set(range(n_cand))
    # selected = []
    # while available:
    #    # Find node in available with min degree in induced subgraph of available
    #    # This is hard to do efficiently without updating degrees.
    
    # Alternative: 
    # Since the graph is defined by G > 2, and G has structure,
    # let's try a simple greedy with a specific ordering that exploits symmetry.
    
    # Order candidates by their first non-zero coordinate, then lexicographically.
    # This breaks symmetry and might lead to a denser packing.
    
    # Let's generate the candidates in a specific order and do a single greedy pass.
    # Then try random shuffles.
    
    # To speed up the check, we will maintain the list of selected indices.
    # Check for new index i: 
    # np.all(G[i, selected_indices] <= 2)
    
    # This is O(K) per candidate. Total O(N*K).
    # N=5280, K~500. 2.6M operations. Very fast.
    
    best_selected = []
    best_k = 0
    
    # Try a few orderings
    # 1. Original order (lexicographic by indices, then signs)
    # 2. Random permutations
    
    orderings = []
    # Original
    orderings.append(np.arange(n_cand))
    
    # Random
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        orderings.append(rng.permutation(n_cand))
    
    for order in orderings:
        selected = []
        for i in order:
            # Check if i is compatible with all selected
            if not selected:
                selected.append(i)
            else:
                # G[i, selected] <= 2
                # selected is a list of ints.
                # G[i] is a vector.
                # We need to check G[i, j] for all j in selected.
                
                # Efficient way:
                # dots = G[i, selected]
                # if np.all(dots <= 2):
                
                # Using numpy indexing on G[i, selected] creates a copy.
                # This is fine for K~500.
                
                if np.all(G[i, selected] <= 2):
                    selected.append(i)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_selected = selected.copy()
            
    # If we still have time, try to improve best_selected via local search?
    # Or just return best_selected.
    
    # Convert best_selected indices to vectors
    if not best_selected:
        # Fallback
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    final_vecs = candidates[best_selected]
    return final_vecs

# EVOLVE-BLOCK-END
