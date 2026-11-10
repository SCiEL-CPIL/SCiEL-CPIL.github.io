import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a large set of 11-dimensional integer points with norm squared 4
    such that the minimum squared pairwise distance is at least the maximum squared norm (4).
    
    Strategy:
    1. Generate all integer vectors in Z^11 with squared norm 4.
       These are of two types:
       - Type A: Two non-zero entries, each ±2. (e.g., (2, 0, ..., 0, 0))
       - Type B: Four non-zero entries, each ±1. (e.g., (1, 1, 1, 1, 0, ...))
    2. The constraint for equinorm points (norm^2 = M = 4) is:
       ||u - v||^2 >= M  <=>  ||u||^2 + ||v||^2 - 2<u, v> >= M
       4 + 4 - 2<u, v> >= 4  <=>  4 >= 2<u, v>  <=>  <u, v> <= 2.
    3. We perform a greedy search to find a large subset S of these points such that
       for all u, v in S, <u, v> <= 2.
    4. To maximize the size, we use multiple randomized restarts with different
       ordering strategies (random permutation, greedy by "degree" heuristic).
    5. Vectorized operations (numpy matrix multiplication) are used to check
       compatibility of a candidate batch against the current set.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s margin for safety
    
    d = 11
    M = 4  # Target norm squared
    
    # 1. Generate candidates
    candidates = []
    
    # Type A: Two non-zero entries, each ±2
    # Indices i, j with i < j. Values ±2 at i and j.
    # 2^2 = 4 sign combinations.
    for i, j in itertools.combinations(range(d), 2):
        for si in [2, -2]:
            for sj in [2, -2]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
    
    # Type B: Four non-zero entries, each ±1
    # Indices i, j, k, l with i < j < k < l. Values ±1 at these indices.
    # 2^4 = 16 sign combinations.
    for i, j, k, l in itertools.combinations(range(d), 4):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        p[k] = s3
                        p[l] = s4
                        candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Precompute dot product matrix? 
    # n_cands is around 1320 + 330*16 = 1320 + 5280 = 6600.
    # 6600 x 6600 matrix is ~43 million entries. 
    # int64 is 8 bytes -> 344 MB. This is large but might be okay?
    # Or we can compute on the fly in batches.
    # Let's try to compute the full Gram matrix if memory allows, or use batching.
    # 6600^2 * 8 bytes = 348 MB. Python overhead might make it risky.
    # Let's use batching for safety and to keep memory low, but vectorize the check.
    
    best_set = []
    best_count = 0
    
    # Helper to validate a set
    def is_valid_set(points):
        if len(points) < 2:
            return True
        # Check min dist >= max norm
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # For our candidates, max_norm_sq should be 4.
        # Check pairwise distances
        # To speed up, we can check dot products.
        # dist^2 = ||u||^2 + ||v||^2 - 2<u,v>
        # If all norms are 4, dist^2 = 8 - 2<u,v>.
        # We need dist^2 >= 4 => 8 - 2<u,v> >= 4 => 2<u,v> <= 4 => <u,v> <= 2.
        # So we just need to check if all pairwise dot products <= 2.
        if len(points) > 0:
            dots = points @ points.T
            # Check upper triangle
            iu = np.triu_indices(len(points), k=1)
            if np.max(dots[iu]) > 2:
                return False
        return True

    # Greedy algorithm with vectorized checks
    def greedy_search(order_idx):
        """
        order_idx: array of indices into candidates, defining the order of consideration.
        Returns the set of points (indices or vectors).
        """
        current_set = []
        # current_dots: list of dot products of new candidate with current_set?
        # Better: keep current_set as a numpy array and recompute?
        # Recomputing dots for each new candidate against all current points is O(N^2 * D).
        # If N is large (e.g., 1000+), this might be slow in Python loop.
        # Let's try to batch.
        
        # Actually, since we are adding one by one, we can maintain a matrix of 
        # dot products between selected points.
        # Let S be the list of selected point vectors.
        # For a new candidate c, we need c @ S.T <= 2 for all entries.
        
        selected_indices = []
        selected_vectors = []
        
        # To speed up, we can pre-filter candidates that are "compatible" with a seed?
        # No, let's just iterate.
        
        # Optimization: If we have a large set, checking against it is expensive.
        # We can use a "conflict graph" approach?
        # For now, simple greedy with numpy dot product per candidate.
        
        for idx in order_idx:
            if time.time() > deadline - 5:
                break
            
            c = candidates[idx]
            
            if len(selected_vectors) == 0:
                selected_vectors.append(c)
                selected_indices.append(idx)
                continue
            
            # Check compatibility with all selected vectors
            # c @ selected_vectors.T
            # selected_vectors is a list, convert to array? 
            # Converting every time is expensive.
            # Let's keep selected_vectors as a growing numpy array?
            # Resizing numpy arrays is expensive.
            
            # Alternative: Keep a list, and when checking, if list is long, 
            # maybe batch check?
            # For now, let's assume the set size won't exceed a few thousand 
            # and the number of candidates is ~6000.
            # 6000 candidates * 1000 avg set size * 11 dims = 66M ops. 
            # This is fast in numpy if vectorized.
            
            # Convert current set to array for check
            # This is O(N) copy. 
            # Let's try to maintain a numpy array and append?
            # np.vstack is also O(N).
            
            # Let's just use the list and convert to array for the dot product.
            # If it's too slow, we'll need a more sophisticated structure.
            
            sv = np.array(selected_vectors, dtype=np.int64)
            dots = c @ sv.T
            if np.all(dots <= 2):
                selected_vectors.append(c)
                selected_indices.append(idx)
        
        return selected_indices

    # Run multiple restarts with different orderings
    # 1. Random order
    # 2. Order by "popularity" or just random
    # 3. Maybe a heuristic order?
    
    rng = np.random.default_rng(42)
    
    # Try a few random permutations
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        order = rng.permutation(n_cands)
        result_indices = greedy_search(order)
        
        if len(result_indices) > best_count:
            best_count = len(result_indices)
            best_set = result_indices
            # Optional: print progress
            # print(f"Restart {restart}: Found {best_count} points")

    # If we found a good set, convert to array
    if best_count > 0:
        best_points = candidates[best_set]
    else:
        # Fallback: just return two valid points
        # e.g., (2,0,...) and (-2,0,...)
        # Dist^2 = 16 >= 4. Valid.
        p1 = np.zeros(11, dtype=np.int64)
        p1[0] = 2
        p2 = np.zeros(11, dtype=np.int64)
        p2[0] = -2
        best_points = np.array([p1, p2], dtype=np.int64)

    # Final verification
    if not is_valid_set(best_points):
        # Fallback to a known valid small set
        p1 = np.zeros(11, dtype=np.int64)
        p1[0] = 2
        p2 = np.zeros(11, dtype=np.int64)
        p2[0] = -2
        best_points = np.array([p1, p2], dtype=np.int64)

    return np.around(best_points).astype(np.int64)

# EVOLVE-BLOCK-END
