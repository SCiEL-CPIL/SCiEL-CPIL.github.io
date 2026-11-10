import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    # Generate all integer vectors with a specific squared norm
    # We focus on norms 3, 4, and 5 which have high counts and potentially high packing density.
    
    def generate_vectors_norm(n_sq, d=11):
        vecs = []
        # Generate combinations of indices and signs
        # We need sum of squares = n_sq.
        # Possible partitions of n_sq into squares of integers (1, 4, 9...)
        # For small n_sq, mostly 1s and maybe one 4.
        
        # Simplified: Assume entries are 0, 1, -1, 2, -2.
        # For n_sq <= 5, only 1s and possibly one 4 (from 2^2).
        
        if n_sq == 3:
            # Three 1s
            for idxs in itertools.combinations(range(d), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, s in zip(idxs, signs):
                        v[idx] = s
                    vecs.append(v)
        elif n_sq == 4:
            # Four 1s
            for idxs in itertools.combinations(range(d), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, s in zip(idxs, signs):
                        v[idx] = s
                    vecs.append(v)
            # One 2, two 1s
            for idx2 in range(d):
                for idx1s in itertools.combinations([i for i in range(d) if i != idx2], 2):
                    for sign2 in [1, -1]:
                        for signs1 in itertools.product([1, -1], repeat=2):
                            v = np.zeros(d, dtype=np.int64)
                            v[idx2] = 2 * sign2
                            for idx, s in zip(idx1s, signs1):
                                v[idx] = s
                            vecs.append(v)
        elif n_sq == 5:
            # Five 1s
            for idxs in itertools.combinations(range(d), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, s in zip(idxs, signs):
                        v[idx] = s
                    vecs.append(v)
            # One 2, one 1
            for idx2 in range(d):
                for idx1 in [i for i in range(d) if i != idx2]:
                    for sign2 in [1, -1]:
                        for sign1 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[idx2] = 2 * sign2
                            v[idx1] = sign1
                            vecs.append(v)
        
        return np.array(vecs, dtype=np.int64) if vecs else np.array([]).reshape(0, d)

    # Strategy:
    # 1. Get all norm-3 vectors. They are all mutually valid.
    # 2. Try to find a large valid subset of norm-4 vectors.
    # 3. Try to find a large valid subset of norm-5 vectors.
    # 4. Return the largest set found.
    
    best_points = np.array([]).reshape(0, 11)
    
    # Step 1: Norm 3
    vecs3 = generate_vectors_norm(3)
    if len(vecs3) > len(best_points):
        best_points = vecs3.copy()
        
    # Helper: Greedy selection with time limit
    def greedy_select(points, max_norm_sq, time_budget):
        n = len(points)
        if n == 0:
            return np.array([]).reshape(0, 11)
        
        # Precompute norms (should be constant for pure norm sets)
        # But we might mix? No, we keep pure sets for simplicity of dot product threshold.
        # Condition: dist^2 >= max_norm_sq
        # dist^2 = norm_u^2 + norm_v^2 - 2 u.v
        # For pure norm N: 2N - 2 u.v >= N  =>  N >= 2 u.v  => u.v <= N/2
        
        threshold = max_norm_sq / 2.0
        
        # Compute Gram matrix? n x n.
        # For n=1320, 1320^2 = 1.7M entries. OK.
        # For n=5280, 5280^2 = 27M entries. OK (27M * 8 bytes = 216 MB). Might be tight but OK.
        # For n=14784, 14784^2 = 218M entries. 1.7 GB. TOO BIG.
        
        if n > 6000:
            # Use a different strategy for large sets: Random sampling + Greedy
            # Or just return empty to save time, relying on smaller sets?
            # Let's try a sparse greedy: pick a random subset of 2000, solve, then try to add more?
            # Actually, just return best_points if n is too large to handle quickly.
            # We'll handle norm 5 separately or skip if too big.
            pass
            
        M = points @ points.T
        
        # Greedy: Sort by "potential" or just random?
        # A good heuristic: Start with a random point, then pick the one with most available connections?
        # Simple greedy: Iterate through points, add if compatible with all selected.
        # To improve, we can do multiple random restarts.
        
        best_subset = []
        
        # Time check
        t_start = time.time()
        
        # Try a few random orderings
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > time_budget:
                break
            perm = np.random.permutation(n)
            current = []
            for i in perm:
                if not current:
                    current.append(i)
                else:
                    # Check compatibility with all in current
                    # M[i, current] <= threshold
                    if np.all(M[i, current] <= threshold):
                        current.append(i)
            
            if len(current) > len(best_subset):
                best_subset = current
                # Early break if we beat the current global best?
                # We don't know global best here, but we can check against best_points length
                if len(best_subset) > len(best_points):
                    # Keep searching for a bit more? Or break?
                    # Let's break if we've found something significantly better
                    if len(best_subset) > len(best_points) * 1.2:
                        break
        
        if best_subset:
            return points[best_subset]
        return np.array([]).reshape(0, 11)

    # Step 2: Norm 4
    if time.time() < deadline - 100:
        vecs4 = generate_vectors_norm(4)
        if len(vecs4) > 0:
            max_norm_sq = 4
            t_budget = min(deadline - 50, time.time() + 200)
            sel4 = greedy_select(vecs4, max_norm_sq, t_budget)
            if len(sel4) > len(best_points):
                best_points = sel4.copy()

    # Step 3: Norm 5
    if time.time() < deadline - 100:
        vecs5 = generate_vectors_norm(5)
        if len(vecs5) > 0:
            max_norm_sq = 5
            # vecs5 is large (14784).
            # We cannot compute full Gram matrix.
            # We need a sparse or approximate greedy.
            # Let's try to select a subset of size 5000 and run greedy on that?
            # Or just use the norm 3 result if it's already good (1320).
            # Let's try a quick greedy on a random sample of 5000.
            sample_size = min(5000, len(vecs5))
            if sample_size > 0:
                sample_idx = np.random.choice(len(vecs5), sample_size, replace=False)
                sample_pts = vecs5[sample_idx]
                max_norm_sq = 5
                threshold = 2.5
                M_sample = sample_pts @ sample_pts.T
                
                best_subset = []
                t_start = time.time()
                t_budget = min(deadline - 50, time.time() + 100)
                
                for r in range(5):
                    if time.time() > t_budget:
                        break
                    perm = np.random.permutation(sample_size)
                    current = []
                    for i in perm:
                        if not current:
                            current.append(i)
                        else:
                            if np.all(M_sample[i, current] <= threshold):
                                current.append(i)
                    
                    if len(current) > len(best_subset):
                        best_subset = current
                
                if best_subset:
                    sel5 = sample_pts[best_subset]
                    if len(sel5) > len(best_points):
                        best_points = sel5.copy()

    # Final Verification
    if len(best_points) < 2:
        # Fallback to norm 2 if everything failed (should not happen)
        return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
    
    k = len(best_points)
    norms = np.sum(best_points**2, axis=1)
    max_norm_sq = int(np.max(norms))
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = best_points[i] - best_points[j]
            d_sq = int(np.sum(diff**2))
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
    
    if min_dist_sq >= max_norm_sq:
        return np.around(best_points).astype(np.int64)
    else:
        # If invalid, return a known valid small set
        return np.array([np.ones(11), -np.ones(11)], dtype=np.int64)
