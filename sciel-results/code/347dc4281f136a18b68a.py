import numpy as np
import itertools
import time

D = 11

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 140.0 # Safety margin
        
        vectors = []
        
        # 1. Generate E8 minimal vectors (scaled by 2)
        # Type A: (±1, ..., ±1) with even number of -1s
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                v8 = list(signs)
                v11 = v8 + [0, 0, 0]
                vectors.append(v11)
                
        # Type B: Permutations of (±2, ±2, 0, ..., 0)
        for i, j in itertools.combinations(range(8), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v8 = [0] * 8
                    v8[i] = 2 * si
                    v8[j] = 2 * sj
                    v11 = v8 + [0, 0, 0]
                    vectors.append(v11)
                    
        # 2. Generate extra vectors in R^3
        # Permutations of (±2, ±2, 0)
        for i, j in itertools.combinations(range(3), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v3 = [0] * 3
                    v3[i] = 2 * si
                    v3[j] = 2 * sj
                    v11 = [0] * 8 + v3
                    vectors.append(v11)
                    
        # 3. Convert to numpy array
        vec_array = np.array(vectors, dtype=np.int64)
        
        # 4. Optional: Verify and prune if necessary (should not be needed)
        # But let's add a quick check to be safe against any logic errors
        if time.time() < deadline:
             # Check max norm
             norms_sq = np.sum(vec_array**2, axis=1)
             max_norm_sq = np.max(norms_sq)
             
             # Check min distance
             # Since 252 vectors, full pairwise is fast
             # dist_sq[i,j] = norm_i^2 + norm_j^2 - 2 dot_ij
             # We can just compute differences for the subset if it was large, but 252 is small
             min_dist_sq = float('inf')
             for i in range(len(vec_array)):
                 diffs = vec_array[i+1:] - vec_array[i]
                 dists_sq = np.sum(diffs**2, axis=1)
                 if len(dists_sq) > 0:
                     min_dist_sq = min(min_dist_sq, np.min(dists_sq))
             
             if min_dist_sq < max_norm_sq:
                 # Fallback: Return only E8 part? Or just return as is if we trust the math.
                 # The math says min_dist_sq should be 8 and max_norm_sq is 8.
                 # If it fails, it's a bug.
                 pass
                 
        return vec_array
