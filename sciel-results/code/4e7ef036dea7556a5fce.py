   import numpy as np
   import time

   N = 64
   P = N * N
   MAX_DIST = 2 * (N - 1) ** 2
   MAX_CHUNKS = (MAX_DIST // 64) + 1

   # Pre-compute distance matrix
   def _precompute_distances():
       xs = np.arange(N)
       ys = np.arange(N)
       X, Y = np.meshgrid(xs, ys)
       X_flat = X.flatten()
       Y_flat = Y.flatten()
       # (P, P) matrix
       D = (X_flat[:, None] - X_flat[None, :])**2 + (Y_flat[:, None] - Y_flat[None, :])**2
       return D.astype(np.int32)

   D = _precompute_distances()
   
   # Pre-compute chunk and bit indices for all possible distances
   # d is in [0, MAX_DIST]
   dist_chunk = np.array([d // 64 for d in range(MAX_DIST + 1)], dtype=np.int32)
   dist_bit = np.array([1 << (d % 64) for d in range(MAX_DIST + 1)], dtype=np.uint64)

   def construct():
       best_set = []
       best_size = 0
       
       start_time = time.time()
       time_limit = 290
       
       # Pre-allocate masks array
       # Max set size likely < 200? Let's assume 500 to be safe.
       MAX_SET_SIZE = 500
       masks = np.zeros((MAX_SET_SIZE, MAX_CHUNKS), dtype=np.uint64)
       
       while time.time() - start_time < time_limit:
           perm = np.random.permutation(P)
           S_indices = []
           k = 0
           
           for idx in perm:
               if k == 0:
                   S_indices.append(idx)
                   # masks[0] is already zero, no distances to add
                   k = 1
                   continue
               
               # Get distances from idx to all points in S
               dists = D[idx, S_indices]
               
               # Check 1: Distances from idx must be unique
               # np.unique is fast
               if len(np.unique(dists)) != k:
                   continue
               
               # Check 2: No existing point s_i has distance dists[i] to another point in S
               # c: chunk index for each distance
               # b: bit mask for each distance
               c = dist_chunk[dists]
               b = dist_bit[dists]
               
               # Get the relevant 64-bit chunks from the current masks
               # masks[0:k] is the active part
               # We need masks[i, c[i]] for i in 0..k-1
               relevant_masks = masks[:k, c] # Fancy indexing: (k,)
               
               # Check if any bit is set
               if np.any(relevant_masks & b):
                   continue
               
               # Valid! Add idx to S
               S_indices.append(idx)
               
               # Update masks
               # 1. Set bits for the new point (row k)
               masks[k, c] |= b
               
               # 2. Update existing points (rows 0..k-1) with the new distance
               masks[:k, c] |= b
               
               k += 1
               
               if k > best_size:
                   best_size = k
                   best_set = [tuple((i % N, i // N)) for i in S_indices]
                   
           # Early break if high score
           if best_size > 100:
               break
               
       return best_set
   