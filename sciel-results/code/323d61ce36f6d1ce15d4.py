import numpy as np
import time

D = 11

def construct():
       start_time = time.time()
       deadline = start_time + 140.0 # 10s margin

       # 1. Generate Axis vectors (Norm 4)
       # 2e_i and -2e_i
       axis_vecs = []
       for i in range(D):
           v1 = np.zeros(D, dtype=np.int64)
           v2 = np.zeros(D, dtype=np.int64)
           v1[i] = 2
           v2[i] = -2
           axis_vecs.append(v1)
           axis_vecs.append(v2)
       axis_vecs = np.array(axis_vecs) # Shape (22, 11)

       # 2. Generate Type B vectors (Norm 4, four +/-1)
       # Indices combinations
       from itertools import combinations, product
       b_vecs = []
       for comb in combinations(range(D), 4):
           for signs in product([1, -1], repeat=4):
               v = np.zeros(D, dtype=np.int64)
               for idx, val in zip(comb, signs):
                   v[idx] = val
               b_vecs.append(v)
       b_vecs = np.array(b_vecs) # Shape (5280, 11)

       # 3. Compute Dot Products for B vecs
       # D_B = b_vecs @ b_vecs.T
       D_B = b_vecs @ b_vecs.T
       
       # Conflicts: D_B > 2
       # Diagonal is 4, so D_B[i,i] > 2 is True.
       # We want to select independent set in graph G where edge (i,j) exists if D_B[i,j] > 2.
       # Note: We don't care about diagonal for selection, just distinct indices.
       
       N = len(b_vecs)
       best_set = []
       best_size = 0
       
       # Greedy Independent Set
       # To speed up, we can precompute the conflict matrix as bool
       Conflict = D_B > 2
       # Set diagonal to False to avoid self-conflict logic issues if any
       np.fill_diagonal(Conflict, False)
       
       rng = np.random.default_rng(42)
       
       for trial in range(100):
           if time.time() > deadline - 5:
               break
               
           perm = rng.permutation(N)
           is_selected = np.zeros(N, dtype=bool)
           selected_count = 0
           
           # We can optimize by tracking degrees or just simple pass
           # Simple pass:
           for idx in perm:
               if is_selected[idx]:
                   continue
               # Check if idx conflicts with any already selected
               # Conflict[idx, is_selected] returns bool array of conflicts with selected
               # If any is True, skip
               if np.any(Conflict[idx, is_selected]):
                   continue
               is_selected[idx] = True
               selected_count += 1
               
           if selected_count > best_size:
               best_size = selected_count
               best_set = np.where(is_selected)[0]
               
       # 4. Combine
       selected_b_vecs = b_vecs[best_set]
       result = np.vstack([axis_vecs, selected_b_vecs])
       
       # Verify (optional, but good for debug)
       # min_dist_sq = min ||u-v||^2
       # max_norm_sq = max ||u||^2
       # Should hold.
       
       return result
