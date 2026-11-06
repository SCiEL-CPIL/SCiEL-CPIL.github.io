import time
import random
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
       start_time = time.time()
       deadline = start_time + 270.0

       # Precompute coordinates
       x_coords = np.repeat(np.arange(N), N)
       y_coords = np.tile(np.arange(N), N)
       points = np.column_stack((x_coords, y_coords))
       
       best_set = []
       best_size = 0
       
       num_restarts = 5000 # We can do more restarts now

       for restart in range(num_restarts):
           if time.time() > deadline:
               break
           
           s_indices = []
           active = np.ones(TOTAL_POINTS, dtype=bool)
           
           # Initial pick
           # Pick a random point to start
           idx = np.random.choice(TOTAL_POINTS)
           s_indices.append(idx)
           active[idx] = False
           
           while True:
               if time.time() > deadline:
                   break
               
               if not np.any(active):
                   break
               
               k = len(s_indices)
               if k == 0:
                   break
                   
               # Get active points coordinates
               active_indices = np.where(active)[0]
               p_active = points[active_indices]
               p_s = points[s_indices]
               
               # Compute distances D_active_s: shape (num_active, k)
               # d = (x1-x2)^2 + (y1-y2)^2
               # Broadcasting: p_active[:, np.newaxis, :] - p_s[np.newaxis, :, :]
               diff = p_active[:, np.newaxis, :] - p_s[np.newaxis, :, :]
               d_active_s = np.sum(diff * diff, axis=2)
               
               # Check Condition 1: All distances from P to S must be unique
               # For each row, check if unique count == k
               # np.unique is slow for many rows.
               # Alternative: Sort and check for adjacent equals.
               sorted_d = np.sort(d_active_s, axis=1)
               # Adjacent equals in sorted array
               # If k=1, no adjacent.
               if k > 1:
                   adjacent_eq = (sorted_d[:, 1:] == sorted_d[:, :-1]).any(axis=1)
                   cond1_valid = ~adjacent_eq
               else:
                   cond1_valid = np.ones(len(active_indices), dtype=bool)
                   
               # Check Condition 2: No distance d(P, Si) equals d(Si, Sj) for any i != j
               # Precompute pairwise distances in S
               # d_s_s: shape (k, k)
               diff_s = p_s[:, np.newaxis, :] - p_s[np.newaxis, :, :]
               d_s_s = np.sum(diff_s * diff_s, axis=2)
               
               # Get unique non-zero distances in S
               # Mask out diagonal (0)
               np.fill_diagonal(d_s_s, -1) # Temporarily mark diagonal
               valid_s_dists = d_s_s[d_s_s > 0]
               unique_s_dists = np.unique(valid_s_dists)
               
               # Check if any element in d_active_s is in unique_s_dists
               # This is a set membership check for each row.
               # np.isin is efficient.
               # d_active_s shape (num_active, k)
               # unique_s_dists shape (m,)
               # We want to know if ANY element in the row is in unique_s_dists.
               # isin returns boolean array of same shape as d_active_s.
               is_in_s_dists = np.isin(d_active_s, unique_s_dists)
               # If any True in a row, then condition 2 is violated.
               cond2_violated = is_in_s_dists.any(axis=1)
               cond2_valid = ~cond2_violated
               
               valid_mask = cond1_valid & cond2_valid
               
               if not np.any(valid_mask):
                   break
                   
               # Pick a random valid candidate
               valid_global_indices = active_indices[valid_mask]
               chosen_idx = np.random.choice(valid_global_indices)
               
               s_indices.append(chosen_idx)
               active[chosen_idx] = False
               
           if len(s_indices) > best_size:
               best_size = len(s_indices)
               best_set = s_indices[:]
               
       # Convert indices back to (x, y) tuples
       result = [(int(points[i][0]), int(points[i][1])) for i in best_set]
       return result
