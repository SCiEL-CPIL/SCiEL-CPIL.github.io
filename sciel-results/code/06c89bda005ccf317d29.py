import numpy as np

def construct():
       N = 50
       # Fibonacci sphere initialization
       R_init = 0.25
       centers = np.zeros((N, 3))
       golden_angle = np.pi * (3.0 - np.sqrt(5.0))
       for i in range(N):
           cos_theta = 1.0 - 2.0 * (i + 0.5) / N
           sin_theta = np.sqrt(1.0 - cos_theta**2)
           phi = i * golden_angle
           centers[i, 0] = R_init * sin_theta * np.cos(phi)
           centers[i, 1] = R_init * sin_theta * np.sin(phi)
           centers[i, 2] = R_init * cos_theta

       # Optimization
       max_iter = 10000
       step_size = 0.05
       delta_active = 0.05
       best_r = 0.0
       best_centers = centers.copy()

       for i in range(max_iter):
           # Compute distances
           diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
           dist = np.linalg.norm(diff, axis=2)
           
           abs_centers = np.abs(centers)
           max_coord = np.max(abs_centers, axis=1)
           wall_dists = 0.5 - max_coord
           
           min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
           min_wall_dist = np.min(wall_dists)
           current_r = min(min_pair_dist / 2.0, min_wall_dist)
           
           if current_r > best_r:
               best_r = current_r
               best_centers = centers.copy()
               
           # Identify active constraints
           # We use a slightly larger threshold to include "near" active constraints
           active_pairs = np.argwhere(dist < 2 * (current_r + delta_active))
           active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
           
           active_walls = np.where(wall_dists < current_r + delta_active)[0]
           
           grad = np.zeros_like(centers)
           
           # Pair contributions
           for i_idx, j_idx in active_pairs:
               v = centers[i_idx] - centers[j_idx]
               norm_v = np.linalg.norm(v)
               if norm_v > 1e-9:
                   v_norm = v / norm_v
                   grad[i_idx] += v_norm
                   grad[j_idx] -= v_norm
                   
           # Wall contributions
           for i_idx in active_walls:
               limiting_coords = np.argmax(abs_centers[i_idx])
               # Gradient of wall dist w.r.t c_i[limiting_coords] is -sign(c_i[limiting_coords])
               # We want to increase wall dist, so move in direction of -grad_wall
               # grad_wall = -sign(c) * e_k
               # So move in +sign(c) * e_k?
               # Wait.
               # r_wall = 0.5 - |c_k|.
               # d(r_wall)/dc_k = -sign(c_k).
               # We want to maximize r. So we move in direction of gradient of r.
               # grad_r = -sign(c_k) * e_k.
               # So we add this to grad.
               direction = -np.sign(centers[i_idx, limiting_coords])
               grad[i_idx, limiting_coords] += direction
               
           grad_norm = np.linalg.norm(grad)
           if grad_norm > 1e-9:
               grad /= grad_norm
               centers += step_size * grad
               
           # Clamp
           centers = np.clip(centers, -0.5, 0.5)
           
           # Decay
           step_size *= 0.9995
           delta_active *= 0.9995
           
       return best_centers
