   import numpy as np
   import time

   def construct():
       start_time = time.time()
       deadline = start_time + 110.0 # 10s margin for 120s limit

       # 1. Define inner hexagons
       epsilon = 0.001
       d1 = np.sqrt(3) * (1 + epsilon)
       d2 = 3.0 * (1 + epsilon)
       
       inner_centers = [(0.0, 0.0)]
       # Ring 1
       for i in range(6):
           ang = np.deg2rad(i * 60.0)
           x = d1 * np.cos(ang)
           y = d1 * np.sin(ang)
           inner_centers.append((x, y))
           
       # Ring 2 (4 hexagons)
       gap_angles_deg = [30.0, 90.0, 150.0, 210.0]
       for deg in gap_angles_deg:
           ang = np.deg2rad(deg)
           x = d2 * np.cos(ang)
           y = d2 * np.sin(ang)
           inner_centers.append((x, y))
           
       inner_angle_deg = 30.0
       inner = [(x, y, inner_angle_deg) for x, y in inner_centers]
       
       # 2. Precompute vertices of inner hexagons
       # Unit hexagon vertices (side 1, rotation 0)
       # Vertices at angles 0, 60, ...
       unit_verts = np.array([(np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))) for a in [0, 60, 120, 180, 240, 300]])
       
       # Rotate by 30 deg
       rot_matrix = np.array([[np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30))], 
                              [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))]])
       base_verts = unit_verts @ rot_matrix.T
       
       all_vertices = []
       for cx, cy, _ in inner:
           verts = base_verts + np.array([cx, cy])
           all_vertices.append(verts)
           
       all_vertices = np.vstack(all_vertices)
       
       # 3. Optimize outer angle
       # Outer hexagon normals for angle alpha:
       # If outer hexagon is rotated by alpha, its normals are at alpha, alpha+60, ...
       # The support constraint is v . n <= S * sqrt(3) / 2
       # So S >= (2/sqrt(3)) * max(v . n)
       
       def calc_S(alpha_deg):
           alpha_rad = np.deg2rad(alpha_deg)
           # Normals of outer hexagon
           normal_angles = [alpha_rad + np.deg2rad(i*60) for i in range(6)]
           normals = np.array([[np.cos(a), np.sin(a)] for a in normal_angles])
           
           # Project all vertices onto all normals
           # proj shape: (N_verts, 6)
           proj = all_vertices @ normals.T
           # Max projection for each vertex
           max_proj = np.max(proj, axis=1)
           # Global max projection
           global_max = np.max(max_proj)
           return (2.0 / np.sqrt(3.0)) * global_max
           
       # Search for best alpha
       best_S = float('inf')
       best_alpha = 0.0
       
       # Grid search first
       for alpha in np.linspace(0, 60, 600):
           S = calc_S(alpha)
           if S < best_S:
               best_S = S
               best_alpha = alpha
               
       # Refine with finer grid around best_alpha
       if time.time() < deadline:
           for alpha in np.linspace(best_alpha - 0.1, best_alpha + 0.1, 100):
               S = calc_S(alpha)
               if S < best_S:
                   best_S = S
                   best_alpha = alpha
                   
       # Final result
       center = (0.0, 0.0)
       side = best_S
       angle = best_alpha
       
       return inner, center, side, angle
   