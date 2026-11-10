import numpy as np
from scipy.optimize import differential_evolution
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    # We pack 12 unit regular hexagons in a regular hexagon.
    # The inner hexagons have side length 1.
    # The outer hexagon has side length s.
    # We want to minimize s.
    
    # Strategy:
    # The centers of the inner hexagons must be at least sqrt(3) apart
    # (distance between centers of adjacent unit hexagons in a perfect packing is sqrt(3)).
    # Actually, the distance between centers of two non-overlapping unit hexagons
    # depends on their relative orientation.
    # If they have the same orientation, the minimum center distance is sqrt(3).
    # If they are rotated, the distance can be smaller? No, the inradius is sqrt(3)/2.
    # The distance between centers must be at least 2 * inradius = sqrt(3) IF they are aligned.
    # More generally, the distance between centers must be at least the sum of their "radii" in the direction connecting them.
    # For regular hexagons, the distance from center to vertex is 1, to edge midpoint is sqrt(3)/2.
    # The minimum distance between centers of two non-overlapping unit hexagons is sqrt(3) if they share an edge.
    # If they are rotated, the minimum distance is still sqrt(3) (when an edge of one touches a vertex of the other? No).
    # Let's verify:
    # Hex 1: center (0,0), angle 0. Vertices at (1,0), (0.5, sqrt(3)/2), ...
    # Hex 2: center (d, 0), angle 0.
    # They overlap if d < sqrt(3).
    # If Hex 2 is rotated by 30 deg, its vertices are at angles 30, 90, ...
    # Vertex at (cos(30), sin(30)) = (sqrt(3)/2, 0.5).
    # Distance from (0,0) to this vertex is 1.
    # But the hexagon extends further.
    # The minimum distance between centers of two non-overlapping unit hexagons is sqrt(3).
    # This is a known result: the kissing number of hexagons is 6, and they form a triangular lattice with spacing sqrt(3).
    
    # So, we can model the centers as points on a triangular lattice with spacing sqrt(3), scaled by some factor?
    # No, the hexagons can be rotated individually.
    # However, for a compact packing, it is optimal to align them.
    # Let's assume all inner hexagons have the same orientation.
    # Then the centers must form a packing of 12 points with minimum distance sqrt(3).
    # We want to fit these 12 points (plus their hexagon extent) into the outer hexagon.
    # The outer hexagon must contain all vertices of the inner hexagons.
    # If all inner hexagons have angle 0, their vertices are at center + (cos(60k), sin(60k)).
    # The outer hexagon has angle 0 (wlog, due to symmetry).
    # The condition is that all inner vertices are inside the outer hexagon.
    
    # Let's use a continuous optimization approach.
    # Variables:
    # - 12 centers (x_i, y_i)
    # - 12 angles (theta_i)
    # - Outer hexagon center (X, Y) and angle (Phi) and side length s.
    # But we can fix the outer hexagon center at (0,0) and angle at 0 by symmetry?
    # Not necessarily. The optimal packing might not have the outer hexagon aligned with the inner ones.
    # However, for a regular hexagon container, aligning it with the cluster often helps.
    # Let's fix outer center at (0,0) and outer angle at 0.
    # Then we optimize the 12 centers and 12 angles, and minimize s.
    # s is determined by the max projection of the inner vertices onto the outer normals.
    # Outer normals are at 30, 90, 150, 210, 270, 330 degrees.
    # s = max( proj ) / (sqrt(3)/2) = 2 * max( proj ) / sqrt(3).
    
    # Constraints:
    # 1. Inner hexagons do not overlap.
    #    This is a non-convex constraint.
    #    We can use a penalty method or differential evolution with constraints.
    #    Or, we can restrict to a lattice-based configuration.
    
    # Let's try a lattice-based approach first, as it is more robust.
    # The centers form a subset of a triangular lattice with spacing d.
    # We want to choose 12 points from a triangular lattice such that the enclosing hexagon is minimized.
    # The lattice spacing d must be at least sqrt(3).
    # Let d = sqrt(3) * scale.
    # We search over scale and the shape of the 12-point cluster.
    
    # Common 12-point clusters in a triangular lattice:
    # 1. 3x4 rectangle (24 points total, pick 12? No, 3x4 is 12 points exactly if it's a parallelogram? No, 3x4 grid has 12 points.)
    #    A 3x4 parallelogram of lattice points.
    # 2. Hexagon of side 2 has 19 points. Remove 7 to get 12.
    # 3. Hexagon of side 1 has 7 points. Add 5 more.
    
    # Let's enumerate a few candidate shapes on the lattice.
    # Lattice vectors: v1 = (d, 0), v2 = (d/2, d*sqrt(3)/2).
    
    # Candidate 1: 3x4 rectangle
    # Points: (i*v1 + j*v2) for i in 0..2, j in 0..3.
    # This is a parallelogram.
    
    # Candidate 2: Hexagon of side 2 minus corners.
    # Side 2 hexagon has points with axial coords (q, r) such that |q|<=2, |r|<=2, |q+r|<=2.
    # Total 19 points.
    # We need to remove 7 points to minimize the bounding box.
    
    # Let's use differential evolution on the centers directly.
    # It's more general.
    
    def calc_s_and_overlap(params):
        # params: 12*2 centers + 12*1 angles = 36 params
        # Fixed outer: center (0,0), angle 0.
        
        if time.time() - start_time > deadline:
            return 1e10  # Timeout penalty
        
        centers = params[:24].reshape(12, 2)
        angles = params[24:36]
        
        # Compute vertices of each inner hexagon
        # Hexagon i has center c_i, angle a_i, side 1.
        # Vertices: c_i + (cos(a_i + 60k), sin(a_i + 60k))
        
        all_vertices = []
        for i in range(12):
            cx, cy = centers[i]
            a = angles[i]
            vs = np.array([[cx + np.cos(a + np.deg2rad(60*k)), cy + np.sin(a + np.deg2rad(60*k))] for k in range(6)])
            all_vertices.append(vs)
        
        all_vertices = np.vstack(all_vertices)
        
        # Calculate s
        # Outer normals
        normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        projs = all_vertices @ normals.T
        max_proj = np.max(projs)
        s_val = 2.0 * max_proj / np.sqrt(3)
        
        # Check overlaps
        # For each pair of hexagons, check if they overlap.
        # This is expensive. 12*11/2 = 66 pairs.
        # We can use a simple distance check as a proxy?
        # No, we need to be sure.
        # Let's use a fast rejection: if center distance > 2, they don't overlap (max distance between two points in a hex is 2).
        # If center distance < sqrt(3), they might overlap.
        
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0:
                    continue
                # Check for overlap using SAT or simple vertex-in-polygon?
                # Let's use a conservative penalty: if dist < sqrt(3) - 1e-5, penalty.
                # This is not exact but helps guide the optimizer.
                if dist < np.sqrt(3) - 1e-4:
                    overlap_penalty += (np.sqrt(3) - dist) ** 2
        
        return s_val + 100.0 * overlap_penalty
    
    # Bounds for DE
    # Centers: roughly in [-5, 5]
    # Angles: [0, 2*pi)
    bounds = [(-5, 5)] * 24 + [(0, 2*np.pi)] * 12
    
    # Initial guess: lattice
    d = np.sqrt(3)
    v1 = np.array([d, 0])
    v2 = np.array([d/2, d*np.sqrt(3)/2])
    
    # 3x4 grid
    init_centers = []
    for i in range(3):
        for j in range(4):
            init_centers.append(i*v1 + j*v2)
    init_centers = np.array(init_centers)
    init_angles = np.zeros(12)
    
    # Center the cluster
    centroid = np.mean(init_centers, axis=0)
    init_centers -= centroid
    
    init_params = np.concatenate([init_centers.flatten(), init_angles])
    
    # Run DE
    # Use a small population and few generations to stay within time limit.
    try:
        res = differential_evolution(
            calc_s_and_overlap, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            init='sobol',
            polish=True,
            tol=1e-6,
            x0=init_params
        )
        
        if res.success or res.fun < 1e9:
            best_params = res.x
            best_s = res.fun
        else:
            best_params = init_params
            best_s = calc_s_and_overlap(init_params)
            
    except Exception as e:
        print("DE failed:", e)
        best_params = init_params
        best_s = calc_s_and_overlap(init_params)
    
    # Refine with local optimization if time permits
    if time.time() - start_time < deadline - 10:
        from scipy.optimize import minimize
        try:
            res2 = minimize(
                calc_s_and_overlap, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res2.fun < best_s:
                best_params = res2.x
                best_s = res2.fun
        except Exception as e:
            print("Minimize failed:", e)
    
    # Extract best result
    centers = best_params[:24].reshape(12, 2)
    angles = best_params[24:36]
    
    # Recalculate s precisely
    all_vertices = []
    for i in range(12):
        cx, cy = centers[i]
        a = angles[i]
        vs = np.array([[cx + np.cos(a + np.deg2rad(60*k)), cy + np.sin(a + np.deg2rad(60*k))] for k in range(6)])
        all_vertices.append(vs)
    all_vertices = np.vstack(all_vertices)
    
    normal_angles = np.deg2rad(30 + np.arange(0, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projs = all_vertices @ normals.T
    max_proj = np.max(projs)
    s_val = 2.0 * max_proj / np.sqrt(3)
    
    # Construct return values
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers[i, 0]
        inner[i, 1] = centers[i, 1]
        inner[i, 2] = np.rad2deg(angles[i])
    
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, s_val
