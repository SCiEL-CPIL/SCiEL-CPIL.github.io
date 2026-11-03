"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Vertices of a unit regular hexagon (side 1), centered at origin, flat-top orientation
    # For a regular hexagon with side length 1, vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # but for "flat-top" the first vertex is at 30 degrees offset... let me use pointy-top
    # Actually, let's define: vertex k at angle (60*k) degrees from x-axis, radius 1
    # This gives a "pointy-top" hexagon with vertex-to-vertex distance 2
    hex_vertices = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    
    # Initial guess: arrange 12 hexagons in a roughly hexagonal pattern
    # Row arrangement: rows of 2,3,4,3 = 12
    # Or: 3-4-3-2 = 12
    # Let's try a compact arrangement
    # Center at origin, place hexagons in a pattern
    
    # Good initial arrangement: 4-3-4-1 or similar
    # Let me use a known good packing: arrange in rows
    # For hexagonal packing, offset rows
    # Row 0: 3 hexagons at y=0, x = -2, 0, 2 (spacing 2 for flat-top adjacent)
    # Actually for pointy-top hexagons, horizontal spacing when side-by-side is sqrt(3)
    # Vertical spacing is 1.5
    
    # Let me think more carefully. Unit hexagon (side 1):
    # Pointy-top: width (left-right) = sqrt(3), height (top-bottom) = 2
    # Flat-top: width = 2, height = sqrt(3)
    
    # For minimal container, orientation matters. Let's use flat-top for inner hexagons initially.
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300, radius 1
    # Width = 2 (from x=-1 to x=1), Height = sqrt(3) (from y=-sqrt(3)/2 to y=sqrt(3)/2)
    
    # Adjacent flat-top hexagons: horizontal spacing = 1.5 (center to center for side-sharing)
    # Actually no. For flat-top, the distance between centers of side-adjacent hexagons:
    # The "width" in the direction of a side normal is 2 (flat to flat).
    # Center-to-center for sharing a side = 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Let me just use a reasonable initial layout and let optimization fix it.
    # Arrange in 3 rows: 4, 4, 4
    # Or 3, 4, 3, 2 = 12
    
    # Simple initial: grid-like arrangement
    # Use flat-top hexagons, side-adjacent spacing
    # Horizontal (along x): for flat-top, adjacent in x direction: spacing = 1.5 * 2 = 3? No.
    
    # Let me just place them in a rough hexagonal cluster
    positions = []
    angles = []
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    # 6 around center
    for k in range(6):
        ang = np.pi/3 * k
        r = 2.0  # distance between centers for side-adjacent flat-top hexagons
        positions.append([r * np.cos(ang), r * np.sin(ang)])
        angles.append(0.0)
    # 5 more around
    for k in range(5):
        ang = np.pi/3 * (k + 0.5)
        r = 3.5
        positions.append([r * np.cos(ang), r * np.sin(ang)])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Total parameters: 12*2 (centers) + 12 (angles) + 2 (outer center) + 1 (outer angle) = 39
    # Objective: minimize max support function over all vertices of all inner hexagons
    
    def get_inner_vertices(pos, ang, idx):
        """Get 6 vertices of inner hexagon idx."""
        c = pos[idx]
        a = np.radians(ang[idx])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        verts = hex_vertices @ R.T + c  # rotate vertices, then translate
        return verts
    
    def get_all_vertices(pos, ang):
        """Get all 72 vertices (12 hexagons * 6)."""
        verts = []
        for i in range(N):
            verts.append(get_inner_vertices(pos, ang, i))
        return np.vstack(verts)
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return np.inf
        
        # Unpack parameters
        n_params = len(params)
        pos = params[:N*2].reshape(N, 2)
        ang = params[N*2:N*2+N]  # in radians
        cx, cy = params[N*2+N:N*2+N+2]
        phi = params[N*2+N+2]  # outer hexagon rotation
        
        # Get all inner vertices
        all_verts = get_all_vertices(pos, ang)
        # Translate relative to outer center
        rel_verts = all_verts - np.array([cx, cy])
        
        # Outer hexagon normals (6 directions), rotated by phi
        # For a regular hexagon, the side normals are at angles 0, 60, 120, 180, 240, 300
        # The support function: max over all points of (n_i . p) where n_i are outward normals
        # For the hexagon to contain all points, we need max_i (n_i . p) <= s for all points
        # So s = max over all points p, all normals n_i of (n_i . p)
        
        # Normals of outer hexagon
        normal_angles = np.pi/3 * np.arange(6) + phi
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Compute support values: for each vertex and each normal, compute dot product
        # Shape: (72, 6)
        support = rel_verts @ normals.T
        
        # The required side length s is the max of all these values (and also the max of -support for opposite sides)
        # Actually, for a regular hexagon centered at origin, the constraint is:
        # For each of the 6 side normals n_i: max_p (n_i . p) <= s
        # AND for the opposite: min_p (n_i . p) >= -s, i.e., max_p (-n_i . p) <= s
        # So s = max( max_i max_p (n_i . p), max_i max_p (-n_i . p) )
        # = max( max over all (n_i . p), max over all (-n_i . p) )
        # = max over all |n_i . p| ... no, that's not right either.
        
        # Let me reconsider. A regular hexagon with side s, centered at origin, with one side normal along x-axis:
        # The 6 side outward normals are at angles 0, 60, 120, 180, 240, 300.
        # The hexagon is the intersection of 6 half-planes: n_i . p <= s for i=0..5
        # where n_i are unit vectors.
        # So the minimum s such that all points are inside is:
        # s = max_{p in all_verts} max_{i=0..5} (n_i . p)
        
        # But wait, this assumes the hexagon is symmetric and the normals are as stated.
        # For a regular hexagon of side s, the apothem (distance from center to side) is s * sqrt(3)/2.
        # So the half-plane constraint is n_i . p <= apothem = s * sqrt(3)/2
        # Therefore s = (2/sqrt(3)) * max_{p,i} (n_i . p)
        
        max_support = np.max(support)
        # Also need to check the "negative" side - but since normals cover all 6 directions,
        # and the hexagon is centrally symmetric, max_support should capture it.
        # Actually no: if a point is at (-10, 0) and normal is (1,0), then n.p = -10, which is fine.
        # But if normal is (-1,0) (which is normal at 180 degrees), then n.p = 10, so max_support >= 10.
        # So yes, taking max over all 6 normals handles both sides.
        
        s_required = (2.0 / np.sqrt(3.0)) * max_support
        
        return s_required
    
    def objective_smooth(params, beta=10.0):
        """Smooth version using log-sum-exp for the max."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return np.inf
        
        pos = params[:N*2].reshape(N, 2)
        ang = params[N*2:N*2+N]
        cx, cy = params[N*2+N:N*2+N+2]
        phi = params[N*2+N+2]
        
        all_verts = get_all_vertices(pos, ang)
        rel_verts = all_verts - np.array([cx, cy])
        
        normal_angles = np.pi/3 * np.arange(6) + phi
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        support = rel_verts @ normals.T  # shape (72, 6)
        flat_support = support.ravel()  # shape (432,)
        
        # Smooth max via log-sum-exp
        max_val = np.max(flat_support)
        smooth_max = max_val + np.log(np.sum(np.exp(flat_support - max_val)) / len(flat_support))
        
        s_required = (2.0 / np.sqrt(3.0)) * smooth_max
        return s_required
    
    # Initial parameter vector
    x0 = np.zeros(N*2 + N + 2 + 1)
    x0[:N*2] = positions.flatten()
    x0[N*2:N*2+N] = np.radians(angles)
    x0[N*2+N:N*2+N+2] = [0.0, 0.0]
    x0[N*2+N+2] = 0.0
    
    # Try multiple initializations
    best_result = None
    best_value = np.inf
    
    # Initial optimization
    result = minimize(objective_smooth, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
    
    if result.fun < best_value:
        best_value = result.fun
        best_result = result.x
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try different initial arrangements
        # Random perturbations
        np.random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Perturb initial positions
            x0_trial = x0.copy()
            x0_trial[:N*2] += np.random.randn(N*2) * 0.5
            x0_trial[N*2:N*2+N] += np.random.randn(N) * 0.3
            x0_trial[N*2+N:N*2+N+2] += np.random.randn(2) * 0.3
            x0_trial[N*2+N+2] += np.random.randn() * 0.3
            
            res = minimize(objective_smooth, x0_trial, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            
            if res.fun < best_value:
                best_value = res.fun
                best_result = res.x
    
    # Final polish with tighter tolerance
    if best_result is not None and time.time() < deadline - 5:
        res_final = minimize(objective, best_result, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
        if res_final.fun < best_value:
            best_value = res_final.fun
            best_result = res_final.x
    
    if best_result is None:
        # Fallback: return the initial guess
        inner = [[positions[i,0], positions[i,1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    pos = best_result[:N*2].reshape(N, 2)
    ang = best_result[N*2:N*2+N]
    cx, cy = best_result[N*2+N:N*2+N+2]
    phi = best_result[N*2+N+2]
    
    # Compute actual side length (non-smooth)
    all_verts = get_all_vertices(pos, np.radians(ang))
    rel_verts = all_verts - np.array([cx, cy])
    normal_angles = np.pi/3 * np.arange(6) + phi
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    support = rel_verts @ normals.T
    max_support = np.max(support)
    side_length = (2.0 / np.sqrt(3.0)) * max_support
    
    # Build inner list: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        inner.append([float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))])
    
    return inner, (float(cx), float(cy)), float(side_length), float(np.degrees(phi))
# EVOLVE-BLOCK-END

