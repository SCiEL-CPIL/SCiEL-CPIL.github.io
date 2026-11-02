"""Program for packing 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 125  # Safety margin before 150s kill
    
    def unit_hexagon_vertices():
        """Return vertices of a unit regular hexagon centered at origin, flat-top orientation."""
        return np.array([[np.cos(np.pi/6 + i*np.pi/3), np.sin(np.pi/6 + i*np.pi/3)] for i in range(6)])
    
    def hexagon_width(vertices, angle_rad):
        """Compute the width of a polygon in 6 directions for a given outer rotation angle.
        Returns the maximum over the 6 directions of (max dot - min dot)."""
        # Outer hexagon has 6 faces with normals at angles angle_rad + k*pi/3
        # The width in each direction must be <= outer_side_length
        # For containment, we need: for all k in 0..5, max_i (v_i . n_k) - min_i (v_i . n_k) <= S
        # where n_k = (cos(angle_rad + k*pi/3), sin(angle_rad + k*pi/3))
        # and S is the outer side length.
        # Actually, for a regular hexagon of side S, the width in direction n is S * |n . t| ... 
        # Let me think more carefully.
        # A regular hexagon of side S centered at origin, rotated by angle_rad, has vertices at
        # S * (cos(angle_rad + j*pi/3), sin(angle_rad + j*pi/3)) for j=0..5.
        # The supporting function in direction d is S * max_j (d . (cos(angle_rad + j*pi/3), sin(angle_rad + j*pi/3)))
        # The width in direction d is S * (max_j - min_j) where the max/min are over the 6 vertices.
        # For a regular hexagon, the width in any direction is S * max over the 6 face normals of |d . n_k|... 
        # Actually simpler: the constraint for a set of points to be inside a hexagon of side S rotated by angle_rad
        # is that for each of the 6 face normals n_k (pointing outward from center),
        # the max over all points of (p . n_k) <= S * (n_k . v_k) where v_k is the corresponding vertex.
        # For a regular hexagon of side S, vertex k is at S * u_k where u_k = (cos(angle_rad + k*pi/3), sin(angle_rad + k*pi/3)).
        # The face normal n_k bisects the angle between u_{k-1} and u_{k+1}, so n_k = (cos(angle_rad + k*pi/3), sin(angle_rad + k*pi/3))... 
        # Wait, for a flat-top hexagon, the face normals are at angles 0, 60, 120, ... and the vertices are at 30, 90, 150, ...
        # Let me use a cleaner formulation.
        # A regular hexagon of side S centered at origin, with one vertex at angle angle_rad, has:
        #   Vertices: S * (cos(angle_rad + j*pi/3), sin(angle_rad + j*pi/3)) for j=0..5
        #   Face normals (outward): (cos(angle_rad + (j+0.5)*pi/3), sin(angle_rad + (j+0.5)*pi/3)) for j=0..5
        #   The constraint for point p to be inside: p . n_j <= S * cos(pi/6) = S * sqrt(3)/2 for all j.
        #   Because the distance from center to each face is S * cos(30°) = S * sqrt(3)/2.
        # So the condition is: for all j=0..5, max_p (p . n_j) <= S * sqrt(3)/2.
        # Thus S >= max_j (max_p (p . n_j)) / (sqrt(3)/2) = (2/sqrt(3)) * max_j (max_p (p . n_j)).
        
        normals = np.array([[np.cos(angle_rad + (j + 0.5) * np.pi / 3), np.sin(angle_rad + (j + 0.5) * np.pi / 3)] for j in range(6)])
        # vertices shape: (M, 2), normals shape: (6, 2)
        # dots: (M, 6)
        dots = vertices @ normals.T
        max_dots = np.max(dots, axis=0)  # (6,)
        return (2.0 / np.sqrt(3.0)) * np.max(max_dots)
    
    def get_all_vertices(positions, angles_deg):
        """Given positions (N, 2) and angles (N,), compute all 6N vertices."""
        base = unit_hexagon_vertices()  # (6, 2)
        all_verts = []
        for i in range(len(positions)):
            theta = np.radians(angles_deg[i])
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            verts = base @ R.T + positions[i]
            all_verts.append(verts)
        return np.vstack(all_verts)
    
    def objective(params):
        """params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, outer_angle]
        where a is in radians, outer_angle is the rotation of the outer hexagon."""
        N_pts = N
        pos = params[:3*N_pts].reshape(N_pts, 2)
        angs = params[3*N_pts:3*N_pts+N_pts]
        cx, cy = params[3*N_pts+N_pts], params[3*N_pts+N_pts+1]
        outer_ang = params[3*N_pts+N_pts+2]
        
        # Translate so center is at origin
        pos = pos - np.array([cx, cy])
        
        verts = get_all_vertices(pos, np.degrees(angs))
        S = hexagon_width(verts, outer_ang)
        
        # Add penalty for overlaps? For now, just minimize S.
        # To handle overlaps, we could add a large penalty if any two hexagons overlap.
        # For simplicity in this first version, we'll rely on the optimizer to find non-overlapping configs
        # or we add a penalty term.
        
        # Penalty for overlaps: check pairwise distances between centers
        # Two unit hexagons with relative orientation delta don't overlap if the distance between centers
        # is >= the minimum distance between boundaries. For same orientation, min distance = 2.
        # For arbitrary orientations, it's more complex. Let's use a simple penalty:
        # if dist < 1.9, add penalty.
        penalty = 0.0
        for i in range(N_pts):
            for j in range(i+1, N_pts):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.95:
                    penalty += (1.95 - d) ** 2
        return S + 100.0 * penalty
    
    def bounds():
        b = []
        for _ in range(N):
            b.append([(-3, 3), (-3, 3), (0, 2*np.pi)])
        b.append([(-1, 1), (-1, 1), (0, 2*np.pi)])
        return b
    
    # Multi-start differential evolution
    best_S = np.inf
    best_params = None
    
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Random initial population
        rng = np.random.default_rng(s)
        x0 = []
        for i in range(N):
            x0.extend([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 2*np.pi)])
        x0.extend([0.0, 0.0, rng.uniform(0, 2*np.pi)])
        
        try:
            res = differential_evolution(
                objective, 
                bounds(), 
                maxiter=500, 
                popsize=20,
                tol=1e-4,
                seed=s,
                polish=False
            )
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Polish with L-BFGS-B
    try:
        if time.time() < deadline - 10:
            res2 = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds(),
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x
    except Exception:
        pass
    
    # Extract results
    N_pts = N
    pos = best_params[:3*N_pts].reshape(N_pts, 2)
    angs_rad = best_params[3*N_pts:3*N_pts+N_pts]
    cx, cy = best_params[3*N_pts+N_pts], best_params[3*N_pts+N_pts+1]
    outer_ang_rad = best_params[3*N_pts+N_pts+2]
    
    # Recompute S without penalty for accurate value
    pos_centered = pos - np.array([cx, cy])
    verts = get_all_vertices(pos_centered, np.degrees(angs_rad))
    S_final = hexagon_width(verts, outer_ang_rad)
    
    inner = []
    for i in range(N_pts):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(np.degrees(angs_rad[i]))])
    
    return inner, (float(cx), float(cy)), float(S_final), float(np.degrees(outer_ang_rad))
# EVOLVE-BLOCK-END
