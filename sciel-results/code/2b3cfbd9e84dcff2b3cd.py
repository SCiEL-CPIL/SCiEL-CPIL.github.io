import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin to assume the outer hexagon is centered there
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon (centered at 0,0)
        # Distance from center to side is L * sqrt(3) / 2
        # Projection of points onto normals must be <= L * sqrt(3) / 2
        # L = max( max_i |p_i . n_k| ) / (sqrt(3)/2)
        
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # We want to minimize L_enc, but heavily penalize overlaps
        # If there are overlaps, the solution is invalid, so we add a large penalty
        return L_enc + 1000.0 * penalty

    def get_solution(params):
        points = params.reshape(-1, 2)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        return L_enc, valid

    best_L = float('inf')
    best_params = None
    
    # Helper to generate hexagonal lattice points
    def generate_hex_lattice(n_points):
        pts = [[0.0, 0.0]]
        ring = 1
        while len(pts) < n_points:
            # Generate points on ring 'ring'
            # A hexagonal ring of radius r has 6*r points
            # We need to generate them in order
            for i in range(6 * ring):
                if len(pts) >= n_points:
                    break
                # Axial coordinates for ring i
                # Directions: 0, 1, 2, 3, 4, 5
                # Step i
                # This is a bit complex to get exactly n points.
                # Easier: generate a large grid and pick closest.
                pass
        
        # Simpler: generate grid and pick
        q_range = range(-4, 5)
        r_range = range(-4, 5)
        candidates = []
        for q in q_range:
            for r in r_range:
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                candidates.append([x, y])
        
        candidates = np.array(candidates)
        dists = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(dists)[:n_points]
        return candidates[indices]

    # Initial guesses
    inits = []
    
    # 1. Hexagonal lattice
    lattice_pts = generate_hex_lattice(N)
    inits.append(lattice_pts.flatten())
    
    # 2. Perturbed lattice
    for _ in range(10):
        perturbed = lattice_pts + np.random.normal(0, 0.2, lattice_pts.shape)
        inits.append(perturbed.flatten())
        
    # 3. Random uniform
    for _ in range(20):
        random_pts = np.random.uniform(-3, 3, (N, 2))
        inits.append(random_pts.flatten())

    # Optimize each initial guess
    for initial_params in inits:
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.fun < best_L:
            L_enc, valid = get_solution(res.x)
            if valid:
                best_L = L_enc
                best_params = res.x
            else:
                # If invalid, we might still update if it's "less invalid" but we prefer valid
                # For now, only update if valid to ensure we return a valid solution
                # But what if we never find a valid one? 
                # The lattice init should be valid or close to it.
                pass

    # If no valid solution found, fallback
    if best_params is None:
        # Try to find ANY valid solution, even if large
        # Use the lattice points directly
        lattice_pts = generate_hex_lattice(N)
        centroid = np.mean(lattice_pts, axis=0)
        pts_c = lattice_pts - centroid
        projections = np.abs(pts_c @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Check validity
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts_c[i] - pts_c[j])
                if d < MIN_DIST - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_L = L_enc
            best_params = lattice_pts.flatten()
        else:
            # Emergency fallback: spread out
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner_pts = best_params.reshape(-1, 2)
    # Re-center to ensure centroid is 0
    centroid = np.mean(inner_pts, axis=0)
    inner_pts = inner_pts - centroid
    
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    # Recalculate L for the final centered points to be precise
    projections = np.abs(inner_pts @ normals.T)
    max_proj = np.max(projections)
    outer_side = max_proj / (np.sqrt(3) / 2)
    
    return inner, (0.0, 0.0), outer_side, 0.0
