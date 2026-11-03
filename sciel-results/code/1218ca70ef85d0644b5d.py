import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Calculate the packing radius for a set of points."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    
    if min_boundary < 0:
        # Invalid, but we return a low score rather than -inf to allow optimizer to work
        # Actually, if it's outside, the "radius" is negative.
        pass

    # Pairwise distances
    # For N=50, 1225 pairs.
    # Vectorized:
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair_dist)

def _soft_min(x, beta=10.0):
    """Soft minimum using log-sum-exp trick for stability."""
    # x is a 1D array of distances
    # We want min(x).
    # soft_min = -1/beta * log(sum(exp(-beta * x)))
    # To avoid overflow, subtract max(-beta*x) = -beta*min(x)
    # Let y = -beta * x
    # min(x) is approx -1/beta * log(sum(exp(y)))
    # y_max = -beta * min(x)
    # sum(exp(y)) = sum(exp(y - y_max)) * exp(y_max)
    # log(sum(exp(y))) = log(sum(exp(y - y_max))) + y_max
    # soft_min = -1/beta * (log_sum_exp(y - y_max) + y_max)
    
    y = -beta * x
    y_max = np.max(y)
    if y_max > 700: # Overflow protection, though y_max is -beta*min(x), and x is positive usually
         y_max = 700
    if y_max < -700:
         y_max = -700
         
    sum_exp = np.sum(np.exp(y - y_max))
    if sum_exp == 0:
        return -np.inf
    log_sum_exp = np.log(sum_exp) + y_max
    return -log_sum_exp / beta

def _optimize_structure(counts, initial_radii=None, initial_phases=None, beta=20.0):
    """
    Optimize radii and phases for a given ring structure.
    counts: list of integers, e.g., [1, 6, 12, ...]
    Returns: (points, score)
    """
    k = len(counts)
    # Variables: r_1, phi_1, r_2, phi_2, ...
    # Bounds for r: [0, 1]
    # Bounds for phi: [0, 2*pi]
    
    if initial_radii is None:
        # Initial guess: spread radii from 0 to 1
        initial_radii = np.linspace(0.1, 0.9, k)
    if initial_phases is None:
        initial_phases = np.zeros(k)
        
    x0 = np.zeros(2*k)
    for i in range(k):
        x0[2*i] = initial_radii[i]
        x0[2*i+1] = initial_phases[i]
        
    def objective(x):
        radii = x[0::2]
        phases = x[1::2]
        
        # Clip radii to valid range to avoid weirdness
        radii = np.clip(radii, 0.0, 1.0)
        
        points = []
        for i in range(k):
            n_i = counts[i]
            r_i = radii[i]
            phi_i = phases[i]
            
            if n_i == 1:
                # Center or single point
                # If it's the first ring, it should be at center?
                # Actually, if r_i > 0 and n_i=1, it's just a point at angle phi_i.
                # But usually the first ring is the center (r=0).
                # Let's just calculate positions.
                angles = np.array([phi_i])
            else:
                angles = phi_i + np.arange(n_i) * (2 * np.pi / n_i)
                
            xs = r_i * np.cos(angles)
            ys = r_i * np.sin(angles)
            points.append(np.column_stack((xs, ys)))
            
        points = np.vstack(points)
        
        if len(points) != N:
            return 1e10 # Penalty
            
        # Calculate distances
        # Boundary
        radii_pts = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii_pts
        
        # Pairwise
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Combine all distances
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        
        # Use soft min for optimization
        # We want to maximize the minimum distance.
        # So minimize -soft_min(all_dists)
        sm = _soft_min(all_dists, beta)
        return -sm

    # Optimization
    # Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 200})
    
    x_opt = res.x
    radii = x_opt[0::2]
    phases = x_opt[1::2]
    radii = np.clip(radii, 0.0, 1.0)
    
    points = []
    for i in range(k):
        n_i = counts[i]
        r_i = radii[i]
        phi_i = phases[i]
        
        if n_i == 1:
            angles = np.array([phi_i])
        else:
            angles = phi_i + np.arange(n_i) * (2 * np.pi / n_i)
            
        xs = r_i * np.cos(angles)
        ys = r_i * np.sin(angles)
        points.append(np.column_stack((xs, ys)))
        
    points = np.vstack(points)
    
    # Calculate true score
    score = _score(points)
    
    return points, score

def _generate_structures():
    """Generate a list of candidate ring count structures."""
    structures = []
    
    # Base structures
    base = [1, 6, 12, 18, 13]
    structures.append(base)
    
    # Variations
    # Move 1 circle between adjacent rings
    for i in range(len(base) - 1):
        if base[i] > 1:
            v1 = base.copy()
            v1[i] -= 1
            v1[i+1] += 1
            structures.append(v1)
        if base[i+1] > 1:
            v2 = base.copy()
            v2[i] += 1
            v2[i+1] -= 1
            structures.append(v2)
            
    # Other plausible structures
    structures.append([1, 6, 12, 17, 14])
    structures.append([1, 6, 11, 18, 14])
    structures.append([1, 6, 12, 16, 15])
    structures.append([1, 6, 13, 18, 12])
    
    # Remove duplicates
    unique_structures = []
    seen = set()
    for s in structures:
        # Ensure sum is N
        if sum(s) != N:
            continue
        # Ensure no ring is empty (except maybe first if we handle it, but we assume 1 center)
        if any(c <= 0 for c in s):
            continue
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)
            
    return unique_structures

def construct():
    start_time = time.time()
    deadline = 110.0 # Leave 40s margin
    
    structures = _generate_structures()
    
    best_score = -1
    best_points = None
    
    # Fallback: Hex lattice
    def seed_construct():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    # Try optimizing each structure
    for counts in structures:
        if time.time() > deadline:
            break
            
        # Try multiple initial phase configurations
        for restart in range(3):
            if time.time() > deadline:
                break
                
            if restart == 0:
                init_phases = np.zeros(len(counts))
            else:
                init_phases = np.random.uniform(0, 2*np.pi, len(counts))
                
            try:
                points, score = _optimize_structure(counts, initial_phases=init_phases)
                if score > best_score:
                    best_score = score
                    best_points = points
            except Exception:
                pass
                
    if best_points is None or not np.all(np.isfinite(best_points)):
        return seed_construct()
        
    return best_points

# EVOLVE-BLOCK-END
