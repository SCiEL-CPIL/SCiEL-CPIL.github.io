import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def cartesian_to_barycentric(cart):
    """Convert (N,2) cartesian to (N,3) barycentric."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def clamp_bary(bary):
    """Ensure barycentric coordinates are valid (sum to 1, non-negative)."""
    bary = np.clip(bary, 0, None)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-12)
    return bary / sums

def calc_areas_bary(bary):
    """Calculate areas of all triangles from barycentric coordinates."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def hard_min_obj(bary_flat):
    """Objective: Negative of the minimum area."""
    bary = clamp_bary(bary_flat.reshape(N, 3))
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
    """Smooth objective: Negative of the log-sum-exp of areas."""
    bary = clamp_bary(bary_flat.reshape(N, 3))
    areas = calc_areas_bary(bary)
    # Soft min approximation: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, use max subtraction
    min_a = np.min(areas)
    scaled = -beta * (areas - min_a)
    log_sum = min_a - (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
    return -log_sum

def generate_hex_init():
    """Generate a hexagonal lattice initialization."""
    # Create a hexagonal grid in Cartesian space
    # Triangle: (0,0), (1,0), (0.5, H)
    # We want to fit 11 points.
    # A hexagonal grid with spacing s.
    # Area of hexagon cell ~ 0.866 * s^2.
    # Total area ~ 0.433.
    # 11 points -> density ~ 11/0.433 ~ 25.
    # s ~ sqrt(0.433/25) ~ 0.13.
    
    s = 0.15
    pts = []
    # Generate points in a bounding box
    # x in [0, 1], y in [0, H]
    # Hex grid:
    # Row j: y = j * s * sqrt(3)/2
    # x = i * s + (j % 2) * s / 2
    
    max_j = int(H / (s * np.sqrt(3)/2)) + 2
    max_i = int(1.0 / s) + 2
    
    for j in range(max_j):
        y = j * s * np.sqrt(3) / 2
        if y > H: break
        for i in range(max_i):
            x = i * s + (j % 2) * s / 2
            if x > 1.0: break
            # Check if (x,y) is inside the triangle
            # Triangle edges:
            # 1. y >= 0
            # 2. y <= H * x  (Left edge? No, (0,0) to (0.5, H) is y = 2*H*x? No.
            #    Slope of (0,0) to (0.5, H) is H/0.5 = 2H = sqrt(3).
            #    So y <= sqrt(3) * x
            # 3. y <= H * (2 - 2x)? No.
            #    Slope of (1,0) to (0.5, H) is (H-0)/(0.5-1) = -2H = -sqrt(3).
            #    Line: y - 0 = -sqrt(3)(x - 1) => y = -sqrt(3)x + sqrt(3).
            #    So y <= sqrt(3) * (1 - x)
            
            if y >= 0 and y <= np.sqrt(3) * x + 1e-9 and y <= np.sqrt(3) * (1 - x) + 1e-9:
                pts.append([x, y])
                
    pts = np.array(pts)
    
    # If we have more than 11 points, select 11 that maximize min distance (greedy)
    # Or just take the first 11. Greedy is better.
    if len(pts) >= N:
        selected = []
        remaining = pts.copy()
        # Start with the center of the triangle
        center = np.array([0.5, H/3])
        # Find closest point to center
        dists = np.linalg.norm(remaining - center, axis=1)
        idx = np.argmin(dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        for _ in range(N - 1):
            if not remaining.size: break
            # For each remaining point, find min distance to selected points
            min_dists = np.inf * np.ones(len(remaining))
            for k in range(len(remaining)):
                d = np.linalg.norm(remaining[k] - np.array(selected), axis=1)
                min_dists[k] = np.min(d)
            # Pick the one with the largest min_dist
            idx = np.argmax(min_dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        pts = np.array(selected)
    else:
        # Pad with random points if needed
        while len(pts) < N:
            # Random point in triangle
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = H * v
            pts = np.vstack([pts, [x, y]])
            
    # Convert to barycentric
    bary = cartesian_to_barycentric(pts[:N])
    return clamp_bary(bary).flatten()

def generate_random_init():
    """Generate a random initialization."""
    u = np.random.random(N)
    v = np.random.random(N)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    x = u + 0.5 * v
    y = H * v
    cart = np.column_stack([x, y])
    bary = cartesian_to_barycentric(cart)
    return clamp_bary(bary).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    best_val = -np.inf
    best_bary = None
    
    def update_best(bary_candidate, val_candidate):
        nonlocal best_val, best_bary
        if val_candidate > best_val:
            best_val = val_candidate
            best_bary = bary_candidate.copy()
            
    # 1. Initial Guesses
    inits = []
    inits.append(generate_hex_init())
    for _ in range(5):
        inits.append(generate_random_init())
        
    for b0 in inits:
        val = -hard_min_obj(b0)
        update_best(b0, val)
        
    # 2. Differential Evolution (Global Search)
    # Use soft objective for smoothness
    if time.time() < deadline - 60:
        try:
            bounds = [(0, 1)] * (N * 3)
            de_result = differential_evolution(
                lambda x: soft_min_obj(x, beta=50.0),
                bounds=bounds,
                seed=42,
                maxiter=30,
                tol=1e-4,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            val_hard = -hard_min_obj(de_result.x)
            update_best(de_result.x, val_hard)
        except Exception:
            pass

    # 3. Local Polish Loop
    # Alternate between Nelder-Mead (Hard) and L-BFGS-B (Soft)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        
        # Nelder-Mead with Hard Min (Non-smooth, robust)
        if time.time() < deadline - 15:
            try:
                result = minimize(
                    hard_min_obj,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                val_hard = -hard_min_obj(result.x)
                update_best(result.x, val_hard)
            except Exception:
                pass
        
        # L-BFGS-B with Soft Min (Gradient based, fast)
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    lambda x: soft_min_obj(x, beta=200.0),
                    best_bary,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (N * 3),
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                val_hard = -hard_min_obj(result.x)
                update_best(result.x, val_hard)
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                hard_min_obj,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            val_hard = -hard_min_obj(result.x)
            update_best(result.x, val_hard)
        except Exception:
            pass

    # Convert to cartesian
    bary_final = clamp_bary(best_bary.reshape(N, 3))
    cart = barycentric_to_cartesian(bary_final)
    
    return cart
