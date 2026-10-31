"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

def bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian in unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = h * bary[:, 2]
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian to barycentric coordinates."""
    h = np.sqrt(3) / 2
    x, y = cart[:, 0], cart[:, 1]
    bary2 = y / h
    bary1 = x - 0.5 * bary2
    bary0 = 1.0 - bary1 - bary2
    return np.column_stack([bary0, bary1, bary2])

def clamp_bary(bary):
    """Clamp barycentric coordinates to be valid (non-negative, sum to 1)."""
    bary = np.maximum(bary, 0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-12)
    bary = bary / s
    return bary

def calc_areas_bary(bary_points):
    """Compute areas of all triangles from barycentric points.
    Area in barycentric = |det| * area_of_reference_triangle
    For unit equilateral triangle, reference area = sqrt(3)/4
    """
    n = len(bary_points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        # The area of triangle formed by three points in barycentric coords
        # is proportional to |det of the 3x3 matrix with rows as bary coords|
        M = np.array([bary_points[i], bary_points[j], bary_points[k]])
        det_val = np.linalg.det(M)
        # Area of reference triangle is sqrt(3)/4
        area = abs(det_val) * (np.sqrt(3) / 4)
        areas.append(area)
    return np.array(areas)

def soft_min_obj(params, beta=50.0):
    """Smooth objective using log-sum-exp for soft minimum of triangle areas."""
    bary = params.reshape(-1, 3)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    if len(areas) == 0:
        return 0.0
    # Soft minimum: -logsumexp(-areas / beta) * beta
    # This is smooth and approximates min(areas) as beta -> 0
    min_val = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-areas / beta)))
    soft_min = -beta * log_sum + beta * np.log(len(areas))
    # Actually: soft_min = -beta * log(mean(exp(-areas/beta))) = -beta * logsumexp(-areas/beta) + beta*log(n)
    soft_min = -beta * (np.log(np.sum(np.exp(-areas / beta))) - np.log(len(areas)))
    return -soft_min

def hard_min_obj(params):
    """Hard minimum objective for final evaluation."""
    bary = params.reshape(-1, 3)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    if len(areas) == 0:
        return 0.0
    return -np.min(areas)

def generate_hexagonal_init():
    """Generate hexagonal-like pattern in barycentric coordinates."""
    points = []
    # Center
    points.append([1/3, 1/3, 1/3])
    # Ring 1: 6 points around center
    r1 = 0.25
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        # In barycentric, we need to map circular to triangular
        # Use a simple mapping: perturb center
        b = [1/3, 1/3, 1/3]
        # Add perturbation in barycentric space
        dx = r1 * np.cos(angle)
        dy = r1 * np.sin(angle)
        # Convert small Cartesian displacement to barycentric
        h = np.sqrt(3)/2
        db2 = dy / h
        db1 = dx - 0.5 * db2
        db0 = -db1 - db2
        b = np.array(b) + np.array([db0, db1, db2])
        b = clamp_bary(b.reshape(1,3))[0]
        points.append(b)
    # Ring 2: 6 more points
    r2 = 0.45
    for angle in np.linspace(np.pi/6, 2*np.pi + np.pi/6, 6, endpoint=False):
        b = [1/3, 1/3, 1/3]
        dx = r2 * np.cos(angle)
        dy = r2 * np.sin(angle)
        h = np.sqrt(3)/2
        db2 = dy / h
        db1 = dx - 0.5 * db2
        db0 = -db1 - db2
        b = np.array(b) + np.array([db0, db1, db2])
        b = clamp_bary(b.reshape(1,3))[0]
        points.append(b)
    return np.array(points[:N])

def generate_random_init(rng):
    """Generate random valid barycentric points."""
    points = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1-u, 1-v
        w = 1 - u - v
        # Add small jitter to avoid exact corners
        u = np.clip(u + 0.02, 0.01, 0.98)
        v = np.clip(v + 0.02, 0.01, 0.98)
        w = np.clip(w + 0.02, 0.01, 0.98)
        s = u + v + w
        points.append([u/s, v/s, w/s])
    return np.array(points)

def generate_perturbed_init(base, rng, sigma=0.1):
    """Generate perturbed version of base configuration."""
    bary = base.copy()
    noise = rng.normal(0, sigma, bary.shape)
    bary = bary + noise
    bary = clamp_bary(bary)
    return bary

def _local_polish_bary(bary_points, deadline):
    """Local refinement in barycentric space using smooth objective."""
    bary = bary_points.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    # Use L-BFGS-B with smooth objective
    for pass_num in range(3):
        if time.time() > deadline - 5:
            break
        
        def objective(params):
            return soft_min_obj(params, beta=30.0)
        
        try:
            result = minimize(objective, bary.flatten(), method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-6})
            if result.fun < current_val:
                new_bary = clamp_bary(result.x.reshape(-1, 3))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val > current_val:
                    bary = new_bary
                    current_val = new_val
        except Exception:
            pass
        
        if time.time() > deadline - 3:
            break
        
        # Coordinate descent in barycentric space
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(N):
                for coord in range(3):
                    for step in [0.02, 0.01, 0.005, 0.001]:
                        old_val = bary[i, coord]
                        bary[i, coord] = old_val + step
                        bary = clamp_bary(bary)
                        new_val = -hard_min_obj(bary.flatten())
                        if new_val > current_val:
                            current_val = new_val
                            improved = True
                            continue
                        bary[i, coord] = old_val - step
                        bary = clamp_bary(bary)
                        new_val = -hard_min_obj(bary.flatten())
                        if new_val > current_val:
                            current_val = new_val
                            improved = True
                            continue
                        bary[i, coord] = old_val
            if not improved:
                break
    
    return bary, current_val

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_bary = None
    best_val = 0.0
    
    # Strategy 1: Differential evolution in barycentric space
    if time.time() < deadline - 40:
        bounds = [(0, 1)] * 3 * N
        
        def de_objective(params):
            return soft_min_obj(params, beta=20.0)
        
        try:
            de_result = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=150, popsize=20,
                tol=1e-7, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol',
                update_interval=10
            )
            de_bary = clamp_bary(de_result.x.reshape(-1, 3))
            de_val = -hard_min_obj(de_bary.flatten())
            if de_val > best_val:
                best_val = de_val
                best_bary = de_bary.copy()
        except Exception:
            pass
    
    # Strategy 2: Multiple initializations with local polish
    if best_bary is None or time.time() < deadline - 25:
        rng = np.random.RandomState(42)
        
        # Start with hexagonal
        hex_init = generate_hexagonal_init()
        polished, val = _local_polish_bary(hex_init, deadline)
        if val > best_val:
            best_val = val
            best_bary = polished
        
        # Random and perturbed initializations
        for seed in range(12):
            if time.time() > deadline - 20:
                break
            
            if seed % 3 == 0:
                init = generate_random_init(rng)
            elif seed % 3 == 1:
                init = generate_perturbed_init(hex_init, rng, sigma=0.15)
            else:
                init = generate_perturbed_init(hex_init, rng, sigma=0.3)
            
            polished, val = _local_polish_bary(init, deadline)
            if val > best_val:
                best_val = val
                best_bary = polished
    
    # Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        # Use tighter tolerance for final polish
        for _ in range(2):
            if time.time() > deadline - 3:
                break
            def final_obj(params):
                return soft_min_obj(params, beta=10.0)
            try:
                result = minimize(final_obj, best_bary.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 50, 'gtol': 1e-8})
                if result.fun < -best_val:
                    new_bary = clamp_bary(result.x.reshape(-1, 3))
                    new_val = -hard_min_obj(new_bary.flatten())
                    if new_val > best_val:
                        best_bary = new_bary
                        best_val = new_val
            except Exception:
                pass
    
    if best_bary is None:
        # Fallback
        best_bary = generate_hexagonal_init()
    
    # Convert to Cartesian
    cart = bary_to_cart(best_bary)
    return cart
# EVOLVE-BLOCK-END
