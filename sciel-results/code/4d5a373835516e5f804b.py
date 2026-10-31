import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
H = np.sqrt(3) / 2.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def bary_to_cart(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def cart_to_bary(cart):
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

def calc_areas(bary):
    """Calculate areas of all triangles from barycentric coordinates."""
    cart = bary_to_cart(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def hard_obj(bary_flat):
    """Objective: Negative of the minimum area."""
    bary = clamp_bary(bary_flat.reshape(N, 3))
    areas = calc_areas(bary)
    return -np.min(areas)

def soft_obj(bary_flat, beta=100.0):
    """Smooth objective: Negative of the log-sum-exp of areas."""
    bary = clamp_bary(bary_flat.reshape(N, 3))
    areas = calc_areas(bary)
    min_a = np.min(areas)
    scaled = -beta * (areas - min_a)
    log_sum = min_a - (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
    return -log_sum

def generate_hex_init():
    """Generate a hexagonal lattice initialization."""
    s = 0.15
    pts = []
    max_j = int(H / (s * np.sqrt(3) / 2)) + 2
    max_i = int(1.0 / s) + 2
    
    for j in range(max_j):
        y = j * s * np.sqrt(3) / 2
        if y > H:
            break
        for i in range(max_i):
            x = i * s + (j % 2) * s / 2
            if x > 1.0:
                break
            if y >= 0 and y <= np.sqrt(3) * x + 1e-9 and y <= np.sqrt(3) * (1 - x) + 1e-9:
                pts.append([x, y])
                
    pts = np.array(pts)
    
    if len(pts) >= N:
        selected = []
        remaining = pts.copy()
        center = np.array([0.5, H / 3])
        dists = np.linalg.norm(remaining - center, axis=1)
        idx = np.argmin(dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        for _ in range(N - 1):
            if not remaining.size:
                break
            min_dists = np.inf * np.ones(len(remaining))
            for k in range(len(remaining)):
                d = np.linalg.norm(remaining[k] - np.array(selected), axis=1)
                min_dists[k] = np.min(d)
            idx = np.argmax(min_dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        pts = np.array(selected)
    else:
        while len(pts) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = H * v
            pts = np.vstack([pts, [x, y]])
            
    bary = cart_to_bary(pts[:N])
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
    bary = cart_to_bary(cart)
    return clamp_bary(bary).flatten()

def generate_perturbed_init(base_bary, sigma=0.05):
    """Generate a perturbed version of a base barycentric configuration."""
    bary = base_bary.reshape(N, 3) + np.random.normal(0, sigma, (N, 3))
    return clamp_bary(bary).flatten()

def refine_local(bary_flat, deadline):
    """Refine a barycentric configuration using local optimization."""
    best_bary = bary_flat.copy()
    best_val = -hard_obj(best_bary)
    
    # Nelder-Mead with hard min
    if time.time() < deadline - 5:
        try:
            result = minimize(
                hard_obj,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            val = -hard_obj(result.x)
            if val > best_val:
                best_val = val
                best_bary = result.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B with soft min
    if time.time() < deadline - 3:
        try:
            result = minimize(
                lambda x: soft_obj(x, beta=200.0),
                best_bary,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-14}
            )
            val = -hard_obj(result.x)
            if val > best_val:
                best_val = val
                best_bary = result.x.copy()
        except Exception:
            pass
    
    return best_bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    
    best_val = -np.inf
    best_bary = None
    
    def update_best(bary_candidate, val_candidate):
        nonlocal best_val, best_bary
        if val_candidate > best_val:
            best_val = val_candidate
            best_bary = bary_candidate.copy()
    
    # 1. Initial guesses
    inits = [generate_hex_init()]
    for _ in range(8):
        inits.append(generate_random_init())
    
    for b0 in inits:
        val = -hard_obj(b0)
        update_best(b0, val)
    
    # 2. Differential evolution with soft objective
    if time.time() < deadline - 50:
        try:
            bounds = [(0, 1)] * (N * 3)
            de_result = differential_evolution(
                lambda x: soft_obj(x, beta=50.0),
                bounds=bounds,
                seed=42,
                maxiter=50,
                tol=1e-4,
                popsize=20,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            val_hard = -hard_obj(de_result.x)
            update_best(de_result.x, val_hard)
        except Exception:
            pass
    
    # 3. Local refinement loop
    for i in range(15):
        if time.time() > deadline - 10:
            break
        
        best_bary, val = refine_local(best_bary, deadline)
        update_best(best_bary, val)
        
        # Perturb and refine
        if time.time() < deadline - 15:
            perturbed = generate_perturbed_init(best_bary, sigma=0.03)
            p_bary, p_val = refine_local(perturbed, deadline)
            update_best(p_bary, p_val)
    
    # 4. Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                hard_obj,
                best_bary,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            val_hard = -hard_obj(result.x)
            update_best(result.x, val_hard)
        except Exception:
            pass
    
    # Convert to cartesian
    bary_final = clamp_bary(best_bary.reshape(N, 3))
    cart = bary_to_cart(bary_final)
    
    return cart
