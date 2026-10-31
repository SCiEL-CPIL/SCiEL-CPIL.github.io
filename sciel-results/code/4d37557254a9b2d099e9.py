import numpy as np
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def objective_value(bary):
    """Return the minimum area (positive value)."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_seed_configurations():
    """
    Generate several diverse seed configurations using symmetry and heuristic patterns.
    """
    configs = []
    
    # 1. Center + 10 points in two concentric rings
    center = np.array([[0.5, 0.5]])
    # Ring 1: 5 points
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r1 = 0.35
    ring1 = []
    for a in angles1:
        # Approximate barycentric from cartesian offset from centroid (1/3, 1/3)
        # Centroid in barycentric is (1/3, 1/3)
        # Cartesian: x = v + 0.5w, y = w*H
        # Let's just generate random-ish but structured
        pass
    
    # Simpler: use known good heuristic - place points near vertices and edges
    # Configuration A: 3 vertices, 3 midpoints of edges, 3 midpoints of medians, 1 center
    # Vertices in barycentric: (1,0), (0,1), (0,0)
    # Midpoints of edges: (0.5, 0), (0, 0.5), (0.5, 0.5)
    # Midpoints of medians: (2/3, 1/6), (1/6, 2/3), (1/3, 1/3) -> wait, centroid is (1/3,1/3)
    # Let's try: 3 corners, 3 edge midpoints, 3 inner ring, 1 center
    
    config_a = np.array([
        [0.9, 0.05], # near V0
        [0.05, 0.9], # near V1
        [0.05, 0.05], # near V2
        [0.45, 0.0],  # edge V0-V1 mid
        [0.0, 0.45],  # edge V1-V2 mid
        [0.45, 0.45], # edge V2-V0 mid (wait, V2 is (0,0) in bary? No, V2 is w=1)
        # Let's be precise.
        # V0: (1,0), V1: (0,1), V2: (0,0)
        # Mid V0V1: (0.5, 0.5)
        # Mid V1V2: (0.5, 0)
        # Mid V2V0: (0, 0.5)
    ])
    # This is getting messy. Let's use a grid-based approach for seeds.
    
    # Seed 1: Uniform grid in barycentric space
    grid_size = 5
    u_vals = np.linspace(0.1, 0.8, grid_size)
    v_vals = np.linspace(0.1, 0.8, grid_size)
    pts = []
    for u in u_vals:
        for v in v_vals:
            if u + v <= 0.9:
                pts.append([u, v])
    if len(pts) >= N:
        # Select N points that are spread out
        # Simple: take first N
        configs.append(np.array(pts[:N]))
    
    # Seed 2: Random but clustered
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        configs.append(bary)
        
    # Seed 3: Symmetric hexagonal-like
    # Center
    c = [1/3, 1/3]
    # 6 points around center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r = 0.25
    ring = []
    for a in angles:
        # Convert cartesian offset to barycentric is complex, let's just use random perturbation of center
        pass
        
    return configs

def simulate_annealing(bary_init, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = objective_value(bary)
    best_bary = bary.copy()
    best_val = current_val
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    alpha = 0.995  # Cooling rate
    max_iter = 50000
    
    rng = np.random.default_rng(123)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a perturbed solution
        # Move a random point randomly
        idx = rng.integers(0, N)
        # Gaussian perturbation
        sigma = 0.05 * T / 0.01 # Scale perturbation with temperature
        new_u = bary[idx, 0] + rng.normal(0, sigma)
        new_v = bary[idx, 1] + rng.normal(0, sigma)
        
        # Clamp to valid range
        new_u = np.clip(new_u, 1e-12, 1 - 1e-12)
        new_v = np.clip(new_v, 1e-12, 1 - 1e-12)
        if new_u + new_v > 1 - 1e-12:
            scale = (1 - 1e-12) / (new_u + new_v)
            new_u *= scale
            new_v *= scale
        
        new_bary = bary.copy()
        new_bary[idx, 0] = new_u
        new_bary[idx, 1] = new_v
        
        new_val = objective_value(new_bary)
        delta = new_val - current_val
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or rng.random() < np.exp(delta / T):
            bary = new_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = bary.copy()
        
        # Cool down
        T = max(T * alpha, T_min)
        
        # Adaptive cooling: if stuck, reheat slightly
        if i % 1000 == 0 and i > 0:
            # Check if we've improved in the last 1000 steps
            # This is a simplification, just keep cooling
            pass
            
    return best_bary, best_val

def coordinate_descent(bary_init, time_limit=5.0):
    """
    Coordinate descent optimization.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = objective_value(bary)
    best_bary = bary.copy()
    best_val = current_val
    
    # Simple gradient-free coordinate ascent
    # For each point, try to move it to maximize the min area
    
    step_size = 0.01
    rng = np.random.default_rng(456)
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            # Try small moves in u and v directions
            for du in [-step_size, 0, step_size]:
                for dv in [-step_size, 0, step_size]:
                    if du == 0 and dv == 0:
                        continue
                    new_u = bary[i, 0] + du
                    new_v = bary[i, 1] + dv
                    if new_u < 0 or new_v < 0 or new_u + new_v > 1:
                        continue
                    new_bary = bary.copy()
                    new_bary[i, 0] = new_u
                    new_bary[i, 1] = new_v
                    new_val = objective_value(new_bary)
                    if new_val > current_val:
                        bary = new_bary
                        current_val = new_val
                        improved = True
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
        
        if current_val > best_val:
            best_val = current_val
            best_bary = bary.copy()
            
    return best_bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    # 1. Generate diverse seeds
    seeds = []
    
    # Random seeds
    rng = np.random.default_rng(42)
    for i in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        seeds.append(np.column_stack([u, v]))
        
    # Structured seed: center + ring
    center = [1/3, 1/3]
    ring_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    ring_r = 0.3
    ring_pts = []
    for a in ring_angles:
        # Approximate barycentric coordinates for a point at radius r from center
        # This is an approximation, will be refined by SA
        u = 1/3 + ring_r * np.cos(a) * 0.5
        v = 1/3 + ring_r * np.sin(a) * 0.5
        # Clamp
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u + v > 1:
            s = 1 / (u + v)
            u *= s
            v *= s
        ring_pts.append([u, v])
    seeds.append(np.array(ring_pts))
    
    # 2. For each seed, run Simulated Annealing
    for i, seed in enumerate(seeds):
        if time.time() > deadline - 20:
            break
            
        remaining_time = deadline - time.time()
        sa_time = min(15.0, remaining_time * 0.5)
        
        bary_sa, val_sa = simulate_annealing(seed, time_limit=sa_time)
        
        if val_sa > best_val:
            best_val = val_sa
            best_bary = bary_sa.copy()
            
    # 3. Refine the best solution with coordinate descent
    if best_bary is not None and time.time() < deadline - 5:
        bary_cd, val_cd = coordinate_descent(best_bary, time_limit=5.0)
        if val_cd > best_val:
            best_val = val_cd
            best_bary = bary_cd.copy()
            
    # 4. Final polish with SA on the best solution
    if best_bary is not None and time.time() < deadline - 2:
        bary_final, val_final = simulate_annealing(best_bary, time_limit=2.0)
        if val_final > best_val:
            best_val = val_final
            best_bary = bary_final.copy()
            
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
