import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def clamp_bary(bary, eps=1e-9):
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    return np.min(calc_areas_bary(bary))

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_lattice_init():
    """
    Generate points on a triangular grid.
    """
    selected = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.0, 0.5],
        [0.5, 0.5],
        [1/3, 1/3],
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.5, 0.25]
    ])
    return clamp_bary(selected)

def generate_symmetric_orbits(r1, a1, r2, a2, r3, a3):
    """
    Generate points using 3 orbits of 3 points + 1 center + 1 extra.
    """
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3
    
    for i in range(3):
        angle = 2 * np.pi * i / 3 + a1
        bary[i] = [cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)]
        
    for i in range(3):
        angle = 2 * np.pi * i / 3 + a2
        bary[3+i] = [cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)]
        
    for i in range(3):
        angle = 2 * np.pi * i / 3 + a3
        bary[6+i] = [cx + r3 * np.cos(angle), cy + r3 * np.sin(angle)]
        
    bary[9] = [cx, cy]
    bary[10] = [cx + r1*0.5 * np.cos(a1), cy + r1*0.5 * np.sin(a1)]
    
    return clamp_bary(bary)

def refine_nm(bary_init, time_limit=5.0):
    """
    Robust Nelder-Mead refinement.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_val = get_min_area(bary_current)
    
    x0 = bary_current.flatten()
    
    try:
        result = minimize(
            hard_min_obj,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        new_bary = clamp_bary(result.x.reshape(N, 2))
        new_val = get_min_area(new_bary)
        
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
            
    except Exception:
        pass
        
    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    candidates = []

    # 1. Lattice Init
    candidates.append(generate_lattice_init())
    
    # 2. Symmetric Orbits (multiple variations)
    for r1 in [0.3, 0.4]:
        for r2 in [0.15, 0.2]:
            for r3 in [0.05, 0.1]:
                for a1 in [0, np.pi/3]:
                    candidates.append(generate_symmetric_orbits(r1, a1, r2, a1+np.pi/6, r3, a1))
                    
    # 3. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))

    # Process candidates
    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break

        remaining = deadline - 20 - time.time()
        if remaining < 2:
            break

        t_limit = min(8.0, remaining / max(1, n_cands - i))
        
        try:
            refined, val = refine_nm(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Final polish on best
    if best_bary is not None and time.time() < deadline - 2:
        try:
            remaining = deadline - 2 - time.time()
            if remaining > 1:
                refined, val = refine_nm(best_bary, time_limit=remaining)
                update_best(refined)
        except Exception:
            pass

    if best_bary is None:
        best_bary = generate_lattice_init()

    cart = bary_to_cart(best_bary)

    return cart
