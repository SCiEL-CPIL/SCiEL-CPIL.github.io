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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = x - 0.5 * (1 - y / H) # Wait, let's use the inverse of bary_to_cart properly
    # x = v + 0.5(1-u-v) = 0.5 + v - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)H  => 1-u-v = y/H => u+v = 1 - y/H
    # From x: 2x - 1 = v - u
    # u + v = 1 - y/H
    # v = (2x - 1 + 1 - y/H) / 2 = (2x - y/H) / 2 = x - y/(2H)
    # u = (1 - y/H - 2x + y/H) / 2 = (1 - 2x) / 2 = 0.5 - x
    v = x - y / (2 * H)
    u = 0.5 - x
    return np.column_stack([u, v])

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
        
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_symmetric_init():
    """
    Generate a configuration with 3-fold rotational symmetry around the centroid.
    Centroid of equilateral triangle is (1/3, 1/3) in barycentric.
    We place 2 points in each of the 3 sectors, plus 1 at the center.
    This reduces the effective dimensionality and exploits the likely optimal structure.
    """
    center = np.array([1/3, 1/3])
    points = [center]
    
    # Define 2 points in one sector, then rotate
    # Sector 1: near vertex 0 (u=1, v=0, w=0)
    # Let's pick points based on a hexagonal lattice pattern centered at centroid
    # Hexagonal lattice points around center (0,0) in local coords
    # Distance r from center
    r1 = 0.15
    r2 = 0.30
    
    # Angles for 6 points around center
    angles = [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]
    
    # We need 10 points total (excluding center).
    # Let's use a pattern: 3 points at r1 (one in each sector), 6 points at r2 (two in each sector)?
    # Or simpler: 2 points per sector at specific radii/angles.
    
    # Let's try a "flower" pattern
    # 1 center
    # 3 points at radius 0.2, angles 0, 120, 240 (aligned with vertices)
    # 6 points at radius 0.4, angles 30, 90, 150, 210, 270, 330 (aligned with edges)
    
    # Barycentric rotation matrix
    # Rotation by 120 degrees around centroid (1/3, 1/3, 1/3)
    # In barycentric coords (u,v,w), rotation permutes coordinates.
    # (u,v,w) -> (w,u,v) for 120 deg?
    # Let's verify: V0=(1,0,0). Rot 120 -> V1=(0,1,0).
    # So (1,0,0) -> (0,1,0).
    # If we represent point as (u,v) where w=1-u-v.
    # P = u*V0 + v*V1 + w*V2.
    # Rotated P' = w*V0 + u*V1 + v*V2.
    # So new u' = w, new v' = u.
    
    def rotate_bary(p):
        u, v = p
        w = 1 - u - v
        return np.array([w, u])
    
    def rotate_bary2(p):
        u, v = p
        w = 1 - u - v
        return np.array([v, w])
        
    # Base points in "sector 1" (near V0, high u)
    # Let's define points relative to center in Cartesian, then convert? 
    # Easier to work in barycentric.
    
    # Center is (1/3, 1/3).
    # Vector from center to V0 is (2/3, -1/3).
    # Let's place points along this vector and perpendicular-ish.
    
    # Point 1: On axis to V0, distance d1
    # P1 = center + d1 * (V0 - center)
    # V0 - center = (2/3, -1/3)
    # Let d1 = 0.3
    p1 = center + 0.3 * (V0 - center)
    # p1 = (1/3 + 0.2, 1/3 - 0.1) = (0.533, 0.233)
    
    # Point 2: Off-axis in sector 1
    # Let's pick a point further out and to the side.
    # Maybe use a hexagonal grid point.
    # In a hex grid with spacing s, neighbors are at distance s.
    # Let s = 0.3.
    # Neighbors of center in hex grid:
    # 1. Along axis to V0: center + 0.3*(2/3, -1/3)
    # 2. Along axis to V1: center + 0.3*(-1/3, 2/3)
    # 3. Along axis to V2: center + 0.3*(-1/3, -1/3)
    
    # Let's try:
    # 1 center
    # 3 points at distance 0.25 along axes to vertices
    # 6 points at distance 0.45 in between axes (closer to edges)
    
    p_ax1 = center + 0.25 * (V0 - center)
    p_ax2 = center + 0.25 * (V1 - center)
    p_ax3 = center + 0.25 * (V2 - center)
    
    # Points in between
    # Angle 60 deg from axis to V0, towards V1
    # In barycentric, "between V0 and V1" means high u, high v, low w.
    # Vector to midpoint of V0-V1 edge: (0.5, 0.5) - (1/3, 1/3) = (1/6, 1/6).
    # Let's place points along this direction.
    p_mid1 = center + 0.4 * (np.array([0.5, 0.5]) - center)
    p_mid2 = center + 0.4 * (np.array([0.5, 0.0]) - center) # Midpoint V1-V2 is (0.5, 0.5) in cart? No.
    # V1=(1,0), V2=(0.5, 0.866). Midpoint = (0.75, 0.433). Bary: u=0.5-0.75=-0.25? No.
    # Let's stick to barycentric geometry.
    # Midpoint of V1-V2 in barycentric is (0, 0.5, 0.5) -> (0, 0.5) in (u,v).
    p_mid_v1v2 = center + 0.4 * (np.array([0.0, 0.5]) - center)
    p_mid_v2v0 = center + 0.4 * (np.array([0.5, 0.0]) - center)
    
    # We have 1 + 3 + 3 = 7 points. We need 11.
    # Add 4 more? Or adjust radii.
    # Let's add 3 more points further out, one in each sector.
    p_out1 = center + 0.6 * (V0 - center)
    p_out2 = center + 0.6 * (V1 - center)
    p_out3 = center + 0.6 * (V2 - center)
    
    # Total: 1 (center) + 3 (in) + 3 (mid) + 3 (out) = 10.
    # Need 1 more. Maybe a second ring of mids?
    # Or just make the "mid" points two per sector?
    
    # Let's try a different symmetric set:
    # 1 center
    # 3 points at r=0.2 (axes)
    # 6 points at r=0.4 (offsets)
    # 1 point at r=0.6 (axis) ?? No, that breaks symmetry unless 3.
    # 11 = 1 + 3 + 3 + 3 + 1? No.
    # 11 = 1 + 3 + 3 + 4?
    
    # Let's use a known good packing:
    # 1 center
    # 3 points near vertices
    # 6 points near edge midpoints
    # 1 point... where?
    # Maybe the optimal isn't perfectly 3-fold symmetric for N=11?
    # But it's a good starting point.
    
    # Let's generate a random symmetric-like set to be safe, then let SA fix it.
    # We will generate 3 "base" points, then rotate them to get 9, plus center, plus 1 random?
    # Actually, let's just generate a good random init and use the SA to find the structure.
    
    # Better Init: Hexagonal lattice
    # Generate a hex grid, pick the 11 points closest to the centroid that fit in the triangle.
    pts = []
    s = 0.2
    for k in range(-5, 6):
        for l in range(-5, 6):
            # Hex grid to Cartesian
            x = s * (k + l/2)
            y = s * (l * np.sqrt(3)/2)
            # Shift to center of triangle
            cx, cy = 0.5, H/3
            x += cx
            y += cy
            # Convert to barycentric
            b = cart_to_bary(np.array([[x, y]]))
            u, v = b[0]
            if u > 0 and v > 0 and u+v < 1:
                pts.append(b[0])
                
    if len(pts) >= N:
        # Sort by distance to center
        center = np.array([1/3, 1/3])
        dists = np.linalg.norm(np.array(pts) - center, axis=1)
        idx = np.argsort(dists)[:N]
        bary = np.array(pts)[idx]
        return bary
        
    return None

def simulated_annealing(bary_init, time_limit=10.0):
    """
    Perform simulated annealing on the barycentric coordinates.
    """
    start_time = time.time()
    current_bary = bary_init.copy()
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    T = 0.01
    cooling = 0.99
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit and T > 1e-6:
        # Propose move
        # Move a random point by a random small amount
        i = rng.integers(0, N)
        delta = rng.normal(0, T, 2)
        
        new_bary = current_bary.copy()
        new_bary[i] += delta
        new_bary = clamp_bary(new_bary)
        
        new_val = -hard_min_obj(new_bary.flatten())
        
        delta_val = new_val - current_val
        
        if delta_val > 0 or rng.random() < np.exp(delta_val / T):
            current_bary = new_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
                
        T *= cooling
        
    return best_bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using gradient-based methods.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Get a good initial guess from hexagonal grid
    hex_init = generate_symmetric_init()
    if hex_init is not None:
        update_best(hex_init)
    else:
        # Fallback random
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        update_best(np.column_stack([u, v]))

    # 2. Simulated Annealing for global exploration
    # SA is good for escaping local minima in non-convex landscapes.
    if best_bary is not None and time.time() < deadline - 30:
        sa_result = simulated_annealing(best_bary, time_limit=20.0)
        update_best(sa_result)

    # 3. Local refinement of the best candidate found by SA
    if best_bary is not None and time.time() < deadline - 15:
        refined = refine_local(best_bary, time_limit=10.0)
        update_best(refined)

    # 4. If time permits, run SA again from a different random start
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        rand_init = np.column_stack([u, v])
        
        sa_result2 = simulated_annealing(rand_init, time_limit=15.0)
        update_best(sa_result2)
        
        if time.time() < deadline - 10:
            refined2 = refine_local(best_bary, time_limit=8.0)
            update_best(refined2)

    # 5. Final high-precision polish on the absolute best
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    if best_bary is None:
        # Fallback to random if everything failed
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
