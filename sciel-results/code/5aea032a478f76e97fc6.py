"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Kinetic Annealing with Topological Collision Resolution +
Multi-restart Nelder-Mead + Barycentric soft-min L-BFGS-B polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIPLETS = len(_TRIPLETS)

DEADLINE = 145  # Safety margin from 150s limit


def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])


def _all_triangle_areas_cart(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas


def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points."""
    return np.min(_all_triangle_areas_cart(pts))


def _soft_min_area_cart(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas_cart(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])


def _kinetic_anneal_step(bary, pts, T, rng):
    """
    Kinetic Annealing step with topological collision resolution.
    
    1. Find the minimum-area triangle (the "collision").
    2. Compute repulsive forces that push apart the vertices of that triangle.
    3. Add random thermal noise scaled by temperature T.
    4. Project back to valid barycentric region.
    
    Returns new barycentric coordinates.
    """
    areas = _all_triangle_areas_cart(pts)
    min_idx = np.argmin(areas)
    min_tri = _TRIPLETS[min_idx]
    
    # Compute gradient-like push for the min triangle vertices
    # For triangle (i, j, k), to increase area we push i away from line jk, etc.
    pi = pts[min_tri[0]]
    pj = pts[min_tri[1]]
    pk = pts[min_tri[2]]
    
    # Area gradient w.r.t. each vertex
    # d(area)/d(pi) = 0.5 * cross(pj - pi, pk - pi) direction perpendicular to edge jk
    # More precisely: to increase area, move each vertex away from the opposite edge
    
    # Normal to edge jk (pointing away from pi)
    edge_jk = pk - pj
    normal_jk = np.array([-edge_jk[1], edge_jk[0]])
    # Ensure normal points away from pi
    to_pi = pi - pj
    if np.dot(normal_jk, to_pi) > 0:
        normal_jk = -normal_jk
    # Normalize
    norm_jk = np.linalg.norm(normal_jk)
    if norm_jk > 1e-12:
        normal_jk /= norm_jk
    
    # Normal to edge ik (pointing away from pj)
    edge_ik = pk - pi
    normal_ik = np.array([-edge_ik[1], edge_ik[0]])
    to_pj = pj - pi
    if np.dot(normal_ik, to_pj) > 0:
        normal_ik = -normal_ik
    norm_ik = np.linalg.norm(normal_ik)
    if norm_ik > 1e-12:
        normal_ik /= norm_ik
    
    # Normal to edge ij (pointing away from pk)
    edge_ij = pj - pi
    normal_ij = np.array([-edge_ij[1], edge_ij[0]])
    to_pk = pk - pi
    if np.dot(normal_ij, to_pk) > 0:
        normal_ij = -normal_ij
    norm_ij = np.linalg.norm(normal_ij)
    if norm_ij > 1e-12:
        normal_ij /= norm_ij
    
    # Push strength inversely proportional to area (stronger push for smaller triangles)
    min_area = areas[min_idx]
    push_strength = 0.02 / (min_area + 1e-6)
    
    # Apply pushes to the three vertices of the min triangle
    new_bary = bary.copy()
    cart_new = pts.copy()
    
    cart_new[min_tri[0]] += push_strength * normal_jk
    cart_new[min_tri[1]] += push_strength * normal_ik
    cart_new[min_tri[2]] += push_strength * normal_ij
    
    # Also add small repulsive forces from all other close pairs
    # Find the 5 smallest triangles and push their vertices apart
    sorted_indices = np.argsort(areas)[:5]
    for idx in sorted_indices:
        tri = _TRIPLETS[idx]
        a = areas[idx]
        strength = 0.005 / (a + 1e-6)
        
        p_a = pts[tri[0]]
        p_b = pts[tri[1]]
        p_c = pts[tri[2]]
        
        # Push each vertex away from the opposite edge
        e_bc = p_c - p_b
        n_bc = np.array([-e_bc[1], e_bc[0]])
        if np.dot(n_bc, p_a - p_b) > 0:
            n_bc = -n_bc
        nn = np.linalg.norm(n_bc)
        if nn > 1e-12:
            n_bc /= nn
        cart_new[tri[0]] += strength * n_bc
        
        e_ac = p_c - p_a
        n_ac = np.array([-e_ac[1], e_ac[0]])
        if np.dot(n_ac, p_b - p_a) > 0:
            n_ac = -n_ac
        nn = np.linalg.norm(n_ac)
        if nn > 1e-12:
            n_ac /= nn
        cart_new[tri[1]] += strength * n_ac
        
        e_ab = p_b - p_a
        n_ab = np.array([-e_ab[1], e_ab[0]])
        if np.dot(n_ab, p_c - p_a) > 0:
            n_ab = -n_ab
        nn = np.linalg.norm(n_ab)
        if nn > 1e-12:
            n_ab /= nn
        cart_new[tri[2]] += strength * n_ab
    
    # Convert back to barycentric
    new_bary = _cart_to_bary(cart_new)
    
    # Add thermal noise
    noise_scale = T * 0.03
    delta = rng.normal(0, noise_scale, size=(N, 2))
    new_bary += delta
    
    # Project to valid region
    new_bary = _project_bary_to_valid(new_bary)
    
    return new_bary


def _snap_to_symmetry(bary, rng):
    """
    Periodic snap-to-symmetry heuristic.
    With some probability, reflect points across the triangle's axes of symmetry
    to explore symmetric configurations.
    """
    if rng.random() > 0.3:
        return bary
    
    # Reflect across the vertical axis of symmetry (x = 0.5)
    # In barycentric: u' = 1 - u - v, v' = v (reflect V0 <-> V1)
    # Actually, let's do a simpler symmetry: swap two random points' roles
    
    # Try reflecting across the median from V2 to midpoint of V0-V1
    # This swaps V0 and V1 in barycentric: (u, v) -> (v, u)
    new_bary = bary.copy()
    
    # With some probability, apply a 120-degree rotation to a random subset
    # 120-degree rotation in barycentric: (u, v) -> (1-v, u)
    # Let's verify: center is (1/3, 1/3, 1/3) in full barycentric
    # Rotation: (u, v, w) -> (w, u, v) where w = 1-u-v
    # So new_u = w = 1-u-v, new_v = u, new_w = v
    # In our 2D barycentric (u, v): new_u = 1-u-v, new_v = u
    
    # Apply rotation to 3-4 random points
    n_rotate = rng.integers(3, 5)
    indices = rng.choice(N, size=n_rotate, replace=False)
    for i in indices:
        u = bary[i, 0]
        v = bary[i, 1]
        new_bary[i, 0] = 1.0 - u - v
        new_bary[i, 1] = u
    
    new_bary = _project_bary_to_valid(new_bary)
    return new_bary


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial configurations
    init_bary_list = []
    
    # Base symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    init_bary_list.append(base_sym)

    # Generate random initial configurations
    for _ in range(40):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        if np.any(mask):
            scale = 1.0 / (u[mask] + v[mask])
            u[mask] *= scale
            v[mask] *= scale
        init_bary_list.append(np.column_stack([u, v]))

    # Add some "near-boundary" configurations
    for _ in range(15):
        u = rng.random(N)
        v = rng.random(N)
        for i in range(N):
            if rng.random() < 0.3:
                edge = rng.integers(0, 3)
                if edge == 0:
                    u[i] = 0
                elif edge == 1:
                    v[i] = 0
                else:
                    s = u[i] + v[i]
                    if s > 0:
                        u[i] /= s
                        v[i] /= s
                    else:
                        u[i] = 0.5
                        v[i] = 0.5
        init_bary_list.append(np.column_stack([u, v]))

    # Lattice-based initial configs
    for _ in range(10):
        bary = np.zeros((N, 2))
        # Place points on a triangular lattice
        for i in range(N):
            row = i // 4
            col = i % 4
            u = (col + 0.5) / 4.0
            v = (row * 0.5 + 0.25) / 3.0
            if u + v > 1:
                u *= 0.5
                v *= 0.5
            bary[i] = [u, v]
        init_bary_list.append(bary)

    # Optimization Loop
    for bary0 in init_bary_list:
        if not _elapsed_ok(25):
            break
        
        current_bary = bary0.copy()
        current_bary = _project_bary_to_valid(current_bary)
        _update_best(current_bary)
        
        # Phase 1: Kinetic Annealing
        if _elapsed_ok(20):
            T_start = 0.1
            T_end = 0.001
            n_kinetic_iter = 3000
            
            for it in range(n_kinetic_iter):
                if not _elapsed_ok(5):
                    break
                # Temperature schedule
                T = T_start * (T_end / T_start) ** (it / n_kinetic_iter)
                
                cart = _bary_to_cart(current_bary)
                new_bary = _kinetic_anneal_step(current_bary, cart, T, rng)
                
                # Periodic snap-to-symmetry
                if it % 500 == 0:
                    new_bary = _snap_to_symmetry(new_bary, rng)
                
                new_bary = _project_bary_to_valid(new_bary)
                _update_best(new_bary)
                
                # Accept/reject: always accept if better, else accept with prob exp(-delta/T)
                new_cart = _bary_to_cart(new_bary)
                new_area = _min_triangle_area_cart(new_cart)
                cur_area = _min_triangle_area_cart(cart)
                
                delta_f = new_area - cur_area
                if delta_f > 0 or rng.random() < np.exp(delta_f / max(T, 1e-8)):
                    current_bary = new_bary

        # Phase 2: Nelder-Mead polish on hard minimum
        if _elapsed_ok(10):
            nm_start = best_bary.copy() if best_bary is not None else bary0.copy()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass

        # Phase 3: L-BFGS-B with soft min for smooth optimization
        if _elapsed_ok(12):
            for beta in [200.0, 1000.0, 5000.0]:
                if not _elapsed_ok(5):
                    break
                
                def objective(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, beta)
                
                x0 = (best_bary if best_bary is not None else bary0).flatten()
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                    current_bary = res.x.reshape(N, 2)
                    current_bary = _project_bary_to_valid(current_bary)
                    _update_best(current_bary)
                except Exception:
                    pass

    # Final polish on the absolute best found
    if best_bary is not None and _elapsed_ok(10):
        def final_nm_obj(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_final = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14,
                                          'adaptive': True})
            final_bary = res_final.x.reshape(N, 2)
            final_bary = _project_bary_to_valid(final_bary)
            _update_best(final_bary)
        except Exception:
            pass
        
        # Second polish with even tighter tolerance if time permits
        if _elapsed_ok(6):
            try:
                res_final2 = minimize(final_nm_obj, best_bary.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-16,
                                               'adaptive': True})
                final_bary2 = res_final2.x.reshape(N, 2)
                final_bary2 = _project_bary_to_valid(final_bary2)
                _update_best(final_bary2)
            except Exception:
                pass

    # Multi-restart Nelder-Mead from perturbed versions of the best
    if best_bary is not None and _elapsed_ok(15):
        for restart in range(5):
            if not _elapsed_ok(5):
                break
            
            # Perturb the best solution
            perturbed = best_bary.copy()
            perturbation = rng.normal(0, 0.02, size=(N, 2))
            perturbed += perturbation
            perturbed = _project_bary_to_valid(perturbed)
            
            def nm_restart_obj(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_r = minimize(nm_restart_obj, perturbed.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13,
                                          'adaptive': True})
                r_bary = res_r.x.reshape(N, 2)
                r_bary = _project_bary_to_valid(r_bary)
                _update_best(r_bary)
            except Exception:
                pass

    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart
