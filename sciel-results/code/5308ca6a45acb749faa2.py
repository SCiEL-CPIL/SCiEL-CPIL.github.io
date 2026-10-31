"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute all triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
M = len(tri_indices)  # 165

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def clamp_bary(bary, eps=1e-7):
    """Clamp barycentric coordinates to valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.maximum(sums[invalid], eps)
        u[invalid] *= scale
        v[invalid] *= scale
    
    return np.column_stack([u, v])

def signed_areas(bary):
    """Compute signed areas (determinants) of all triangles in barycentric coords.
    Returns array of shape (M,) with signed values."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    return det

def min_abs_area(bary):
    """Compute the minimum absolute triangle area."""
    sa = signed_areas(bary)
    return np.min(np.abs(sa))

def subgradient_step(bary, sa, alpha, sgn, lr):
    """
    One step of projected subgradient descent for the LP formulation.
    
    We want: sgn[m] * sa[m] >= alpha for all m, and maximize alpha.
    Subgradient of max(0, alpha - sgn[m]*sa[m]) w.r.t. bary:
      If sgn[m]*sa[m] < alpha: gradient = -sgn[m] * grad_sa[m]
    
    We compute the sum of violated constraints' gradients and step.
    """
    # grad_sa[m] is a (N, 2) matrix: gradient of sa[m] w.r.t. bary
    # sa[m] = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    # For point i: d(sa)/du_i = vj*wk - vk*wj, d(sa)/dv_i = wk - wj
    # For point j: d(sa)/du_j = -(uj*wk - uk*wj)... let's be careful
    
    # Actually let's compute gradients properly.
    # sa = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    # = ui*vj*wk - ui*vk*wj - vi*uj*wk + vi*uk*wj + wi*uj*vk - wi*uk*vj
    
    # d(sa)/du_i = vj*wk - vk*wj
    # d(sa)/dv_i = wk - wj
    # d(sa)/du_j = ui*wk - vi*wk + wi*vk  = wk*(ui - vi) + wi*vk
    # d(sa)/dv_j = -ui*wj + vi*wj + wi*uk  = wj*(vi - ui) + wi*uk
    # d(sa)/du_k = -ui*wj + vi*wj + wi*vj  = wj*(vi - ui) + wi*vj
    # d(sa)/dv_k = ui*vj - vi*uj - wi*uk  = uj*(ui - vi) - wi*uk  ... let me redo
    
    # Let me use the product rule more carefully.
    # sa = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    
    # d(sa)/du_i = vj*wk - vk*wj
    # d(sa)/dv_i = wk - wj
    # d(sa)/du_j = -vi*wk + wi*vk
    # d(sa)/dv_j = ui*wk - uk*wi  ... wait
    #   -vi*(uj*wk - uk*wj): d/dv_j = -vi*(0 - 0) = 0... no
    #   Let me expand: -vi*uj*wk + vi*uk*wj
    #   d/dv_j: -vi*uj*0 + vi*uk*0... no, v_j appears in wk = 1-ui-vi-vj
    #   This is getting complex. Let me use automatic differentiation via finite differences
    #   or compute the gradient matrix directly.
    
    # Actually, the cleanest way: sa[m] is linear in each bary point's coordinates.
    # For triangle (i,j,k):
    #   sa = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    #   where w = 1 - u - v
    #   
    #   Let me write in terms of u_i, v_i, u_j, v_j, u_k, v_k:
    #   w_i = 1-u_i-v_i, etc.
    #   
    #   d(sa)/du_i = vj*wk - vk*wj
    #   d(sa)/dv_i = wk - wj
    #   d(sa)/du_j = -(vi*wk) + (wi*vk)
    #   d(sa)/dv_j = -(uj*wk) + (uk*wi)  ... let me verify
    #     -vi*(uj*wk - uk*wj) = -vi*uj*wk + vi*uk*wj
    #     d/dv_j: -vi*uj*(d(wk)/dv_j) + vi*uk*(d(wj)/dv_j)
    #     wk = 1-u_k-v_k, so d(wk)/dv_j = 0 (different points)
    #     wj = 1-u_j-v_j, so d(wj)/dv_j = -1
    #     So d/dv_j of -vi*uj*wk + vi*uk*wj = 0 + vi*uk*(-1) = -vi*uk
    #     Wait, that's not right either. v_j only appears in w_j.
    #     
    #   OK let me just do it systematically. Points are distinct, so u_i only appears in w_i, etc.
    #   Actually no: u_i appears in the term ui directly, and in w_i = 1-ui-vi.
    #   But u_i does NOT appear in w_j or w_k (those depend on u_j,v_j and u_k,v_k).
    #   
    #   So:
    #   sa = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    #   
    #   d(sa)/du_i: ui appears in first term (coefficient vj*wk - vk*wj) and in wi (coefficient uj*vk - uk*vj)
    #   d(wi)/du_i = -1
    #   So d(sa)/du_i = (vj*wk - vk*wj) - (uj*vk - uk*vj)
    #   
    #   d(sa)/dv_i: vi appears in second term (coefficient -(uj*wk - uk*wj)) and in wi
    #   d(wi)/dv_i = -1
    #   So d(sa)/dv_i = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
    #   
    #   d(sa)/du_j: uj appears in second term (coefficient -vi*wk) and third term (coefficient wi*vk)
    #   d(sa)/du_j = -vi*wk + wi*vk
    #   
    #   d(sa)/dv_j: vj appears in first term (coefficient ui*wk) and third term (coefficient -wi*uk)
    #   d(sa)/dv_j = ui*wk - wi*uk
    #   
    #   d(sa)/du_k: uk appears in second term (coefficient vi*wj) and third term (coefficient -wi*vj)
    #   d(sa)/du_k = vi*wj - wi*vj
    #   
    #   d(sa)/dv_k: vk appears in first term (coefficient -ui*wj) and third term (coefficient wi*uj)
    #   d(sa)/dv_k = -ui*wj + wi*uj
    
    # Build gradient matrix: shape (M, N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    
    # Gradient of sa[m] w.r.t. each point's (u,v)
    # Point i:
    du_i = (vj*wk - vk*wj) - (uj*vk - uk*vj)
    dv_i = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
    # Point j:
    du_j = -vi*wk + wi*vk
    dv_j = ui*wk - wi*uk
    # Point k:
    du_k = vi*wj - wi*vj
    dv_k = -ui*wj + wi*uj
    
    # Now compute subgradient for violated constraints
    # Constraint: sgn[m] * sa[m] >= alpha
    # Violation: alpha - sgn[m]*sa[m] > 0
    # Subgradient contribution: -sgn[m] * grad_sa[m] (since we're minimizing violation)
    
    violation = alpha - sgn * sa
    mask = violation > 0
    n_violated = np.sum(mask)
    
    if n_violated == 0:
        # All constraints satisfied, try to increase alpha
        return bary, alpha + lr * 0.1, sgn
    
    # Accumulate gradient
    grad = np.zeros((N, 2))
    
    # For violated constraints, add -sgn[m] * grad_sa[m]
    vm = mask
    sg = sgn[vm]
    # du_i for violated
    grad[:, 0] += -sg * du_i[vm]
    grad[:, 1] += -sg * dv_i[vm]
    
    # Add contributions from j
    grad[tri_idx[vm, 1], 0] += -sg * du_j[vm]
    grad[tri_idx[vm, 1], 1] += -sg * dv_j[vm]
    
    # Add contributions from k
    grad[tri_idx[vm, 2], 0] += -sg * du_k[vm]
    grad[tri_idx[vm, 2], 1] += -sg * dv_k[vm]
    
    # Normalize gradient
    norm = np.linalg.norm(grad)
    if norm < 1e-12:
        return bary, alpha, sgn
    
    grad = grad / norm
    
    # Step
    new_bary = bary - lr * grad
    
    # Update alpha: move toward the minimum of sgn*sa
    current_min = np.min(sgn * sa)
    # Increase alpha if we're making progress
    if current_min > alpha:
        new_alpha = alpha + lr * 0.5
    else:
        new_alpha = alpha
    
    # Update signs based on current signed areas
    new_sgn = np.sign(sa)
    new_sgn[new_sgn == 0] = 1.0
    
    return new_bary, new_alpha, new_sgn

def project_to_triangle(bary, eps=1e-7):
    """Project barycentric coordinates to the valid triangle region."""
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Project onto u >= eps, v >= eps, u+v <= 1-eps
    u = np.clip(u, eps, None)
    v = np.clip(v, eps, None)
    
    # Handle u+v constraint
    s = u + v
    too_big = s > (1.0 - eps)
    if np.any(too_big):
        # Scale down
        scale = (1.0 - eps) / s[too_big]
        u[too_big] *= scale
        v[too_big] *= scale
    
    return np.column_stack([u, v])

def solve_lp_subgradient(bary_init, alpha_init, max_iters=5000, lr_init=0.01, deadline=None):
    """Solve the LP feasibility problem using projected subgradient descent."""
    bary = bary_init.copy()
    alpha = alpha_init
    sgn = np.sign(signed_areas(bary))
    sgn[sgn == 0] = 1.0
    
    best_bary = bary.copy()
    best_alpha = -np.inf
    best_min_area = 0.0
    
    lr = lr_init
    patience = 0
    last_improve = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        sa = signed_areas(bary)
        current_min = np.min(np.abs(sa))
        
        if current_min > best_min_area:
            best_min_area = current_min
            best_bary = bary.copy()
            best_alpha = alpha
            patience = 0
        else:
            patience += 1
        
        # Update signs
        sgn = np.sign(sa)
        sgn[sgn == 0] = 1.0
        
        # Subgradient step
        bary, alpha, sgn = subgradient_step(bary, sa, alpha, sgn, lr)
        bary = project_to_triangle(bary)
        
        # Adaptive learning rate
        if patience > 50:
            lr *= 0.95
            if lr < 1e-6:
                break
        
    return best_bary, best_min_area

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.5, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1.0 - 1e-4 and u >= 1e-4 and v >= 1e-4:
                    points.append([u, v])
        
        if len(points) >= N:
            # Pick N points most spread out
            points = np.array(points[:N])
            val = min_abs_area(points)
            if val > best_val:
                best_val = val
                best_bary = points.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_symmetric_init():
    """Generate a symmetric initial configuration based on known Heilbronn structure."""
    # For n=11, a good structure has points on a triangular grid
    # with some perturbations for symmetry
    bary = np.array([
        [0.5, 0.5],           # center
        [0.1, 0.1],           # corners
        [0.9, 0.05],
        [0.05, 0.9],
        [0.3, 0.1],
        [0.7, 0.1],
        [0.1, 0.3],
        [0.9, 0.3],
        [0.3, 0.7],
        [0.7, 0.7],
        [0.5, 0.1],
    ])
    return clamp_bary(bary)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_abs_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False
    
    # Strategy 1: Start with hexagonal lattice
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    # Strategy 2: Start with symmetric configuration
    sym_init = generate_symmetric_init()
    update_best(sym_init)
    
    # Strategy 3: LP subgradient from multiple initializations
    inits = [hex_init, sym_init]
    for i in range(8):
        inits.append(generate_random_init(seed=100 + i))
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        # Try multiple alpha starting values
        for alpha_start in [0.005, 0.01, 0.015, 0.02]:
            if time.time() > deadline - 10:
                break
            
            try:
                result_bary, result_val = solve_lp_subgradient(
                    init, alpha_start, 
                    max_iters=3000, lr_init=0.008,
                    deadline=deadline - 5
                )
                update_best(result_bary)
            except Exception:
                pass
    
    # Strategy 4: Local refinement with finer subgradient
    if best_bary is not None:
        for i in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb best solution
            rng = np.random.default_rng(200 + i)
            perturbed = best_bary.copy()
            perturbed += rng.normal(0, 0.01, perturbed.shape)
            perturbed = clamp_bary(perturbed)
            
            try:
                result_bary, result_val = solve_lp_subgradient(
                    perturbed, best_val * 0.95,
                    max_iters=2000, lr_init=0.005,
                    deadline=deadline - 5
                )
                update_best(result_bary)
            except Exception:
                pass
    
    # Strategy 5: Very fine local search on best solution
    if best_bary is not None and time.time() < deadline - 5:
        for i in range(10):
            if time.time() > deadline - 2:
                break
            
            rng = np.random.default_rng(300 + i)
            perturbed = best_bary.copy()
            perturbed += rng.normal(0, 0.003, perturbed.shape)
            perturbed = clamp_bary(perturbed)
            update_best(perturbed)
    
    # Final refinement: coordinate descent on the best solution
    if best_bary is not None and time.time() < deadline - 3:
        bary_cd = best_bary.copy()
        for coord_iter in range(200):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(N):
                for dim in range(2):
                    # Try small perturbations in each coordinate
                    for delta in [-0.005, -0.002, -0.001, 0.001, 0.002, 0.005]:
                        trial = bary_cd.copy()
                        trial[i, dim] += delta
                        trial = clamp_bary(trial)
                        if min_abs_area(trial) > min_abs_area(bary_cd):
                            bary_cd = trial
                            improved = True
                            break
            if not improved:
                break
        update_best(bary_cd)
    
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    bary_final = clamp_bary(best_bary, eps=1e-6)
    cart = bary_to_cart(bary_final)
    
    # Safety check: ensure points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-8:
            cart[i, 1] = 1e-8
        if y > np.sqrt(3) * x - 1e-8:
            cart[i, 1] = np.sqrt(3) * x - 1e-8
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-8:
            cart[i, 1] = np.sqrt(3) - 1e-8 - np.sqrt(3) * x
    
    return cart
