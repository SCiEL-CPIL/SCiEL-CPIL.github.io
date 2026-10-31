import numpy as np
import time
import math
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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
    # Inverse of bary_to_cart
    # x = v + 0.5*(1-u-v) = 0.5 - 0.5u + 0.5v
    # y = (1-u-v)*H
    # => v = x - 0.5 + 0.5u
    # => y/H = 1 - u - (x - 0.5 + 0.5u) = 1.5 - x - 1.5u
    # => 1.5u = 1.5 - x - y/H
    u = (1.5 - cart[:, 0] - cart[:, 1] / H) / 1.5
    v = cart[:, 0] - 0.5 + 0.5 * u
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

def compute_gradients(bary, areas, min_area, alpha=10.0):
    """
    Compute the gradient of the 'soft minimum' of areas.
    We want to maximize the minimum area.
    Let f(x) = min_i A_i(x).
    We approximate the gradient using a smooth minimum:
    F(x) = -1/alpha * log(sum(exp(-alpha * A_i)))
    grad F = (sum(exp(-alpha * A_i) * grad A_i) / sum(exp(-alpha * A_i)))
    
    Since we want to maximize min area, we move in the direction of -grad F (which is +grad min area approx).
    Actually, let's just compute the weighted average of gradients of the smallest triangles.
    """
    # We work in Cartesian for gradient calculation as it's more intuitive for geometry, 
    # but we can do it in Barycentric too. Let's stick to Barycentric for consistency with state.
    
    # The area of triangle (i,j,k) in barycentric coords is |det|.
    # Derivative of det w.r.t bary coords of point p:
    # If p is i: dA/du_i = v_j w_k - v_k w_j ... etc.
    
    # To avoid complex manual differentiation, we can use finite differences 
    # or simply use the fact that for a fixed pair (j,k), the area is linear in the position of i.
    # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
    
    # Let's use Cartesian gradients for clarity and then map back?
    # Or just use finite differences on the barycentric coords? 
    # Finite differences are robust but slow. 
    # However, N=11 is small.
    
    # Let's implement the analytical gradient in Barycentric coordinates.
    # Area A_ijk = | det([u_i, v_i, w_i], [u_j, v_j, w_j], [u_k, v_k, w_k]) |
    # Let D = det. A = |D|.
    # dA/du_i = sign(D) * dD/du_i
    # dD/du_i = v_j w_k - v_k w_j
    # dD/dv_i = u_k w_j - u_j w_k
    # Note: w_i = 1 - u_i - v_i, so dD/dw_i is not independent.
    # But we are optimizing u_i, v_i.
    
    grad_u = np.zeros(N)
    grad_v = np.zeros(N)
    
    # Weights for soft min
    # To avoid underflow, subtract min area
    exp_args = -alpha * (areas - min_area)
    weights = np.exp(exp_args)
    weights /= np.sum(weights)
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # We need the sign of the determinant to handle the absolute value.
    # D = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
    
    # Let's compute D for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    D = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    signs = np.sign(D)
    # If D is 0, sign is 0, which is fine (gradient 0)
    
    # Gradient contribution of triangle m to point i (which is tri_idx[m, 0])
    # dD/du_i = v_j w_k - v_k w_j
    # dD/dv_i = u_k w_j - u_j w_k
    
    # We need to accumulate these weighted by weights[m] * signs[m]
    
    # This is a sparse accumulation. 
    # For each triangle m:
    #   i = tri_idx[m, 0]
    #   j = tri_idx[m, 1]
    #   k = tri_idx[m, 2]
    #   w_m = weights[m] * signs[m]
    #   grad_u[i] += w_m * (vj[m] * wk[m] - vk[m] * wj[m])
    #   grad_v[i] += w_m * (uk[m] * wj[m] - uj[m] * wk[m])
    #
    #   Similarly for j and k?
    #   dD/du_j = v_k w_i - v_i w_k
    #   dD/dv_j = u_i w_k - u_k w_i
    #
    #   dD/du_k = v_i w_j - v_j w_i
    #   dD/dv_k = u_j w_i - u_i w_j
    
    # Let's vectorize this accumulation.
    # Since N is small, a loop over triangles is fast enough.
    
    for m in range(len(tri_indices)):
        if weights[m] < 1e-12: continue
        
        i, j, k = tri_idx[m]
        wm = weights[m] * signs[m]
        
        # Point i
        dDu_i = vj[m] * wk[m] - vk[m] * wj[m]
        dDv_i = uk[m] * wj[m] - uj[m] * wk[m]
        grad_u[i] += wm * dDu_i
        grad_v[i] += wm * dDv_i
        
        # Point j
        dDu_j = vk[m] * wi[m] - vi[m] * wk[m]
        dDv_j = ui[m] * wk[m] - uk[m] * wi[m]
        grad_u[j] += wm * dDu_j
        grad_v[j] += wm * dDv_j
        
        # Point k
        dDu_k = vi[m] * wj[m] - vj[m] * wi[m]
        dDv_k = uj[m] * wi[m] - ui[m] * wj[m]
        grad_u[k] += wm * dDu_k
        grad_v[k] += wm * dDv_k
        
    return np.column_stack([grad_u, grad_v])

def generate_hexagonal_init(seed=0):
    """
    Generate a hexagonal lattice-like initialization.
    This is often a good starting point for packing problems.
    """
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Manually place 11 points in a hexagonal pattern inside the triangle
    # Center at (1/3, 1/3)
    # Ring 1: 6 points
    # Ring 2: 4 points?
    
    # Let's use a simpler robust init: Grid + Perturbation
    # But the prompt asks for a structural change. 
    # Let's use a "Voronoi" style init by projecting random points to a lattice.
    
    # For now, use a dense grid and pick 11 best spaced points.
    # Or just use the lattice from before but refined.
    
    # Let's try a specific known good configuration for 11 points.
    # Based on literature, 11 points in a triangle often have 1 center, 6 in a hexagon, 4 near vertices?
    
    # Let's just use a random init but with a "repulsion" pre-step.
    bary = rng.random(N)
    bary[:, 1] *= (1 - bary[:, 0])
    
    # Pre-relaxation: simple repulsion
    for _ in range(50):
        for i in range(N):
            for j in range(N):
                if i == j: continue
                # Simple distance based repulsion in barycentric space (approx)
                diff = bary[i] - bary[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-6:
                    bary[i] += rng.normal(0, 0.01, 2)
                else:
                    force = diff / dist_sq
                    bary[i] += 0.001 * force
                    
    bary = clamp_bary(bary)
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate initial candidates
    inits = []
    for i in range(10):
        inits.append(generate_hexagonal_init(i))
    
    # Main Loop: Gradient Ascent with Adaptive Step Size
    for init_bary in inits:
        if time.time() > deadline - 15:
            break
            
        bary = init_bary.copy()
        current_val = np.min(calc_areas_bary(bary))
        update_best(bary)
        
        # Adaptive parameters
        step_size = 0.05
        alpha = 5.0 # Sharpness of soft min
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5:
            areas = calc_areas_bary(bary)
            min_area = np.min(areas)
            
            if min_area > best_val:
                best_val = min_area
                best_bary = bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break
            
            # Compute gradient
            grads = compute_gradients(bary, areas, min_area, alpha=alpha)
            
            # Normalize gradient to prevent huge steps
            norm = np.linalg.norm(grads)
            if norm > 1e-9:
                grads = grads / norm
                
            # Move
            new_bary = bary + step_size * grads
            new_bary = clamp_bary(new_bary)
            
            new_areas = calc_areas_bary(new_bary)
            new_min_area = np.min(new_areas)
            
            if new_min_area > min_area:
                bary = new_bary
                # Slightly increase step size if we're making good progress
                step_size = min(step_size * 1.05, 0.1)
                # Increase alpha to focus on the minimum more sharply
                alpha = min(alpha * 1.01, 50.0)
            else:
                # Backtrack
                step_size *= 0.8
                # Decrease alpha to smooth out the landscape
                alpha = max(alpha * 0.95, 1.0)
                
                # If step size is too small, break
                if step_size < 1e-5:
                    break

    # Final Refinement: SLSQP on the best point found
    if best_bary is not None and time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            # Define SLSQP objective and constraints
            def obj(x):
                b = x[:N*2].reshape(N, 2)
                b = clamp_bary(b)
                return -np.min(calc_areas_bary(b))
                
            def con_fun(x):
                b = x[:N*2].reshape(N, 2)
                b = clamp_bary(b)
                u = b[:, 0]
                v = b[:, 1]
                cons = np.zeros(3*N)
                cons[0::3] = u
                cons[1::3] = v
                cons[2::3] = 1.0 - u - v
                return cons
                
            x0 = best_bary.flatten()
            bounds = [(0.0, 1.0)] * (2*N)
            
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': con_fun},
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.x is not None:
                res_bary = res.x[:N*2].reshape(N, 2)
                update_best(res_bary)
        except Exception:
            pass

    # Fallback
    if best_bary is None:
        best_bary = generate_hexagonal_init(0)
        
    cart = bary_to_cart(best_bary)
    return cart

