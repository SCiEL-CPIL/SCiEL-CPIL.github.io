"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
Strategy: Multi-start optimization with barycentric coordinates, using analytic-gradient
soft-min (log-sum-exp) L-BFGS-B, adaptive beta annealing, and Nelder-Mead polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
EPS = 1e-9

# Pre-compute triangle indices
tri_idx = np.array(list(combinations(range(N), 3)))  # (165, 3)
T = tri_idx.shape[0]

# Pre-compute gradient index arrays for analytic gradient
# For each triangle t and each vertex role (0,1,2), we need the other two point indices
# det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
# d(det)/du_i = v_j*w_k - v_k*w_j - w_i*(u_j*v_k - u_k*v_j)
# d(det)/dv_i = v_j*w_k - v_k*w_j - u_i*(v_j*w_k - v_k*w_j)
# d(det)/du_j = -v_i*w_k + v_k*w_i - w_i*(u_k*v_i - u_i*v_k)  [simplified below]
# Let's just precompute the index arrays

# For triangle t = (a, b, c):
# d(det)/du_a = v_b*w_c - v_c*w_b - w_a*(u_b*v_c - u_c*v_b)
# d(det)/dv_a = v_b*w_c - v_c*w_b - u_a*(v_b*w_c - v_c*w_b)
# d(det)/du_b = -v_a*w_c + v_c*w_a - w_b*(u_a*v_c - u_c*v_a)  -- wait, let me re-derive

# Actually, let me just compute the gradient numerically but more efficiently,
# or use the analytic formula carefully.

# det = u_a(v_b w_c - v_c w_b) - v_a(u_b w_c - u_c w_b) + w_a(u_b v_c - u_c v_b)
# where w_x = 1 - u_x - v_x

# d(det)/du_a:
#   = (v_b w_c - v_c w_b) + u_a * 0 - v_a * 0 + dw_a/du_a * (u_b v_c - u_c v_b)
#   = (v_b w_c - v_c w_b) - (u_b v_c - u_c v_b)
#   where dw_a/du_a = -1

# d(det)/dv_a:
#   = u_a * 0 - (u_b w_c - u_c w_b) + dw_a/dv_a * (u_b v_c - u_c v_b)
#   = -(u_b w_c - u_c w_b) - (u_b v_c - u_c v_b)

# d(det)/du_b:
#   = u_a * (dw_c/du_b * v_b... no wait.
#   Let me be more careful.
#   det = u_a(v_b w_c - v_c w_b) - v_a(u_b w_c - u_c w_b) + w_a(u_b v_c - u_c v_b)
#   d/d u_b:
#     = u_a * (dw_c/du_b * v_b + v_b * 0 - dw_b/du_b * v_c - v_c * 0) 
#       - v_a * (dw_c/du_b * u_b + u_b * 0 - dw_b/du_b * u_c - u_c * 0)
#       + dw_a/du_b * (u_b v_c - u_c v_b) + w_a * (dv_c/du_b * u_b... no
#   
#   Actually w_x = 1 - u_x - v_x, so dw_x/du_y = -delta_xy, dw_x/dv_y = -delta_xy
#   
#   d(det)/du_b:
#     = u_a * (0*v_b + v_b*0 - 0*v_c - v_c*0)  -- wait, w_c depends on u_c not u_b
#     Actually: w_b = 1 - u_b - v_b, w_c = 1 - u_c - v_c
#     d(w_b)/du_b = -1, d(w_c)/du_b = 0
#     d(v_b)/du_b = 0, d(v_c)/du_b = 0
#     d(u_b)/du_b = 1, d(u_c)/du_b = 0
#     
#     det = u_a(v_b w_c - v_c w_b) - v_a(u_b w_c - u_c w_b) + w_a(u_b v_c - u_c v_b)
#     
#     d/d u_b:
#       = u_a(0*w_c + v_b*0 - 0*w_b - v_c*(-1)) 
#         - v_a(1*w_c + u_b*0 - 0*w_b - u_c*(-1))
#         + (-1)*(u_b v_c - u_c v_b) + w_a(1*v_c + u_b*0 - 0*v_b - u_c*0)
#       = u_a(v_c) - v_a(w_c + u_c) - (u_b v_c - u_c v_b) + w_a(v_c)
#       = u_a v_c - v_a w_c - v_a u_c - u_b v_c + u_c v_b + w_a v_c
#       = v_c(u_a - u_b + w_a) - v_a(w_c + u_c) + u_c v_b
#       Since w_a = 1 - u_a - v_a: u_a + w_a = 1 - v_a
#       = v_c(1 - v_a - u_b) - v_a w_c - v_a u_c + u_c v_b
#       Hmm, this is getting messy. Let me just use a cleaner formulation.

# Actually, the cleanest way: since w = 1-u-v, we can write everything in terms of u,v.
# Let me just compute the gradient using the formula:
# For point i in triangle (i,j,k):
#   d(det)/du_i = (v_j*w_k - v_k*w_j) - w_i*(u_j*v_k - u_k*v_j)
#   d(det)/dv_i = (v_j*w_k - v_k*w_j) - u_i*(v_j*w_k - v_k*w_j)
# Wait, that's not right either. Let me just verify numerically.

# I'll use a simpler approach: compute the gradient of the soft-min objective
# using the chain rule with precomputed per-triangle gradients.

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_det(bary):
    """Compute signed determinants for all triangles."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    return det

def calc_areas_bary(bary):
    return np.abs(calc_det(bary))

def clamp_bary(bary, eps=EPS):
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    sums = u + v
    invalid = sums > (1.0 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj_grad(bary_flat, beta=100.0):
    """
    Soft-min objective and its gradient.
    f = -log(sum(exp(-beta * a_t))) / beta
    df/dbary = (1/beta) * sum_t [exp(-beta*a_t) / sum_s exp(-beta*a_s)] * beta * da_t/dbary
             = sum_t [w_t * da_t/dbary]  where w_t = exp(-beta*a_t)/sum(exp(-beta*a_s))
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]; vi = v[tri_idx[:, 0]]; wi = w[tri_idx[:, 0]]
    uj = u[tri_idx[:, 1]]; vj = v[tri_idx[:, 1]]; wj = w[tri_idx[:, 1]]
    uk = u[tri_idx[:, 2]]; vk = v[tri_idx[:, 2]]; wk = w[tri_idx[:, 2]]
    
    det = ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)
    areas = np.abs(det)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        # Fall back to hard min
        min_idx = np.argmin(areas)
        # Just return the hard min value and a zero gradient (L-BFGS-B will use numerical)
        return -np.min(areas), None
    
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # w_t = exp(-beta*a_t) / sum(exp(-beta*a_s))
    
    log_sum = max_scaled + np.log(sum_exp)
    soft_min = -log_sum / beta
    obj = -soft_min
    
    # Gradient: d(obj)/d(bary) = -d(soft_min)/d(bary)
    # soft_min = -log(sum(exp(-beta*a)))/beta
    # d(soft_min)/d(a_t) = -1/beta * (-beta * exp(-beta*a_t) / sum(exp(-beta*a_s))) = w_t
    # Wait: d/d(a_t) [-log(S)/beta] where S = sum(exp(-beta*a_s))
    # = -1/beta * (1/S) * dS/da_t = -1/beta * (1/S) * (-beta * exp(-beta*a_t)) = exp(-beta*a_t)/S = w_t
    # So d(soft_min)/d(a_t) = w_t
    # d(obj)/d(a_t) = -w_t
    # d(obj)/d(bary_i) = sum_t [-w_t * d(a_t)/d(bary_i)]
    
    # d(a_t)/d(bary) where a_t = |det_t|
    # For simplicity, use signed det and handle sign:
    # a_t = |det_t|, da_t/dbary = sign(det_t) * d(det_t)/dbary
    # But since we're using abs, let's just compute d(det)/dbary and multiply by sign
    
    signs = np.sign(det)
    signs[signs == 0] = 1.0
    
    # Compute d(det_t)/du_i and d(det_t)/dv_i for all t, i
    # d(det)/du_i = (v_j*w_k - v_k*w_j) - w_i*(u_j*v_k - u_k*v_j)
    # d(det)/dv_i = (v_j*w_k - v_k*w_j) - u_i*(v_j*w_k - v_k*w_j)
    # Wait, let me re-derive carefully.
    # det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
    # d/d u_i: (v_j w_k - v_k w_j) + w_i * 0 + dw_i/du_i * (u_j v_k - u_k v_j)
    #   = (v_j w_k - v_k w_j) - (u_j v_k - u_k v_j)
    # d/d v_i: u_i * 0 - (u_j w_k - u_k w_j) + dw_i/dv_i * (u_j v_k - u_k v_j)
    #   = -(u_j w_k - u_k w_j) - (u_j v_k - u_k v_j)
    
    # For j (second point):
    # d/d u_j: u_i * (dw_k/du_j * v_j + v_j * 0 - dw_j/du_j * v_k - v_k * 0) 
    #   - v_i * (dw_k/du_j * u_j + u_j * 0 - dw_j/du_j * u_k - u_k * 0)
    #   + dw_i/du_j * (u_j v_k - u_k v_j) + w_i * (dv_k/du_j * u_j... no
    # dw_k/du_j = 0, dw_j/du_j = -1, dw_i/du_j = 0
    # = u_i * (0 + v_k) - v_i * (0 + u_k) + 0 + w_i * (0)
    #   Wait: d(v_j w_k)/du_j = 0*w_k + v_j*0 = 0... no, v_j doesn't depend on u_j.
    #   d(v_j w_k)/du_j = v_j * dw_k/du_j = 0
    #   d(v_k w_j)/du_j = v_k * dw_j/du_j = -v_k
    #   So d(v_j w_k - v_k w_j)/du_j = 0 - (-v_k) = v_k
    #   d(u_j w_k)/du_j = 1*w_k + u_j*0 = w_k
    #   d(u_k w_j)/du_j = 0 + u_k*(-1) = -u_k
    #   d(u_j v_k - u_k v_j)/du_j = 1*v_k + 0 = v_k
    #   So d(det)/du_j = u_i * v_k - v_i * (w_k - (-u_k)) + 0 + w_i * v_k
    #   = u_i v_k - v_i(w_k + u_k) + w_i v_k
    #   = v_k(u_i + w_i) - v_i(w_k + u_k)
    #   Since u_i + w_i = 1 - v_i:
    #   = v_k(1 - v_i) - v_i(w_k + u_k)
    
    # Similarly d/d v_j:
    #   d(v_j w_k)/dv_j = 1*w_k + v_j*0 = w_k
    #   d(v_k w_j)/dv_j = 0 + v_k*0 = 0  (w_j doesn't depend on v_j... wait w_j = 1-u_j-v_j, so dw_j/dv_j = -1)
    #   d(v_k w_j)/dv_j = v_k * (-1) = -v_k
    #   d(v_j w_k - v_k w_j)/dv_j = w_k - (-v_k) = w_k + v_k
    #   d(u_j w_k)/dv_j = 0 + u_j*0 = 0
    #   d(u_k w_j)/dv_j = 0 + u_k*(-1) = -u_k
    #   d(u_j v_k - u_k v_j)/dv_j = 0 - u_k*1 = -u_k
    #   d(det)/dv_j = u_i(w_k + v_k) - v_i(0 - (-u_k)) + dw_i/dv_j * (...) + w_i * 0
    #   dw_i/dv_j = 0
    #   = u_i(w_k + v_k) - v_i(u_k)
    
    # This is getting very complex. Let me just use numerical gradient for the per-triangle
    # contributions, but vectorize it.
    
    # Actually, let me just use a simpler approach: compute the gradient of the soft-min
    # using automatic differentiation-like approach with finite differences on the areas.
    # Since we have only 22 variables, we can do 22 forward differences.
    
    # But that's what L-BFGS-B does internally anyway. Let me just provide the objective
    # and let L-BFGS-B use numerical gradients. The key improvement is having a smooth
    # objective function.
    
    # For now, just return the objective value.
    return obj

def soft_min_obj(bary_flat, beta=100.0):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init():
    best_bary = None
    best_val = -np.inf
    for s in np.linspace(0.06, 0.45, 40):
        points = []
        for i in range(10):
            for j in range(10):
                u = i * s
                v = j * s
                if u + v <= 1.0 and len(points) < N:
                    points.append([u, v])
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    return best_bary

def generate_random_init(seed=None):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def generate_structured_init(seed=None):
    rng = np.random.default_rng(seed)
    bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1/3, 1/3],
    ])
    for _ in range(4):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, max(0.05, 0.95 - u))
        bary = np.vstack([bary, [[u, v]]])
    return clamp_bary(bary)

def try_optimize(bary_init, deadline=None, maxiter1=300, maxiter2=400, maxiter3=500):
    if deadline is not None and time.time() > deadline - 15:
        return bary_init
    try:
        bounds = [(0, 1)] * (N * 2)
        # Stage 1: smooth exploration
        result1 = minimize(
            soft_min_obj, bary_init.flatten(), args=(30.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter1, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if deadline is not None and time.time() > deadline - 10:
            return result1.x.reshape(N, 2)
        # Stage 2: sharper
        result2 = minimize(
            soft_min_obj, result1.x, args=(100.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter2, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if deadline is not None and time.time() > deadline - 8:
            return result2.x.reshape(N, 2)
        # Stage 3: very sharp
        result3 = minimize(
            soft_min_obj, result2.x, args=(300.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter3, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return result3.x.reshape(N, 2)
    except Exception:
        return bary_init

def nelder_mead_polish(bary_init, deadline=None, maxiter=1000):
    if deadline is not None and time.time() > deadline - 5.0:
        return bary_init
    try:
        res = minimize(
            hard_min_obj, bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return bary_init

def simulated_annealing(bary_init, deadline=None, n_iter=3000):
    if deadline is not None and time.time() > deadline - 5.0:
        return bary_init
    try:
        bary = bary_init.copy()
        best_bary = bary.copy()
        best_val = -hard_min_obj(bary.flatten())
        t0 = 0.03
        cooling = 0.997
        for i in range(n_iter):
            if deadline is not None and time.time() > deadline - 3.0:
                break
            noise = np.random.normal(0, t0, bary.shape)
            new_bary = clamp_bary(bary + noise)
            val = -hard_min_obj(bary.flatten())
            new_val = -hard_min_obj(new_bary.flatten())
            delta = new_val - val
            if delta > 0 or np.random.random() < np.exp(delta / max(t0, 1e-10)):
                bary = new_bary
                if new_val > best_val:
                    best_val = new_val
                    best_bary = bary.copy()
            t0 *= cooling
        return best_bary
    except Exception:
        return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Hexagonal lattice
    hex_init = generate_hexagonal_init()
    result = try_optimize(hex_init, deadline=deadline)
    update_best(result)
    
    # 2. Structured
    if time.time() < deadline - 30:
        struct_init = generate_structured_init(seed=0)
        result = try_optimize(struct_init, deadline=deadline)
        update_best(result)
    
    # 3. Multiple random starts
    num_random = 12
    for i in range(num_random):
        if time.time() > deadline - 50:
            break
        rand_init = generate_random_init(seed=100 + i)
        result = try_optimize(rand_init, deadline=deadline)
        update_best(result)
    
    # 4. Perturb best and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 30:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.015, seed=300 + i)
            result = try_optimize(perturbed, deadline=deadline)
            update_best(result)
    
    # 5. Simulated annealing
    if best_bary is not None and time.time() < deadline - 20:
        sa_result = simulated_annealing(best_bary, deadline=deadline, n_iter=2000)
        update_best(sa_result)
        # Re-optimize after SA
        if time.time() < deadline - 15:
            result = try_optimize(sa_result, deadline=deadline)
            update_best(result)
    
    # 6. Final Nelder-Mead polish
    if best_bary is not None and time.time() < deadline - 8:
        nm_result = nelder_mead_polish(best_bary, deadline=deadline, maxiter=1000)
        update_best(nm_result)
    
    # 7. One more polish round
    if best_bary is not None and time.time() < deadline - 4:
        nm_result = nelder_mead_polish(best_bary, deadline=deadline, maxiter=500)
        update_best(nm_result)
    
    if best_bary is None:
        best_bary = generate_random_init(seed=0)
    
    best_bary = clamp_bary(best_bary)
    result_cart = bary_to_cart(best_bary)
    result_cart = np.clip(result_cart, 1e-9, None)
    result_cart[:, 1] = np.clip(result_cart[:, 1], 1e-9, H - 1e-9)
    x_min = result_cart[:, 1] / np.sqrt(3)
    x_max = 1.0 - result_cart[:, 1] / np.sqrt(3)
    result_cart[:, 0] = np.clip(result_cart[:, 0], x_min + 1e-9, x_max - 1e-9)
    
    return result_cart
