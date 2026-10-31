import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Electrostatic simulation parameters
SIM_STEPS = 300
SIM_DT = 0.02
SIM_TEMP_INIT = 0.05
SIM_TEMP_DECAY = 0.995

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

def electrostatic_forces(bary, beta=50.0):
    """
    Compute forces on each point using gradient of the log-sum-exp of areas.
    
    The force on point i is derived from:
    F_i = -grad_{p_i} [ -1/beta * log(sum_j exp(-beta * A_j)) ]
         = 1/beta * sum_j [ exp(-beta*A_j) / sum_k exp(-beta*A_k) ] * grad_{p_i} A_j
    
    For triangles involving point i, grad_{p_i} A_j gives the direction
    that increases area A_j.
    
    Returns: (N, 2) array of forces in barycentric coordinates.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Compute all areas
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
    areas = np.abs(det)
    
    # Avoid log(0)
    areas_safe = np.maximum(areas, 1e-15)
    
    # Compute weights for log-sum-exp
    scaled = -beta * areas_safe
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # Compute forces: for each triangle, compute gradient w.r.t. each point
    forces = np.zeros((N, 2))
    
    for t_idx in range(len(tri_idx)):
        i, j, k = tri_idx[t_idx]
        wt = weights[t_idx]
        
        if wt < 1e-10:
            continue
        
        # Area of triangle (i,j,k) in barycentric coords
        # det = u_i(v_j*w_k - v_k*w_j) - v_i(u_j*w_k - u_k*w_j) + w_i(u_j*v_k - u_k*v_j)
        
        # Gradient w.r.t. point i's barycentric coords (u_i, v_i)
        # Note: w_i = 1 - u_i - v_i, so dw_i/du_i = -1, dw_i/dv_i = -1
        
        # d(det)/du_i = (v_j*w_k - v_k*w_j)
        # d(det)/dv_i = -(u_j*w_k - u_k*w_j)
        # d(det)/dw_i = (u_j*v_k - u_k*v_j)
        
        # Since we're using (u,v) as independent variables and w = 1-u-v:
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + w_i * d(u_j*v_k - u_k*v_j)/du_i
        # But u_j, v_j, etc. don't depend on u_i, so:
        # d(det)/du_i = (v_j*w_k - v_k*w_j)
        # d(det)/dv_i = -(u_j*w_k - u_k*w_j)
        
        # Actually, let me recompute properly.
        # det = u_i*(v_j*w_k - v_k*w_j) - v_i*(u_j*w_k - u_k*w_j) + w_i*(u_j*v_k - u_k*v_j)
        # where w_i = 1 - u_i - v_i
        
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + (1-u_i-v_i)*0 - (u_j*v_k - u_k*v_j)
        # Wait, w_i depends on u_i: dw_i/du_i = -1
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + (-1)*(u_j*v_k - u_k*v_j)
        
        # d(det)/dv_i = -(u_j*w_k - u_k*w_j) + (-1)*(u_j*v_k - u_k*v_j)
        
        # For point j:
        # d(det)/du_j = -v_i*(w_k) + w_i*(v_k)  [since dw_j/du_j = -1, and terms with w_j]
        # Let me be more careful:
        
        # det = u_i*v_j*w_k - u_i*v_k*w_j - v_i*u_j*w_k + v_i*u_k*w_j + w_i*u_j*v_k - w_i*u_k*v_j
        
        # d(det)/du_i = v_j*w_k - v_k*w_j
        # d(det)/dv_i = -u_j*w_k + u_k*w_j
        # d(det)/du_j = -v_i*w_k + w_i*v_k
        # d(det)/dv_j = u_i*w_k - u_k*w_i  [wait, let me redo]
        
        # Let me use the fact that in barycentric coords with w=1-u-v:
        # The area formula simplifies. For triangle with vertices at barycentric (u1,v1), (u2,v2), (u3,v3):
        # Area = |det| where det = (u1-v1)(v2-v3) + (u2-v2)(v3-v1) + (u3-v3)(v1-v2)
        # No wait, that's not right either.
        
        # Let me just compute numerically for the gradient.
        # Actually, for efficiency, let me use the analytical gradient.
        
        # In barycentric coordinates, the area of triangle (i,j,k) is:
        # A = |u_i(v_j - v_k) + u_j(v_k - v_i) + u_k(v_i - v_j)|
        # Wait no. Let me use the standard formula.
        
        # The determinant is:
        # |u_i v_i w_i|
        # |u_j v_j w_j| = u_i(v_j*w_k - v_k*w_j) - v_i(u_j*w_k - u_k*w_j) + w_i(u_j*v_k - u_k*v_j)
        # |u_k v_k w_k|
        
        # Since w = 1-u-v, we have:
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + w_i * d(u_j*v_k - u_k*v_j)/du_i - v_i * d(u_j*w_k - u_k*w_j)/du_i
        # But u_j, v_j, u_k, v_k don't depend on u_i.
        # w_j = 1-u_j-v_j doesn't depend on u_i.
        # So: d(det)/du_i = (v_j*w_k - v_k*w_j)
        # And since w_i = 1-u_i-v_i, dw_i/du_i = -1:
        # d(det)/du_i = (v_j*w_k - v_k*w_j) + (-1)*(u_j*v_k - u_k*v_j)
        
        # Similarly:
        # d(det)/dv_i = -(u_j*w_k - u_k*w_j) + (-1)*(u_j*v_k - u_k*v_j)
        
        # For point j:
        # d(det)/du_j = -v_i*w_k + w_i*v_k  [from -v_i*(u_j*w_k - u_k*w_j) term, du_j appears in u_j*w_k]
        # Actually: -v_i*u_j*w_k + v_i*u_k*w_j + w_i*u_j*v_k - w_i*u_k*v_j
        # d/du_j: -v_i*w_k + w_i*v_k
        # d/dv_j: -u_i*w_k + u_k*w_i
        
        # For point k:
        # d(det)/du_k = -u_i*v_k + v_i*u_k  [wait]
        # From u_i*v_j*w_k - u_i*v_k*w_j - v_i*u_j*w_k + v_i*u_k*w_j + w_i*u_j*v_k - w_i*u_k*v_j
        # d/du_k: -u_i*v_k + v_i*w_j + w_i*(-v_j)  [since w_k = 1-u_k-v_k, dw_k/du_k = -1]
        # = -u_i*v_k + v_i*w_j - w_i*v_j
        
        # d/dv_k: u_i*w_k - u_i*v_k  [wait, let me redo]
        # d/dv_k: -u_i*w_k - v_i*u_j*(-1) + w_i*u_j*(-1)  [dw_k/dv_k = -1]
        # Hmm, this is getting complicated. Let me just use finite differences for the gradient.
        
        # For now, use a simpler approach: the force direction for each point
        # is toward increasing the areas of triangles it participates in.
        
        # For point i in triangle (i,j,k):
        # Moving point i in the direction that increases area means moving
        # away from the line segment jk.
        
        # In barycentric coords, the "normal" to edge jk at point i's position
        # can be computed.
        
        # For simplicity, use the gradient of the area w.r.t. (u_i, v_i):
        # grad_u = (v_j*w_k - v_k*w_j) - (u_j*v_k - u_k*v_j)
        # grad_v = -(u_j*w_k - u_k*w_j) - (u_j*v_k - u_k*v_j)
        
        # But wait, we need to account for the sign of det (since we use |det|).
        # For now, assume det > 0 (we can handle sign separately).
        
        sign = np.sign(det[t_idx])
        if sign == 0:
            sign = 1.0
        
        # Gradient w.r.t. point i
        gi_u = sign * ((vj[t_idx]*wk[t_idx] - vk[t_idx]*wj[t_idx]) - (uj[t_idx]*vk[t_idx] - uk[t_idx]*vj[t_idx]))
        gi_v = sign * (-(uj[t_idx]*wk[t_idx] - uk[t_idx]*wj[t_idx]) - (uj[t_idx]*vk[t_idx] - uk[t_idx]*vj[t_idx]))
        
        # Gradient w.r.t. point j
        gj_u = sign * (-vi[t_idx]*wk[t_idx] + wi[t_idx]*vk[t_idx])
        gj_v = sign * (ui[t_idx]*wk[t_idx] - uk[t_idx]*wi[t_idx])
        
        # Gradient w.r.t. point k
        gk_u = sign * (-ui[t_idx]*vk[t_idx] + vi[t_idx]*wj[t_idx] - wi[t_idx]*vj[t_idx])
        gk_v = sign * (ui[t_idx]*wj[t_idx] - vi[t_idx]*uk[t_idx] - wi[t_idx]*uj[t_idx])
        
        forces[i, 0] += wt * gi_u
        forces[i, 1] += wt * gi_v
        forces[j, 0] += wt * gj_u
        forces[j, 1] += wt * gj_v
        forces[k, 0] += wt * gk_u
        forces[k, 1] += wt * gk_v
    
    return forces

def electrostatic_relax(bary_init, time_limit=10.0):
    """
    Run electrostatic relaxation to find a good configuration.
    
    Uses gradient-based force computation with adaptive step size and
    temperature annealing for exploration.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    temperature = SIM_TEMP_INIT
    dt = SIM_DT
    beta = 50.0
    
    for step in range(SIM_STEPS):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = electrostatic_forces(bary, beta)
        
        # Add random perturbation (annealing)
        rng = np.random.default_rng(step)
        noise = rng.normal(0, temperature, bary.shape)
        
        # Update positions
        bary_new = bary + dt * forces + noise
        bary_new = clamp_bary(bary_new)
        
        # Evaluate
        val = -hard_min_obj(bary_new.flatten())
        
        if val > best_val:
            best_val = val
            best_bary = bary_new.copy()
            bary = bary_new.copy()
            # Reduce temperature on improvement
            temperature *= 0.98
        else:
            # Accept with Metropolis criterion
            if np.log(rng.random()) < (val - best_val) / temperature:
                bary = bary_new.copy()
            # Decay temperature
            temperature *= SIM_TEMP_DECAY
        
        # Gradually increase beta for sharper min
        beta = 50.0 + step * 0.5
    
    return best_bary

def generate_mixed_init():
    """
    Generate a mixed initialization: some points on boundary, some interior.
    
    Strategy: 
    - 3 corners
    - 3 points near bottom edge
    - 3 points near left edge
    - 2 points near right edge
    - With small random perturbations into interior
    """
    rng = np.random.default_rng(42)
    bary = np.zeros((N, 2))
    
    # Corners
    bary[0] = [0.01, 0.01]
    bary[1] = [0.99, 0.01]
    bary[2] = [0.01, 0.99]
    
    # Bottom edge points (with small interior perturbation)
    for i in range(3):
        u = np.linspace(0.2, 0.8, 3)[i]
        bary[3 + i] = [u + rng.normal(0, 0.02), rng.uniform(0.01, 0.05)]
    
    # Left edge points
    for i in range(3):
        v = np.linspace(0.2, 0.8, 3)[i]
        bary[6 + i] = [rng.uniform(0.01, 0.05), v + rng.normal(0, 0.02)]
    
    # Right edge points
    for i in range(2):
        t = np.linspace(0.3, 0.7, 2)[i]
        bary[9 + i] = [(1-t) + rng.normal(0, 0.02), t + rng.normal(0, 0.02)]
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with graduated beta
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.6:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.9:
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
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Electrostatic relaxation from multiple initial configurations
    if time.time() < deadline - 60:
        try:
            rng = np.random.default_rng(42)
            
            # Generate diverse initial configurations
            inits = []
            
            # Hexagonal lattice init
            hex_init = generate_mixed_init()
            inits.append(hex_init)
            
            # Random inits
            for _ in range(4):
                random_bary = rng.uniform(0.05, 0.95, (N, 2))
                random_bary = clamp_bary(random_bary)
                inits.append(random_bary)
            
            # Grid-like init
            grid_bary = np.zeros((N, 2))
            positions = [
                [0.1, 0.1], [0.9, 0.1], [0.1, 0.9],
                [0.3, 0.1], [0.7, 0.1], [0.1, 0.3],
                [0.1, 0.7], [0.3, 0.3], [0.7, 0.3],
                [0.5, 0.5], [0.3, 0.7]
            ]
            for i, pos in enumerate(positions):
                grid_bary[i] = pos
            inits.append(clamp_bary(grid_bary))
            
            # Run electrostatic relaxation on each
            for init in inits:
                if time.time() > deadline - 30:
                    break
                
                remaining = deadline - start_time - 20
                if remaining < 3:
                    break
                
                relaxed = electrostatic_relax(init, time_limit=min(8.0, remaining))
                update_best(relaxed)
                
                # Refine the best from electrostatic with local optimization
                if best_bary is not None and time.time() < deadline - 15:
                    refined = refine_local(best_bary, time_limit=min(5.0, deadline - start_time - 10))
                    update_best(refined)
        except Exception:
            pass
    
    # 2. Mixed initialization refinement
    if time.time() < deadline - 20:
        try:
            mixed = generate_mixed_init()
            refined_mixed = refine_local(mixed, time_limit=min(10.0, deadline - start_time - 10))
            update_best(refined_mixed)
        except Exception:
            pass
    
    # 3. Basin hopping with electrostatic re-relaxation
    if best_bary is not None and time.time() < deadline - 15:
        bh_time = min(15.0, deadline - start_time - 10.0)
        if bh_time > 3:
            try:
                start_bh = time.time()
                rng = np.random.default_rng(123)
                current_bary = best_bary.copy()
                best_bh_bary = best_bary.copy()
                best_bh_val = best_val
                
                temperature = 0.03
                pert_scale = 0.05
                
                while time.time() - start_bh < bh_time:
                    # Perturb
                    noise = rng.normal(0, pert_scale, current_bary.shape)
                    perturbed = clamp_bary(current_bary + noise)
                    
                    # Use electrostatic relaxation instead of local optimization
                    remaining_bh = bh_time - (time.time() - start_bh)
                    if remaining_bh < 2:
                        break
                    
                    # Short electrostatic relaxation
                    relaxed = electrostatic_relax(perturbed, time_limit=min(3.0, remaining_bh - 1))
                    relaxed_val = -hard_min_obj(relaxed.flatten())
                    
                    # Accept if better or with Metropolis
                    if relaxed_val > best_bh_val:
                        best_bh_val = relaxed_val
                        best_bh_bary = relaxed.copy()
                        current_bary = relaxed.copy()
                    elif np.log(rng.random()) < (relaxed_val - best_bh_val) / temperature:
                        current_bary = relaxed.copy()
                    
                    temperature *= 0.97
                    pert_scale *= 0.99
                    
                    if time.time() - start_bh > bh_time * 0.95:
                        break
                
                update_best(best_bh_bary)
            except Exception:
                pass
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
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
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass if time permits
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
