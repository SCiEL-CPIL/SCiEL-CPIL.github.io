"""
Strategy: Joint Node-Weight Optimization via Multi-Start L-BFGS-B with Worst-Case Objective.

1. Reframe the problem as minimizing the worst-case relative error of the quadrature
   approximation against high-precision reference integrals over a diverse set of
   cosmological parameters (w0, wa) and redshifts (z).

2. Use a dense, physics-informed training set to define the objective function.
   This ensures the rule generalizes to the disjoint validation set.

3. Parameterize the 16 variables (8 nodes, 8 weights) directly.
   - Nodes are constrained to (0, 1) and strictly increasing.
   - Weights are constrained to (0, 1) and sum to 1 (via penalty or normalization).

4. Employ a multi-start strategy:
   - Start 1: Standard Gauss-Legendre nodes and weights.
   - Start 2: Nodes shifted towards 0 (biased sampling of the high-integrand region).
   - Start 3: Nodes shifted towards 1.
   - Start 4: Gauss-Lobatto nodes (include endpoints, useful if boundary behavior is sharp).

5. Use scipy.optimize.minimize (L-BFGS-B) for local refinement.
   - The objective function is the maximum relative error over the training set.
   - To handle the non-differentiability of the max function, we use a smooth
     approximation: log-sum-exp of the squared relative errors (or just the max
     with a small epsilon, but L-BFGS-B prefers smoothness).
     Actually, minimizing the max error directly is a linear programming problem
     if the approximation were linear in weights, but it's not (nodes are in the
     integrand).
     We will minimize the sum of squared relative errors (L2) first to get close,
     then switch to minimizing the max error (L-inf) using a smooth proxy or
     just SLSQP which can handle non-smoothness better?
     L-BFGS-B is faster. Let's use a "soft-max" approximation of the error:
     E = (1/k) * log(sum(exp(k * rel_err_i^2))).
     This approaches the max error as k -> infinity.

6. Efficient vectorization:
   - Pre-compute true integrals.
   - Vectorize the computation of approximate integrals for all training points
     given a candidate (nodes, weights) pair.
"""

import time
import numpy as np
from scipy.optimize import minimize, roots_legendre, roots_lobatto
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0 # 5s margin

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0."""
    Omega_de = 1.0 - Omega_m
    # Handle z=0 carefully to avoid 0^0 issues in powers if any, though 1+0=1
    # The formula is robust for z>=0
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_de * (1 + z)**(3 * (1 + w0 + wa * z)))

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute \int_0^1 dt / E(z*t) using quad."""
    if z == 0:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
    return val

def _approx_integral(z, w0, wa, nodes, weights, Omega_m=0.3):
    """Compute \sum w_i / E(z*t_i)."""
    if z == 0:
        return np.sum(weights)
    z_nodes = z * nodes
    E_vals = _E_of_z(z_nodes, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E_vals)

def construct():
    if time.time() > _DEADLINE - 10:
        # Fallback: Standard Gauss-Legendre
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 1. Generate a diverse training set
    # Cover a wide range of w0, wa and z
    w0_vals = np.linspace(-1.5, -0.5, 10)
    wa_vals = np.linspace(-0.5, 0.5, 10)
    z_vals = np.logspace(-2, np.log10(5.0), 20)
    
    # Create a meshgrid for training points
    W0, WA, Z = np.meshgrid(w0_vals, wa_vals, z_vals, indexing='ij')
    W0_flat = W0.flatten()
    WA_flat = WA.flatten()
    Z_flat = Z.flatten()
    
    num_train = len(Z_flat)
    
    # Pre-compute true integrals
    true_vals = np.zeros(num_train)
    for i in range(num_train):
        true_vals[i] = _true_integral(Z_flat[i], W0_flat[i], WA_flat[i])

    # Helper to compute errors for a given set of nodes and weights
    def compute_errors(nodes, weights):
        # Vectorized computation
        # Z_flat: (N, 1)
        # nodes: (1, 8)
        # z_nodes: (N, 8)
        z_nodes = Z_flat[:, None] * nodes[None, :]
        
        # Compute E
        Om_m = 0.3
        Om_de = 0.7
        one_plus_z = 1 + z_nodes
        term_m = Om_m * one_plus_z**3
        # w0, wa are vectors of length N
        # Broadcast to (N, 8)
        exp_arg = 3 * (1 + W0_flat[:, None] + WA_flat[:, None] * z_nodes)
        term_de = Om_de * one_plus_z**exp_arg
        E_vals = np.sqrt(term_m + term_de)
        inv_E = 1.0 / E_vals
        
        # Approx integral
        approx_vals = np.dot(inv_E, weights)
        
        # Relative errors
        rel_errs = np.abs(approx_vals - true_vals) / true_vals
        return rel_errs

    # Objective function for optimization
    # We minimize a smooth approximation of the max relative error
    # E = (1/k) * log(sum(exp(k * rel_errs^2)))
    k_smooth = 10.0
    
    def objective(params):
        nodes = params[:8]
        weights = params[8:]
        
        # Check validity
        if not np.all(nodes > 0) or not np.all(nodes < 1):
            return 1e6
        if not np.all(np.diff(nodes) > 1e-6):
            return 1e6
        if not np.all(weights > 0):
            return 1e6
            
        # Normalize weights to sum to 1 to handle constraint loosely during optimization
        # Actually, it's better to enforce sum(w)=1 strictly in the parameterization
        # or via penalty. Let's use penalty for sum(w)-1.
        sum_w_penalty = (np.sum(weights) - 1.0)**2 * 100.0
        
        rel_errs = compute_errors(nodes, weights)
        
        # Smooth max
        # Clip errors to avoid overflow in exp
        errs_sq = np.clip(rel_errs**2, 0, 100)
        smooth_max = (1.0 / k_smooth) * np.log(np.sum(np.exp(k_smooth * errs_sq)))
        
        return smooth_max + sum_w_penalty

    # Prepare bounds
    # Nodes: (1e-6, 1-1e-6)
    # Weights: (1e-6, 1)
    bounds = []
    for i in range(8):
        bounds.append((1e-6, 1 - 1e-6)) # Nodes
    for i in range(8):
        bounds.append((1e-6, 1.0)) # Weights
        
    # Starting points
    starts = []
    
    # 1. Standard Gauss-Legendre
    n_gl, w_gl = roots_legendre(8)
    nodes_gl = (n_gl + 1) / 2
    weights_gl = w_gl / 2
    starts.append(np.concatenate([nodes_gl, weights_gl]))
    
    # 2. Gauss-Lobatto (includes 0 and 1, might be better for boundary behavior)
    try:
        n_lb, w_lb = roots_lobatto(8)
        nodes_lb = (n_lb + 1) / 2
        weights_lb = w_lb / 2
        starts.append(np.concatenate([nodes_lb, weights_lb]))
    except:
        pass

    # 3. Shifted GL towards 0 (mass where integrand is higher)
    # t_i = (1 + x_i)/2 * 0.8 + 0.1 ? No, keep in [0,1].
    # Let's just shift the center.
    # t_i = 0.4 * (1 + x_i) ? Range [0, 0.8].
    # We need nodes in [0,1].
    # Let's try a non-linear map: t_i = x_i^2 mapped to [0,1]?
    # x_i in [-1,1]. x_i^2 in [0,1].
    # This clusters nodes near 0.
    nodes_shift = n_gl**2
    # Ensure strictly increasing
    # n_gl are increasing, so n_gl^2 are increasing for n_gl>0.
    # But n_gl includes negative values.
    # Let's use a simpler shift: t_i = (1 + x_i)/2 * 0.9 + 0.05
    nodes_shift2 = (1 + n_gl)/2 * 0.9 + 0.05
    # Check bounds
    if np.all(nodes_shift2 > 0) and np.all(nodes_shift2 < 1) and np.all(np.diff(nodes_shift2) > 1e-6):
         starts.append(np.concatenate([nodes_shift2, weights_gl]))
         
    # 4. Uniform nodes
    nodes_uniform = np.linspace(0.05, 0.95, 8)
    weights_uniform = np.ones(8) / 8
    starts.append(np.concatenate([nodes_uniform, weights_uniform]))

    best_params = None
    best_score = np.inf

    for i, start_params in enumerate(starts):
        if time.time() > _DEADLINE - 15:
            break
            
        try:
            # Optimize
            res = minimize(objective, start_params, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Evaluate actual worst-case error (not the smooth proxy)
                nodes_opt = res.x[:8]
                weights_opt = res.x[8:]
                
                # Normalize weights
                weights_opt = weights_opt / np.sum(weights_opt)
                
                # Check constraints
                if not (np.all(nodes_opt > 1e-6) and np.all(nodes_opt < 1 - 1e-6)):
                    continue
                if not np.all(np.diff(nodes_opt) > 1e-6):
                    continue
                if not np.all(weights_opt > 0):
                    continue
                    
                rel_errs = compute_errors(nodes_opt, weights_opt)
                worst_err = np.max(rel_errs)
                
                if worst_err < best_score:
                    best_score = worst_err
                    best_params = res.x.copy()
                    
        except Exception as e:
            continue

    if best_params is None:
        # Fallback
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # Final Refinement:
    # The L-BFGS-B result might have weights not summing exactly to 1 due to the penalty.
    # We perform a final constrained optimization on weights only, fixing the nodes,
    # to minimize the worst-case error exactly.
    
    nodes_final = best_params[:8]
    weights_init = best_params[8:]
    weights_init = weights_init / np.sum(weights_init)
    
    def weight_objective(w):
        # w sums to 1? We assume the optimizer enforces it or we normalize.
        # Let's normalize inside to be safe, or use SLSQP with constraint.
        w_norm = w / np.sum(w)
        rel_errs = compute_errors(nodes_final, w_norm)
        # Minimize max error
        # Use smooth max again for differentiability
        errs_sq = np.clip(rel_errs**2, 0, 100)
        smooth_max = (1.0 / k_smooth) * np.log(np.sum(np.exp(k_smooth * errs_sq)))
        return smooth_max

    # Use SLSQP for equality constraint sum(w)=1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds_w = [(1e-8, 1.0)] * 8
    
    try:
        res_w = minimize(weight_objective, weights_init, method='SLSQP', bounds=bounds_w,
                         constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
        if res_w.success:
            weights_final = res_w.x
        else:
            weights_final = weights_init
    except:
        weights_final = weights_init
    
    weights_final = weights_final / np.sum(weights_final)
    
    # Final sanity check
    if not np.all(np.diff(nodes_final) > 1e-6):
        nodes_final, weights_final = roots_legendre(8)
        nodes_final = (nodes_final + 1) / 2
        weights_final = weights_final / 2
        
    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
