"""
Strategy: Fix Gauss-Legendre nodes for structural simplicity and use
a multi-stage optimization of weights. The key insight is that with
fixed nodes, we can precompute the integrand values for many models,
then optimize weights to minimize worst-case relative error efficiently.

We use:
1. A broad grid of CPL models for the objective
2. Differential Evolution for global weight optimization
3. Local refinement with Nelder-Mead
4. Final validation against additional held-out models
"""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import roots_legendre

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    om_de0 = 1.0 - om_m - om_r - om_k
    # H^2/H0^2 = om_m a^-3 + om_r a^-4 + om_de0 a^(-3(1+w0+wa)) + om_k
    h2 = om_m * a**(-3) + om_r * a**(-4) + om_k + om_de0 * a**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand_values(nodes, z, w0, wa, om_m, om_r, om_k=0.0):
    """Compute 1/E(a) at the given nodes for a specific model and redshift."""
    a = 1.0 / (1.0 + z * nodes)
    E = _E_cpl(a, w0, wa, om_m, om_r, om_k)
    return 1.0 / E

def _true_integral(nodes, weights, z, w0, wa, om_m, om_r, om_k=0.0, n_ref=256):
    """Compute reference integral using high-order Gauss-Legendre."""
    x_ref, w_ref = roots_legendre(n_ref)
    x_ref = (x_ref + 1.0) / 2.0
    w_ref = w_ref / 2.0
    
    a_ref = 1.0 / (1.0 + z * x_ref)
    E_ref = _E_cpl(a_ref, w0, wa, om_m, om_r, om_k)
    f_ref = 1.0 / E_ref
    return np.dot(f_ref, w_ref)

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s limit
    
    # Fix Gauss-Legendre nodes on [0,1]
    x_gl, w_gl = roots_legendre(8)
    gl_nodes = (x_gl + 1.0) / 2.0
    gl_weights = w_gl / 2.0
    
    # Define training models - broad coverage of parameter space
    np.random.seed(12345)
    
    # Systematic grid
    z_vals = [0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    wa_vals = [-1.5, -0.5, 0.0, 0.5, 1.5]
    om_m_vals = [0.25, 0.3, 0.35]
    
    models = []
    # Include key combinations (not full cross product to save time)
    for z in z_vals:
        for w0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for wa in [-1.0, 0.0, 1.0]:
                for om_m in [0.3]:
                    models.append((z, w0, wa, om_m, 9e-5, 0.0))
    
    # Add more diversity
    for _ in range(60):
        z = np.random.uniform(0.05, 8.0)
        w0 = np.random.uniform(-2.0, 2.0)
        wa = np.random.uniform(-2.0, 2.0)
        om_m = np.random.uniform(0.2, 0.4)
        models.append((z, w0, wa, om_m, 9e-5, 0.0))
    
    # Precompute integrand values for all models at GL nodes
    # F[m, i] = 1/E(a_i) for model m at node i
    n_models = len(models)
    F = np.zeros((n_models, 8))
    I_true = np.zeros(n_models)
    
    for m, (z, w0, wa, om_m, om_r, om_k) in enumerate(models):
        F[m, :] = _integrand_values(gl_nodes, z, w0, wa, om_m, om_r, om_k)
        I_true[m] = _true_integral(gl_nodes, gl_weights, z, w0, wa, om_m, om_r, om_k, n_ref=256)
    
    def compute_worst_error(weights):
        """Compute worst relative error given weights."""
        Q = F.dot(weights)  # shape (n_models,)
        rel_err = np.abs(Q - I_true) / np.abs(I_true)
        return np.max(rel_err)
    
    # Optimization: use log-weights for positivity constraint
    def objective_log(log_w):
        w = np.exp(log_w)
        w = w / np.sum(w)
        return compute_worst_error(w)
    
    # Initial guess: GL weights
    x0_log = np.log(gl_weights)
    
    best_log_w = x0_log.copy()
    best_score = objective_log(x0_log)
    
    # Stage 1: Differential Evolution (global search)
    if time.time() < deadline - 20.0:
        try:
            bounds = [(-4.0, 0.5)] * 8  # log weights between ~0.018 and 1.65
            
            res = differential_evolution(
                objective_log,
                bounds,
                seed=42,
                maxiter=80,
                popsize=12,
                tol=1e-7,
                polish=True,
                init='sobol',
                mutation=(0.3, 0.8),
                recombination=0.7
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_log_w = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            w_current = np.exp(best_log_w)
            w_current = w_current / np.sum(w_current)
            
            res_nm = minimize(
                lambda w: compute_worst_error(w / np.sum(w)),
                w_current,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                w_final = res_nm.x
                w_final = np.maximum(w_final, 1e-10)
                w_final = w_final / np.sum(w_final)
                best_log_w = np.log(w_final)
        except Exception:
            pass
    
    # Stage 3: Fine-tune with multiple restarts if time permits
    if time.time() < deadline - 10.0:
        for seed in [7, 13, 23, 42]:
            if time.time() >= deadline - 10.0:
                break
            try:
                perturbation = np.random.normal(0, 0.1, 8)
                x0_pert = best_log_w + perturbation
                
                res_de = differential_evolution(
                    objective_log,
                    [(-4.0, 0.5)] * 8,
                    seed=seed,
                    maxiter=40,
                    popsize=10,
                    tol=1e-8,
                    polish=True,
                    init='latinhypercube'
                )
                
                if res_de.fun < best_score:
                    best_score = res_de.fun
                    best_log_w = res_de.x.copy()
            except Exception:
                continue
    
    # Final weights
    final_weights = np.exp(best_log_w)
    final_weights = np.maximum(final_weights, 1e-15)
    final_weights = final_weights / np.sum(final_weights)
    
    # Validation against held-out models
    np.random.seed(99999)
    holdout_models = []
    for _ in range(30):
        z = np.random.uniform(0.05, 10.0)
        w0 = np.random.uniform(-2.0, 2.0)
        wa = np.random.uniform(-2.0, 2.0)
        om_m = np.random.uniform(0.2, 0.4)
        holdout_models.append((z, w0, wa, om_m, 9e-5, 0.0))
    
    max_holdout_err = 0.0
    for (z, w0, wa, om_m, om_r, om_k) in holdout_models:
        f_nodes = _integrand_values(gl_nodes, z, w0, wa, om_m, om_r, om_k)
        Q = np.dot(f_nodes, final_weights)
        I_ref = _true_integral(gl_nodes, gl_weights, z, w0, wa, om_m, om_r, om_k, n_ref=256)
        if I_ref > 0:
            err = abs(Q - I_ref) / I_ref
            max_holdout_err = max(max_holdout_err, err)
    
    # Ensure validity
    if not (np.all(gl_nodes > 0) and np.all(gl_nodes < 1) and np.all(np.diff(gl_nodes) > 1e-10)):
        # Fallback (shouldn't happen with GL nodes)
        gl_nodes = np.linspace(1/17, 16/17, 8)
        final_weights = np.ones(8) / 8.0
    
    if not np.all(final_weights > 0):
        final_weights = np.ones(8) / 8.0
    
    final_weights = final_weights / np.sum(final_weights)
    
    return {
        "nodes": gl_nodes,
        "weights": final_weights
    }
