"""Quadrature rule for luminosity distance - using Chebyshev-based initial nodes and alternating optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.special import roots_chebyshev

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _compute_all_errors(nodes, weights, z_samples, models):
    """Compute all relative errors for given nodes and weights."""
    errs = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            errs.append(err)
    return np.array(errs)

def _find_optimal_weights(nodes, z_samples, models):
    """Find weights that minimize worst-case relative error using LP approach."""
    n = len(nodes)
    n_cosmo = len(models)
    n_z = len(z_samples)
    
    # Build constraint matrix
    A = np.zeros((n_cosmo * n_z, n))
    b_exact = np.zeros(n_cosmo * n_z)
    
    for ci, (wa, wb, w0) in enumerate(models):
        for zi, z in enumerate(z_samples):
            idx = ci * n_z + zi
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a, wa, wb, w0)
            A[idx, :] = 1.0 / E_vals
            exact_dL_val = _exact_dL(z, wa, wb, w0)
            if z > 1e-6:
                b_exact[idx] = exact_dL_val / ((1.0 + z) * z)
            else:
                b_exact[idx] = 0.0
    
    # Start with equal weights
    w = np.ones(n) / n
    
    # Iterative LP to minimize worst-case error
    from scipy.optimize import linprog
    for it in range(20):
        # Compute current errors
        Aw = A @ w
        rel_errs = np.abs(Aw - b_exact) / np.maximum(b_exact, 1e-15)
        t = np.max(rel_errs)
        
        # Identify active constraints (largest errors)
        active_idx = np.where(rel_errs > 0.7 * t)[0]
        
        if len(active_idx) == 0:
            break
        
        # Solve LP to minimize t with active constraints
        c = np.zeros(n + 1)
        c[-1] = 1.0
        
        A_ub = np.zeros((2 * len(active_idx), n + 1))
        b_ub = np.zeros(2 * len(active_idx))
        
        for k, idx in enumerate(active_idx):
            A_ub[2*k, :n] = A[idx, :]
            A_ub[2*k, n] = -b_exact[idx]
            A_ub[2*k+1, :n] = -A[idx, :]
            A_ub[2*k+1, n] = b_exact[idx]
        
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        bounds = [(1e-12, None)] * n + [(0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
            if res.success:
                w_new = res.x[:n]
                t_new = res.x[n]
                if t_new < t * 0.999:
                    w = w_new
                    t = t_new
                else:
                    break
            else:
                break
        except:
            break
    
    return w, t

def _objective_nodes(nodes_flat, z_samples, models):
    """Objective for node optimization: worst-case error with LP-optimal weights."""
    nodes = np.sort(nodes_flat)
    if np.any(np.diff(nodes) <= 1e-10):
        return 1e6
    if nodes[0] < 1e-6 or nodes[-1] > 1.0 - 1e-6:
        return 1e6
    
    weights, t = _find_optimal_weights(nodes, z_samples, models, max_iter=15)
    return t

def _chebyshev_nodes(n):
    """Generate Chebyshev nodes on [0,1]."""
    x, _ = roots_chebyshev(n)
    x = 0.5 * (x + 1.0)
    idx = np.argsort(x)
    x = x[idx]
    return x

def _gauss_legendre_nodes(n):
    """Generate Gauss-Legendre nodes and weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    w = w / np.sum(w)
    return x, w

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    n = 8
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
    ]
    
    # Z samples for evaluation
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Start with multiple initial points
    initial_nodes_list = [
        _gauss_legendre_nodes(n)[0],
        _chebyshev_nodes(n),
        np.array([0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.995]),
    ]
    
    best_nodes = None
    best_weights = None
    best_score = np.inf
    
    # Evaluate each initial point
    for init_nodes in initial_nodes_list:
        weights, t = _find_optimal_weights(init_nodes, z_samples, models, max_iter=25)
        if t < best_score:
            best_score = t
            best_nodes = init_nodes.copy()
            best_weights = weights.copy()
    
    # Differential evolution for node optimization
    if time.time() < deadline - 25:
        bounds = [(0.001, 0.999)] * n
        time_limit = min(40, deadline - time.time() - 10)
        
        try:
            result = differential_evolution(
                _objective_nodes, bounds,
                args=(z_samples, models),
                maxiter=50, popsize=15,
                tol=1e-7, seed=42,
                polish=False,
                workers=1,
                time_limit=time_limit
            )
            
            if result.fun < best_score:
                best_nodes = np.sort(result.x)
                if np.all(np.diff(best_nodes) > 1e-10):
                    best_weights, best_score = _find_optimal_weights(best_nodes, z_samples, models, max_iter=30)
        except Exception:
            pass
    
    # Final refinement: optimize both nodes and weights jointly
    if time.time() < deadline - 5:
        def _joint_objective(params):
            log_nodes = params[:n]
            log_weights = params[n:]
            
            # Convert log params to nodes in (0,1)
            nodes = 0.5 * (1.0 + np.tanh(log_nodes))
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            for i in range(1, n):
                if nodes[i] <= nodes[i-1] + 1e-10:
                    nodes[i] = nodes[i-1] + 1e-8
            
            # Convert log weights to positive weights summing to 1
            log_w = log_weights - np.max(log_weights)
            w = np.exp(log_w)
            w = w / np.sum(w)
            
            errs = _compute_all_errors(nodes, w, z_samples, models)
            return np.max(errs)
        
        # Initialize from current best
        log_nodes_init = np.arctanh(2.0 * best_nodes - 1.0)
        log_weights_init = np.log(best_weights)
        params_init = np.concatenate([log_nodes_init, log_weights_init])
        
        try:
            result = minimize(
                _joint_objective, params_init,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 2000, 'ftol': 1e-12}
            )
            
            if result.fun < best_score:
                log_nodes = result.x[:n]
                log_weights = result.x[n:]
                
                nodes = 0.5 * (1.0 + np.tanh(log_nodes))
                nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
                for i in range(1, n):
                    if nodes[i] <= nodes[i-1] + 1e-10:
                        nodes[i] = nodes[i-1] + 1e-8
                
                log_w = log_weights - np.max(log_weights)
                w = np.exp(log_w)
                w = w / np.sum(w)
                
                errs = _compute_all_errors(nodes, w, z_samples, models)
                final_score = np.max(errs)
                
                if final_score < best_score:
                    best_nodes = nodes
                    best_weights = w
                    best_score = final_score
        except Exception:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
