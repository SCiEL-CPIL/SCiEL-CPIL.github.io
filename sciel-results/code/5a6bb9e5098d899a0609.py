"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution

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

def _exact_dL(z, wa, wb, w0, npts=4000):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    from scipy.integrate import quad
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

def _find_optimal_weights(nodes, z_samples, models, max_iter=50):
    """Find weights that minimize worst-case relative error via iterative LP."""
    n = len(nodes)
    n_cosmo = len(models)
    n_z = len(z_samples)
    
    # Build constraint matrix: for each (cosmo, z), we want
    # sum_i w_i / E_i(cosmo, z) ≈ exact_integral(cosmo, z)
    # where exact_integral = exact_dL / ((1+z)*z)
    
    A = np.zeros((n_cosmo * n_z, n))
    b_exact = np.zeros(n_cosmo * n_z)
    
    for ci, (wa, wb, w0) in enumerate(models):
        for zi, z in enumerate(z_samples):
            idx = ci * n_z + zi
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a, wa, wb, w0)
            A[idx, :] = 1.0 / E_vals
            exact_dL_val = _exact_dL(z, wa, wb, w0)
            if z > 0:
                b_exact[idx] = exact_dL_val / ((1.0 + z) * z)
            else:
                b_exact[idx] = 0.0
    
    # Minimize worst-case relative error:
    # minimize t such that |A w - b| / b <= t * b (for all rows)
    # This is: A w - b <= t * b  and  -(A w - b) <= t * b
    # i.e., A w <= (1+t) * b  and  A w >= (1-t) * b
    # With constraints: sum(w) = 1, w > 0
    
    # Use iterative approach: start with equal weights, then refine
    w = np.ones(n) / n
    
    for it in range(max_iter):
        # Compute current errors
        Aw = A @ w
        rel_errs = np.abs(Aw - b_exact) / np.maximum(b_exact, 1e-15)
        t = np.max(rel_errs)
        
        # Identify active constraints (largest errors)
        active_idx = np.where(rel_errs > 0.8 * t)[0]
        
        # Solve LP to minimize t with active constraints
        # Variables: w_0..w_{n-1}, t
        # Minimize t
        c = np.zeros(n + 1)
        c[-1] = 1.0
        
        # Constraints:
        # For each active row i: A_i @ w - t * b_i <= 0  and  -A_i @ w + t * b_i <= 0
        A_ub = np.zeros((2 * len(active_idx), n + 1))
        b_ub = np.zeros(2 * len(active_idx))
        
        for k, idx in enumerate(active_idx):
            A_ub[2*k, :n] = A[idx, :]
            A_ub[2*k, n] = -b_exact[idx]
            A_ub[2*k+1, :n] = -A[idx, :]
            A_ub[2*k+1, n] = b_exact[idx]
        
        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w >= 1e-10, t >= 0
        bounds = [(1e-10, None)] * n + [(0, None)]
        
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

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline (10s margin)
    
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
    
    n = 8
    best_nodes = (np.arange(n) + 0.5) / n
    best_weights = np.full(n, 1.0 / n)
    best_score = np.inf
    
    # Initial evaluation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # Differential evolution for nodes
    def objective(nodes_flat):
        nodes = np.sort(nodes_flat)
        # Check strictly increasing
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e6
        if nodes[0] < 0 or nodes[-1] > 1:
            return 1e6
        
        # Find optimal weights for these nodes
        weights, t = _find_optimal_weights(nodes, z_samples, models, max_iter=20)
        
        # Evaluate worst-case error
        errs = []
        for z in z_samples:
            for wa, wb, w0 in models:
                err = _relative_error(z, nodes, weights, wa, wb, w0)
                errs.append(err)
        return np.max(errs)
    
    bounds = [(0.001, 0.999)] * n
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=15,
            tol=1e-6, seed=42,
            polish=True,
            workers=1,
            time_limit=60
        )
        
        if result.success or result.fun < best_score:
            best_nodes = np.sort(result.x)
            if np.all(np.diff(best_nodes) > 1e-12):
                best_weights, _ = _find_optimal_weights(best_nodes, z_samples, models, max_iter=30)
                
                # Final evaluation
                errors = []
                for z in z_samples:
                    for wa, wb, w0 in models:
                        err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
                        errors.append(err)
                final_score = np.max(errors)
                if final_score < best_score:
                    best_score = final_score
    except Exception as e:
        pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
