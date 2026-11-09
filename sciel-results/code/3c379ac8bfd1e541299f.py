"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

def _E_of_a(a, wa, wb):
    """CPL dark energy E(a) = H(a)/H0 for flat universe."""
    # E^2 = (1-a)^3 + wa * a^(-3*(1+wb)) + (1-wa-wb)*a^(-3*wb)
    # Actually standard CPL: E^2 = (1-a)^3 + wa * a^(-3(1+wb)) + (1-wa-wb)*a^(-3*wb)
    # Let me use the standard form: E^2 = (1-a)^3 + wa * a^(-3(1+wb)) + (1-wa-wb)*a^(-3*wb)
    # More precisely: E^2(a) = (1-a)^3 + wa * a^(-3(1+wb)) + (1-wa-wb)*a^(-3*wb)
    # Standard CPL: w = wa + wb(1-a)
    # E^2 = (1-a)^3 + wa * a^(-3(1+wb)) + (1-wa-wb) * a^(-3*wb)
    # Let me verify: at a=1, E^2=1. 
    # (1-1)^3 + wa*1 + (1-wa-wb)*1 = 1. Good.
    # At a->0, if wb<0, a^(-3wb) -> 0, so E^2 -> (1-a)^3 + wa -> 1+wa... 
    # Actually the standard form is:
    # E^2 = (1-a)^3 + wa * a^(-3(1+wb)) + (1-wa-wb) * a^(-3*wb)
    # Let me just use a robust numerical form.
    
    a_safe = np.maximum(a, 1e-15)
    term_matter = (1.0 - a_safe)**3
    term_wa = wa * a_safe**(-3.0*(1.0+wb))
    term_wb = (1.0 - wa - wb) * a_safe**(-3.0*wb)
    return np.sqrt(term_matter + term_wa + term_wb)

def _exact_lum_dist(z, wa, wb, tol=1e-10):
    """Compute exact luminosity distance dL/(c/H0) via adaptive quadrature."""
    # dL/(c/H0) = (1+z) * ∫_0^z z' / E(1/(1+z')) dz'
    # Change variable: a = 1/(1+z), so z = 1/a - 1, dz = -da/a^2
    # When z=0, a=1; when z=z, a=1/(1+z)
    # ∫_0^z z'/E(a') dz' = ∫_1^{1/(1+z)} (1/a' - 1)/E(a') * (-da'/a'^2)
    # = ∫_{1/(1+z)}^1 (1-a')/(a'^3 * E(a')) da'
    # So dL/(c/H0) = (1+z) * ∫_{1/(1+z)}^1 (1-a)/(a^3 * E(a)) da
    
    a_min = 1.0/(1.0 + z)
    
    def integrand(a):
        Ea = _E_of_a(a, wa, wb)
        return (1.0 - a) / (a**3 * Ea)
    
    val, err = quad(integrand, a_min, 1.0, epsabs=tol, epsrel=tol, limit=200)
    return (1.0 + z) * val

def _quadrature_lum_dist(z, nodes, weights, wa, wb):
    """Compute quadrature approximation of dL/(c/H0)."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a_vals, wa, wb)
    # Integrand in z-space: f(z') = z' / E(1/(1+z'))
    # But we're integrating over z from 0 to z with nodes in [0,1]
    # The formula: dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i)
    # where a_i = 1/(1 + z * nodes[i])
    # This means we're approximating ∫_0^1 u / E(1/(1+zu)) du * z ... 
    # Actually the contract says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # So the integrand being approximated is 1/E(a(u)) where u in [0,1]
    # and a(u) = 1/(1+zu)
    # The true integral is ∫_0^1 1/E(1/(1+zu)) du
    # And dL/(c/H0) = (1+z) * z * ∫_0^1 1/E(1/(1+zu)) du
    
    integral_approx = np.dot(weights, 1.0 / E_vals)
    return (1.0 + z) * z * integral_approx

def _rel_error(z, nodes, weights, wa, wb):
    """Relative error in percent."""
    exact = _exact_lum_dist(z, wa, wb)
    approx = _quadrature_lum_dist(z, nodes, weights, wa, wb)
    if abs(exact) < 1e-15:
        return 0.0
    return 100.0 * abs(approx - exact) / abs(exact)

def _worst_case_error(nodes, weights, z_list, model_list):
    """Worst relative error across all (z, model) pairs."""
    max_err = 0.0
    for z in z_list:
        for (wa, wb) in model_list:
            err = _rel_error(z, nodes, weights, wa, wb)
            if err > max_err:
                max_err = err
    return max_err

def _solve_weights_lp(nodes, model_list, z_list):
    """Solve for optimal weights via LP to minimize worst-case error.
    
    This is complex; instead we use a simpler approach:
    For given nodes, find weights that minimize max deviation across models.
    We'll use a penalty-based optimization.
    """
    n = len(nodes)
    
    def objective(w):
        # Ensure positivity
        w_pos = np.maximum(w, 1e-10)
        w_pos = w_pos / np.sum(w_pos)
        return _worst_case_error(nodes, w_pos, z_list, model_list)
    
    # Initial guess: uniform
    w0 = np.full(n, 1.0/n)
    
    # Use SLSQP with bounds
    bounds = [(1e-10, 1.0) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, 
                   constraints=constraints, options={'maxiter': 50, 'ftol': 1e-12})
    
    if res.success:
        w_opt = np.maximum(res.x, 1e-10)
        w_opt = w_opt / np.sum(w_opt)
        return w_opt
    else:
        return w0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Define validation models and redshifts
    # Use a range of CPL parameters
    model_list = [
        (-0.9, 0.0),   # LambdaCDM
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, 0.5),   # CPL variant
        (-0.7, 0.3),   # Another CPL
        (-1.2, 0.2),   # Phantom-like
        (-0.3, -0.5),  # Different regime
        (0.0, 0.0),    # Matter only (edge case)
        (-0.8, -0.2),  # Another
    ]
    
    z_list = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes as initial guess
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(n)
    nodes_init = (x_gl + 1.0) / 2.0
    
    # First, optimize nodes with a global-ish approach
    # Use multiple random restarts with Nelder-Mead or differential evolution
    
    best_nodes = nodes_init.copy()
    best_weights = w_gl / 2.0
    best_err = _worst_case_error(best_nodes, best_weights, z_list, model_list)
    
    # Try several initial configurations
    initial_configs = [
        nodes_init,
        (np.arange(n) + 0.5) / n,  # Midpoint
        np.linspace(0.1, 0.9, n),  # Evenly spaced interior
    ]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(3):
        pert = nodes_init + rng.normal(0, 0.05, n)
        pert = np.sort(np.clip(pert, 1e-6, 1.0 - 1e-6))
        initial_configs.append(pert)
    
    for nodes_init in initial_configs:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize nodes
        def node_objective(nodes_flat):
            nodes = np.sort(np.clip(nodes_flat, 1e-6, 1.0 - 1e-6))
            # Solve weights LP for these nodes
            weights = _solve_weights_lp(nodes, model_list, z_list)
            return _worst_case_error(nodes, weights, z_list, model_list)
        
        # Use Nelder-Mead on nodes
        res = minimize(node_objective, nodes_init, method='Nelder-Mead',
                       options={'maxiter': 30, 'fatol': 1e-10, 'xatol': 1e-10})
        
        if res.success and res.fun < best_err:
            best_nodes = np.sort(np.clip(res.x, 1e-6, 1.0 - 1e-6))
            best_weights = _solve_weights_lp(best_nodes, model_list, z_list)
            best_err = res.fun
    
    # Final refinement: optimize both nodes and weights together
    def full_objective(params):
        nodes = np.sort(np.clip(params[:n], 1e-6, 1.0 - 1e-6))
        weights = np.maximum(params[n:], 1e-10)
        weights = weights / np.sum(weights)
        return _worst_case_error(nodes, weights, z_list, model_list)
    
    params_init = np.concatenate([best_nodes, best_weights])
    
    if time.time() < deadline - 10.0:
        res = minimize(full_objective, params_init, method='Nelder-Mead',
                       options={'maxiter': 50, 'fatol': 1e-12, 'xatol': 1e-12})
        if res.success and res.fun < best_err:
            best_nodes = np.sort(np.clip(res.x[:n], 1e-6, 1.0 - 1e-6))
            best_weights = np.maximum(res.x[n:], 1e-10)
            best_weights = best_weights / np.sum(best_weights)
            best_err = res.fun
    
    # Ensure constraints are met
    best_nodes = np.sort(np.clip(best_nodes, 1e-6, 1.0 - 1e-6))
    best_weights = np.maximum(best_weights, 1e-10)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
