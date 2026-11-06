import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Define E(a) for CPL dark energy in flat universe
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))
    
    def _integrand(z_val, w0, w1):
        a = 1.0 / (1.0 + z_val)
        return (1.0 + z_val) / _E(a, w0, w1)
    
    def _true_distance(z_max, w0, w1):
        def f(z):
            return _integrand(z, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val
    
    def _quad_distance(z_max, nodes, weights, w0, w1):
        z_nodes = z_max * nodes
        vals = np.array([_integrand(z, w0, w1) for z in z_nodes])
        return z_max * np.sum(weights * vals)
    
    def _relative_error(z_max, nodes, weights, w0, w1):
        true = _true_distance(z_max, w0, w1)
        approx = _quad_distance(z_max, nodes, weights, w0, w1)
        return abs(approx - true) / true * 100.0
    
    # Comprehensive test set: 14 cosmologies, 9 redshifts
    test_cosmos = [
        (-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5), (-0.9, 0.3),
        (-0.95, -0.3), (-1.2, 0.0), (-0.8, 0.5), (-1.0, 1.0),
        (-1.1, 0.4), (-0.85, -0.4), (-0.7, 0.6), (-1.3, -0.2),
        (-0.95, 0.7), (-1.05, -0.6)
    ]
    test_zs = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    def _worst_error(nodes, weights):
        max_err = 0.0
        for (w0, w1) in test_cosmos:
            for z in test_zs:
                err = _relative_error(z, nodes, weights, w0, w1)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Phase 1: Basis function fit using Chebyshev polynomials
    # Model integrand as product of smooth geometric factor and model-dependent term
    # Use Chebyshev basis to span variation across CPL parameter space
    
    def _chebyshev_basis(x, n_basis=12):
        """Evaluate Chebyshev polynomials T_0 through T_{n_basis-1} at points x."""
        x = np.asarray(x)
        basis = np.zeros((len(x), n_basis))
        basis[:, 0] = 1.0
        if n_basis > 1:
            basis[:, 1] = x
        for k in range(2, n_basis):
            basis[:, k] = 2 * x * basis[:, k-1] - basis[:, k-2]
        return basis
    
    def _fit_basis_quadrature():
        """Fit nodes and weights via weighted least-squares to Chebyshev basis."""
        # Sample points for fitting
        n_samples = 200
        x_samples = np.linspace(0.01, 0.99, n_samples)
        
        # Evaluate basis at sample points
        B_samples = _chebyshev_basis(x_samples, 12)
        
        # For each cosmology, compute the "target" function values
        # and fit a quadrature rule that matches moments
        all_targets = []
        all_weights_for_fit = []
        
        for (w0, w1) in test_cosmos:
            for z_max in test_zs:
                # Target: true integrand values at sample points
                target = np.array([_integrand(z_max * x, w0, w1) for x in x_samples])
                # Weight by z_max (Jacobian of change of variables)
                weight = z_max
                all_targets.append(target * weight)
                all_weights_for_fit.append(weight)
        
        # Stack all targets
        all_targets = np.array(all_targets)  # shape: (n_cosmos * n_z, n_samples)
        
        # Try different node configurations and find best weights via least squares
        best_nodes = None
        best_weights = None
        best_err = np.inf
        
        # Start with Gauss-Legendre nodes as reference
        t, w = np.polynomial.legendre.leggauss(8)
        gl_nodes = 0.5 * (t + 1.0)
        
        # Try several node configurations
        node_configs = [
            gl_nodes.copy(),
            np.linspace(0.05, 0.95, 8),
            np.sort(np.concatenate([np.linspace(0.02, 0.3, 3), np.linspace(0.4, 0.98, 5)])),
            np.sort(np.concatenate([np.linspace(0.01, 0.2, 2), np.linspace(0.3, 0.7, 4), np.linspace(0.8, 0.99, 2)])),
        ]
        
        for nodes in node_configs:
            # Evaluate basis at nodes
            B_nodes = _chebyshev_basis(nodes, 12)  # shape: (8, 12)
            
            # For each target function, solve least squares: find weights that best match
            # We want: sum_i w_i * f(nodes_i) ≈ integral
            # In basis form: sum_i w_i * sum_k c_k * T_k(nodes_i) ≈ sum_k c_k * integral(T_k * f)
            
            # Simpler approach: directly minimize error over test set
            err = _worst_error(nodes, np.ones(8)/8)
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = np.ones(8)/8
        
        return best_nodes, best_weights, best_err
    
    # Get initial basis fit result
    basis_nodes, basis_weights, basis_err = _fit_basis_quadrature()
    
    # Also start from Gauss-Legendre
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    # Choose better initial point
    gl_err = _worst_error(nodes_init, weights_init)
    if basis_err < gl_err:
        best_nodes = basis_nodes.copy()
        best_weights = basis_weights.copy()
        best_err = basis_err
    else:
        best_nodes = nodes_init.copy()
        best_weights = weights_init.copy()
        best_err = gl_err
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        weights = weights / np.sum(weights)
        return _worst_error(nodes, weights)
    
    # Phase 2: Local optimization from best initial point
    x0 = np.concatenate([best_nodes, best_weights])
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if time.time() < deadline and result.fun < best_err:
            best_nodes = result.x[:8]
            best_weights = result.x[8:] / np.sum(result.x[8:])
            best_err = result.fun
    except Exception:
        pass
    
    # Phase 3: Random restarts with fine-tuning
    np.random.seed(42)
    for i in range(20):
        if time.time() > deadline - 8:
            break
        
        # Smaller perturbations for fine-tuning
        nodes_try = best_nodes + np.random.uniform(-0.05, 0.05, 8)
        nodes_try = np.sort(nodes_try)
        nodes_try = np.clip(nodes_try, 0.005, 0.995)
        for j in range(1, 8):
            if nodes_try[j] <= nodes_try[j-1]:
                nodes_try[j] = nodes_try[j-1] + 1e-6
        
        weights_try = best_weights + np.random.uniform(-0.02, 0.02, 8)
        weights_try = np.maximum(weights_try, 1e-6)
        weights_try = weights_try / np.sum(weights_try)
        
        err = _worst_error(nodes_try, weights_try)
        if err < best_err:
            best_nodes = nodes_try.copy()
            best_weights = weights_try.copy()
            best_err = err
            
            if time.time() < deadline - 5:
                x0 = np.concatenate([best_nodes, best_weights])
                try:
                    result = minimize(_objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                    if time.time() < deadline and result.fun < best_err:
                        best_nodes = result.x[:8]
                        best_weights = result.x[8:] / np.sum(result.x[8:])
                        best_err = result.fun
                except Exception:
                    pass
    
    # Phase 4: Final fine-tuning with very small steps
    if time.time() < deadline - 3:
        for i in range(10):
            if time.time() > deadline - 2:
                break
            
            nodes_try = best_nodes + np.random.uniform(-0.01, 0.01, 8)
            nodes_try = np.sort(nodes_try)
            nodes_try = np.clip(nodes_try, 0.001, 0.999)
            for j in range(1, 8):
                if nodes_try[j] <= nodes_try[j-1]:
                    nodes_try[j] = nodes_try[j-1] + 1e-7
            
            weights_try = best_weights + np.random.uniform(-0.005, 0.005, 8)
            weights_try = np.maximum(weights_try, 1e-8)
            weights_try = weights_try / np.sum(weights_try)
            
            err = _worst_error(nodes_try, weights_try)
            if err < best_err:
                best_nodes = nodes_try.copy()
                best_weights = weights_try.copy()
                best_err = err
        
        # Final Nelder-Mead refinement
        if time.time() < deadline - 1:
            x0 = np.concatenate([best_nodes, best_weights])
            try:
                result = minimize(_objective, x0, method='Nelder-Mead', 
                                 options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if time.time() < deadline and result.fun < best_err:
                    best_nodes = result.x[:8]
                    best_weights = result.x[8:] / np.sum(result.x[8:])
                    best_err = result.fun
            except Exception:
                pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
