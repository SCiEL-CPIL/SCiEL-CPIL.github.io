import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E(a, w0, w1, Omega_m):
    """E(a) = H(a)/H0 for CPL dark energy with flat spatial geometry."""
    # For flat universe: E^2(a) = Omega_m/a^3 + Omega_de(a)
    # With CPL: w(z) = w0 + w1*z
    # Omega_de(a) = (1-Omega_m) * a^{-3(1+w0)} * exp(3*w1*(1-a)/a)
    Omega_de = 1.0 - Omega_m
    # Handle a=0 edge case
    a_safe = np.maximum(a, 1e-15)
    de_term = Omega_de * a_safe**(-3.0*(1.0+w0)) * np.exp(3.0*w1*(1.0-a_safe)/a_safe)
    m_term = Omega_m * a_safe**(-3.0)
    return np.sqrt(m_term + de_term)

def _integrand_z(z, w0, w1, Omega_m):
    """Integrand for luminosity distance in z-space: (1+z)/E(1/(1+z))"""
    if z < 0:
        return 0.0
    a = 1.0 / (1.0 + z)
    return (1.0 + z) / _E(a, w0, w1, Omega_m)

def _true_distance(z, w0, w1, Omega_m, n_points=5000):
    """Compute true luminosity distance via adaptive quadrature."""
    if z <= 0:
        return 0.0
    val, _ = quad(lambda zz: _integrand_z(zz, w0, w1, Omega_m), 0, z, limit=200, epsabs=1e-12, epsrel=1e-10)
    return val

def _quad_distance(z, nodes, weights, w0, w1, Omega_m):
    """Compute quadrature approximation of luminosity distance."""
    if z <= 0:
        return 0.0
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1, Omega_m)
    return z * np.sum(weights * vals)

def _relative_error(z, nodes, weights, w0, w1, Omega_m):
    """Compute relative error in percent."""
    true = _true_distance(z, w0, w1, Omega_m)
    if true == 0:
        return 0.0
    approx = _quad_distance(z, nodes, weights, w0, w1, Omega_m)
    return abs(approx - true) / true * 100.0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin before 90s kill
    
    # Test cosmologies: (w0, w1, Omega_m) triples
    # Include diverse dark energy models and matter densities
    test_cosmos = [
        (-1.0, 0.0, 0.3),      # LambdaCDM standard
        (-1.0, 0.5, 0.3),      # CPL with w1>0
        (-1.0, -0.5, 0.3),     # CPL with w1<0
        (-0.9, 0.3, 0.3),      # Quintessence-like
        (-0.95, -0.3, 0.3),    # Another variant
        (-1.2, 0.0, 0.3),      # Phantom
        (-0.8, 0.5, 0.3),      # Strong evolution
        (-1.0, 1.0, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.2),      # Lower matter density
        (-1.0, 0.0, 0.4),      # Higher matter density
        (-0.95, 0.5, 0.25),    # Mixed
        (-1.1, -0.5, 0.35),    # Phantom with evolution
    ]
    
    # Redshifts to test
    test_zs = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    # Precompute true distances for all test cases
    print(f"Precomputing true distances...")
    true_dists = {}
    for (w0, w1, Om) in test_cosmos:
        for z in test_zs:
            true_dists[(w0, w1, Om, z)] = _true_distance(z, w0, w1, Om)
    print(f"Precomputation done at {time.time()-start_time:.1f}s")
    
    # Start with Gauss-Legendre as initial guess
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    
    def _worst_error_fast(nodes, weights):
        """Compute worst relative error using precomputed true distances."""
        max_err = 0.0
        for (w0, w1, Om) in test_cosmos:
            for z in test_zs:
                true = true_dists[(w0, w1, Om, z)]
                if true == 0:
                    continue
                approx = _quad_distance(z, nodes, weights, w0, w1, Om)
                err = abs(approx - true) / true * 100.0
                if err > max_err:
                    max_err = err
        return max_err
    
    best_err = _worst_error_fast(best_nodes, best_weights)
    print(f"Initial GL error: {best_err:.6f}% at {time.time()-start_time:.1f}s")
    
    # Objective function with normalization
    def _objective_norm(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check node constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return _worst_error_fast(nodes, weights)
    
    # Initial params
    x0 = np.concatenate([nodes_init, weights_init])
    
    # Try Nelder-Mead from GL start
    try:
        result = minimize(_objective_norm, x0, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': 1000})
        if time.time() < deadline and result.fun < best_err:
            best_nodes = result.x[:8]
            best_weights = result.x[8:] / np.sum(result.x[8:])
            best_err = result.fun
            print(f"After NM1: {best_err:.6f}% at {time.time()-start_time:.1f}s")
    except Exception as e:
        print(f"NM1 failed: {e}")
    
    # Try multiple random restarts
    np.random.seed(42)
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation of GL nodes
        nodes_try = nodes_init + np.random.uniform(-0.1, 0.1, 8)
        nodes_try = np.sort(nodes_try)
        nodes_try = np.clip(nodes_try, 0.005, 0.995)
        # Ensure strict increase
        for j in range(1, 8):
            if nodes_try[j] <= nodes_try[j-1]:
                nodes_try[j] = nodes_try[j-1] + 1e-5
        
        weights_try = weights_init + np.random.uniform(-0.05, 0.05, 8)
        weights_try = np.maximum(weights_try, 1e-5)
        weights_try = weights_try / np.sum(weights_try)
        
        x0_try = np.concatenate([nodes_try, weights_try])
        
        try:
            result = minimize(_objective_norm, x0_try, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': 500})
            if time.time() < deadline and result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
                print(f"Restart {i+1}: {best_err:.6f}% at {time.time()-start_time:.1f}s")
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result = minimize(_objective_norm, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 2000})
            if time.time() < deadline and result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
                print(f"After polish: {best_err:.6f}% at {time.time()-start_time:.1f}s")
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strict increase
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-10
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    print(f"Final error: {best_err:.6f}% at {time.time()-start_time:.1f}s")
    print(f"Nodes: {best_nodes}")
    print(f"Weights: {best_weights}")
    
    return {"nodes": best_nodes, "weights": best_weights}
