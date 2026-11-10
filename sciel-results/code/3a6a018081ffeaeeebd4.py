import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    
    # Define cosmology models
    # Omega_m, Omega_L, w0, wa
    models = [
        (0.3, 0.7, -1.0, 0.0),   # LCDM
        (0.3, 0.7, -1.0, 0.3),   # CPL
        (0.3, 0.7, -0.8, -0.5),  # CPL
        (0.3, 0.7, -1.2, 0.2),   # CPL
        (0.3, 0.7, -0.5, 0.0),   # wCDM
    ]
    
    # Test z values
    z_vals = np.linspace(0.01, 10.0, 50)
    
    def E_of_z(z, Om, OL, w0, wa):
        if wa == 0 and w0 == -1:
            return np.sqrt(Om * (1+z)**3 + OL)
        else:
            # CPL
            # Integral of (1+w(z))/(1+z) from 0 to z
            # w(z) = w0 + wa * z/(1+z)
            # (1+w)/(1+z) = (1+w0)/(1+z) + wa/(1+z)^2
            # Int = (1+w0) ln(1+z) + wa * (1 - 1/(1+z)) = (1+w0) ln(1+z) + wa * z/(1+z)
            # Wait, integral of 1/(1+z)^2 is -1/(1+z). From 0 to z: -1/(1+z) - (-1) = 1 - 1/(1+z) = z/(1+z).
            # So exponent is 3 * [ (1+w0) ln(1+z) + wa * z/(1+z) ]
            # E^2 = Om (1+z)^3 + OL * exp(3 * [ (1+w0) ln(1+z) + wa * z/(1+z) ])
            # = Om (1+z)^3 + OL * (1+z)^(3(1+w0)) * exp(3 * wa * z/(1+z))
            term1 = Om * (1+z)**3
            term2 = OL * (1+z)**(3*(1+w0)) * np.exp(3*wa*z/(1+z))
            return np.sqrt(term1 + term2)
            
    def true_distance(z, Om, OL, w0, wa):
        # (1+z) * integral_0^z dz'/E(z')
        # Use quad for accuracy in optimization
        if z == 0: return 0
        integrand = lambda zp: 1.0 / E_of_z(zp, Om, OL, w0, wa)
        val, _ = quad(integrand, 0, z, limit=50)
        return (1+z) * val

    def quadrature_distance(z, nodes, weights, Om, OL, w0, wa):
        # z * sum w_i / E(z * u_i)
        if z == 0: return 0
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1+z) * z * integral_approx

    def objective(params):
        # params: [u1, d1, d2, ..., d7, w1, ..., w8]
        # u1 > 0, di > 0
        # u_{i+1} = u_i + d_i
        # u8 < 1
        u1 = params[0]
        ds = params[1:8]
        ws = params[8:16]
        
        # Reconstruct nodes
        nodes = np.zeros(8)
        nodes[0] = u1
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + ds[i-1]
            
        # Check bounds
        if nodes[-1] >= 1.0 or nodes[0] <= 0:
            return 1e6
            
        # Normalize weights
        w_sum = np.sum(ws)
        if w_sum <= 0:
            return 1e6
        ws = ws / w_sum
        
        max_err = 0
        for z in z_vals:
            for (Om, OL, w0, wa) in models:
                d_true = true_distance(z, Om, OL, w0, wa)
                d_quad = quadrature_distance(z, nodes, ws, Om, OL, w0, wa)
                if d_true == 0: continue
                rel_err = abs(d_quad - d_true) / d_true
                if rel_err > max_err:
                    max_err = rel_err
                    # Early exit if too large?
                    if max_err > 0.5:
                        break
            if max_err > 0.5:
                break
                
        return max_err

    # Initial guess: Gauss-Legendre
    # Generate GL nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    # Map to (0,1)
    init_nodes = (gl_nodes + 1) / 2
    init_weights = gl_weights / 2
    
    # Convert to params format
    u1 = init_nodes[0]
    ds = np.diff(init_nodes)
    ws = init_weights
    
    init_params = np.concatenate(([u1], ds, ws))
    
    # Bounds
    # u1 > 1e-6
    # ds > 1e-6
    # ws > 1e-6
    bounds = []
    bounds.append((1e-6, 1.0)) # u1
    for i in range(7):
        bounds.append((1e-6, 1.0)) # ds
    for i in range(8):
        bounds.append((1e-6, 1.0)) # ws
        
    # Optimize
    best_params = init_params
    best_score = objective(init_params)
    
    # Use L-BFGS-B
    # Time limit 90s. Leave margin.
    deadline = start_time + 60.0
    
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'maxfun': 10000})
        if res.success or res.fun < best_score:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        # Return best found so far
        pass
    else:
        # Try a few more random restarts if time permits?
        # Or just refine with Nelder-Mead?
        # L-BFGS-B is usually good.
        
        # Let's do a quick local search with Nelder-Mead if time allows
        # But Nelder-Mead doesn't handle bounds well.
        # Stick with L-BFGS-B result.
        
        # Try one more optimization with a different starting point?
        # Midpoint rule
        mid_nodes = (np.arange(8) + 0.5) / 8
        mid_weights = np.full(8, 1/8)
        mid_params = np.concatenate(([mid_nodes[0]], np.diff(mid_nodes), mid_weights))
        
        if time.time() < deadline - 5:
            try:
                res2 = minimize(objective, mid_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x
            except Exception:
                pass

    # Extract final nodes and weights
    u1 = best_params[0]
    ds = best_params[1:8]
    ws = best_params[8:16]
    
    nodes = np.zeros(8)
    nodes[0] = u1
    for i in range(1, 8):
        nodes[i] = nodes[i-1] + ds[i-1]
        
    # Ensure strict increasing and in [0,1]
    # Clip and normalize if necessary, but optimizer should keep it valid.
    # Re-normalize weights to sum exactly 1
    ws = ws / np.sum(ws)
    
    return {"nodes": nodes, "weights": ws}
