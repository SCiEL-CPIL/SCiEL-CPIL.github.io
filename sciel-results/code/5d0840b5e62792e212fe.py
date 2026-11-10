import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.linalg import solve

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Gauss-Legendre 8-point nodes and weights on [-1, 1] as initial guess
    t_gl = np.array([
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346425004484,
         0.1834346425004484,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    ])
    w_gl = np.array([
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    ])
    
    nodes_gl = 0.5 * t_gl + 0.5
    weights_gl = 0.5 * w_gl
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Best solution so far
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_obj = np.inf
    
    # CPL cosmology models: (w0, wa) pairs covering typical range
    cpl_models = [
        (0, 0),       # Lambda CDM
        (-1, 0),      # Cosmological constant
        (-1, 1),      # Quintessence-like
        (-0.5, 0),    # Mild dark energy
        (-1, -1),     # Phantom-ish
        (0.5, 0),     # Matter-dominated-ish
        (-1, 0.5),    # Evolving DE
        (-0.8, 0.2),  # Intermediate
        (-1, -0.5),   # Slightly phantom
        (-0.7, -0.3), # Mild evolution
        (-1, 1.5),    # Strong evolution
        (0, 1),       # Matter + evolving
        (-1.2, 0.3),  # Phantom with evolution
        (-0.9, -0.4), # Near -1 with negative evolution
        (-0.6, 0.4),  # Another intermediate
        (-1.1, -0.2), # Slightly phantom with negative evolution
    ]
    
    # Redshifts to test at
    z_values = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    def _E(a, w0, wa):
        """H(a)/H0 for CPL model. a = 1/(1+z)"""
        Om_m = 0.3
        Om_La = 1.0 - Om_m
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        term_m = Om_m * a**(-3.0)
        term_de = Om_La * a**(exponent_de)
        E_sq = term_m + term_de
        return np.sqrt(np.maximum(E_sq, 1e-30))
    
    def _true_integral(w0, wa, z):
        """Compute true integral using adaptive quadrature"""
        if z < 1e-10:
            return 0.0
        def integrand(u):
            a = 1.0 / (1.0 + z * u)
            return 1.0 / _E(a, w0, wa)
        result, _ = quad(integrand, 0, 1, limit=100)
        return (1.0 + z) * z * result
    
    def _quadrature_value(nodes, weights, w0, wa, z):
        """Compute quadrature approximation"""
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx
    
    def _objective(x):
        """Objective: minimize worst-case relative error"""
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        
        log_weights = x[8:16]
        w_max = np.max(log_weights)
        exp_w = np.exp(log_weights - w_max)
        weights = exp_w / np.sum(exp_w)
        
        if not np.all(nodes > 1e-15) or not np.all(nodes < 1 - 1e-15):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-15):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10
        
        max_rel_err = 0.0
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_integral(w0, wa, z)
                quad_val = _quadrature_value(nodes, weights, w0, wa, z)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    if rel_err > max_rel_err:
                        max_rel_err = rel_err
        return max_rel_err
    
    def _encode(nodes, weights):
        """Encode nodes and weights into parameter vector"""
        spacings = np.diff(nodes)
        log_spacings = np.log(spacings)
        log_weights = np.log(weights)
        return np.concatenate([log_spacings, log_weights])
    
    # Rational function approximation to guide node placement
    def _rational_fit_initial_guess():
        """
        Fit a degree-7/7 rational function to 1/E(u) averaged over cosmologies
        to get an initial guess for node placement.
        The idea: if we can approximate 1/E(u) ≈ P7(u)/Q7(u), then the zeros
        of Q7 give us natural cluster points for nodes.
        """
        # Sample u values in [0,1]
        u_grid = np.linspace(0.001, 0.999, 50)
        
        # For each u, compute average 1/E over cosmologies (at z=1 for simplicity)
        # Actually, the integrand depends on z through a = 1/(1+z*u)
        # Let's use a representative set of (w0,wa,z) combinations
        sample_combos = []
        for w0, wa in cpl_models[:8]:
            for z in [0.5, 1.0, 3.0]:
                sample_combos.append((w0, wa, z))
        
        # Build data: for each u, compute f(u) = mean of 1/E(1/(1+z*u)) over samples
        f_data = []
        for u in u_grid:
            vals = []
            for w0, wa, z in sample_combos:
                a = 1.0 / (1.0 + z * u)
                E_val = _E(a, w0, wa)
                vals.append(1.0 / E_val)
            f_data.append(np.mean(vals))
        f_data = np.array(f_data)
        
        # Fit rational function P7(u)/Q7(u) to f_data
        # Set up linear system: P7(u_i) - f_i * Q7(u_i) = 0
        # Normalize Q7(0) = 1
        # Unknowns: p0..p7 (8) and q1..q7 (7) = 15 unknowns
        # We have 50 equations, so overdetermined - use least squares
        
        # Build matrix A and vector b
        # For each u_i: sum_j p_j * u_i^j - f_i * sum_j q_j * u_i^j = 0
        # With q_0 = 1: sum_j p_j * u_i^j - f_i * (1 + sum_{j=1}^7 q_j * u_i^j) = 0
        # => sum_j p_j * u_i^j - f_i * sum_{j=1}^7 q_j * u_i^j = f_i
        
        A = np.zeros((len(u_grid), 15))
        b = np.zeros(len(u_grid))
        
        for i, u in enumerate(u_grid):
            # p coefficients
            for j in range(8):
                A[i, j] = u**j
            # q coefficients (q1..q7)
            for j in range(1, 8):
                A[i, 8 + (j-1)] = -f_data[i] * u**j
            b[i] = f_data[i]
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            p_coeffs = coeffs[:8]
            q_coeffs = np.concatenate([[1.0], coeffs[8:15]])
            
            # Find roots of Q7(u) = 0
            q_roots = np.roots(q_coeffs)
            # Keep real roots in (0,1)
            real_roots = [r.real for r in q_roots if abs(r.imag) < 1e-8 and 0 < r.real < 1]
            real_roots.sort()
            
            if len(real_roots) >= 4:
                # Use poles as initial node cluster centers
                # Interpolate to get 8 nodes
                if len(real_roots) >= 8:
                    # Take 8 evenly spaced from the real roots
                    indices = np.linspace(0, len(real_roots)-1, 8, dtype=int)
                    initial_nodes = np.array([real_roots[i] for i in indices])
                else:
                    # Mix poles with GL nodes
                    n_poles = len(real_roots)
                    n_gl = 8 - n_poles
                    gl_subset = nodes_gl[:n_gl]
                    initial_nodes = np.concatenate([real_roots, gl_subset])
                    initial_nodes.sort()
                
                # Ensure strictly increasing and in (0,1)
                if np.all(np.diff(initial_nodes) > 1e-6) and np.all(initial_nodes > 1e-6) and np.all(initial_nodes < 1-1e-6):
                    # Use GL weights as initial weights
                    initial_weights = weights_gl.copy()
                    return initial_nodes, initial_weights
            else:
                # Fallback: use GL nodes but slightly perturbed based on pole locations
                return nodes_gl.copy(), weights_gl.copy()
                
        except Exception:
            return nodes_gl.copy(), weights_gl.copy()
    
    # Get initial guess from rational fit
    initial_nodes, initial_weights = _rational_fit_initial_guess()
    
    # If rational fit gave something different, use it; otherwise use GL
    if not np.allclose(initial_nodes, nodes_gl, atol=1e-10):
        best_nodes = initial_nodes.copy()
        best_weights = initial_weights.copy()
    
    x0 = _encode(best_nodes, best_weights)
    
    # Phase 1: L-BFGS-B local refinement from rational-fit initial point
    if time.time() - start_time < deadline - 20:
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                log_spacings = res.x[:8]
                spacings = np.exp(log_spacings)
                total = np.sum(spacings)
                candidate_nodes = np.cumsum(spacings) / total
                log_weights = res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Phase 2: Differential evolution for global search
    if time.time() - start_time < deadline - 15:
        try:
            bounds = [(-6, 6)] * 8 + [(-6, 6)] * 8
            de_res = differential_evolution(_objective, bounds, 
                                           maxiter=80, popsize=25, 
                                           tol=1e-8, seed=42, 
                                           polish=False,
                                           init='latinhypercube')
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                log_spacings = de_res.x[:8]
                spacings = np.exp(log_spacings)
                total = np.sum(spacings)
                candidate_nodes = np.cumsum(spacings) / total
                log_weights = de_res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Phase 3: Final local refinement with Nelder-Mead
    if time.time() - start_time < deadline - 5:
        try:
            x_current = _encode(best_nodes, best_weights)
            nm_res = minimize(_objective, x_current, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if nm_res.fun < best_obj:
                best_obj = nm_res.fun
                log_spacings = nm_res.x[:8]
                spacings = np.exp(log_spacings)
                total = np.sum(spacings)
                candidate_nodes = np.cumsum(spacings) / total
                log_weights = nm_res.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Phase 4: One more L-BFGS-B polish
    if time.time() - start_time < deadline - 2:
        try:
            x_current = _encode(best_nodes, best_weights)
            res2 = minimize(_objective, x_current, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
            if res2.fun < best_obj:
                best_obj = res2.fun
                log_spacings = res2.x[:8]
                spacings = np.exp(log_spacings)
                total = np.sum(spacings)
                candidate_nodes = np.cumsum(spacings) / total
                log_weights = res2.x[8:16]
                w_max = np.max(log_weights)
                exp_w = np.exp(log_weights - w_max)
                candidate_weights = exp_w / np.sum(exp_w)
                
                if np.all(candidate_nodes > 1e-15) and np.all(candidate_nodes < 1 - 1e-15) and \
                   np.all(np.diff(candidate_nodes) > 1e-15) and np.all(candidate_weights > 0):
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception:
            pass
    
    # Final validation and normalization
    best_weights = best_weights / np.sum(best_weights)
    
    # Verify all constraints
    if not (len(best_nodes) == 8 and len(best_weights) == 8 and
            np.all(best_nodes > 0) and np.all(best_nodes < 1) and
            np.all(np.diff(best_nodes) > 0) and
            np.all(best_weights > 0) and
            abs(np.sum(best_weights) - 1.0) < 1e-12):
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
