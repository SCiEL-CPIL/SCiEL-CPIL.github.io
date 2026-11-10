import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # 1. Define the Cosmological Models and Redshifts
    # We use a diverse set of CPL (w0, wa) models to ensure robustness.
    # The evaluator uses disjoint validation sets, so we must cover a wide range.
    cpl_models = [
        (0.0, 0.0),    # Matter only (unrealistic but boundary)
        (-1.0, 0.0),   # Lambda CDM
        (-1.0, 1.0),   # Quintessence
        (-1.0, -1.0),  # Phantom
        (-0.5, 0.0),   # Mild DE
        (-0.9, 0.5),   # Evolving DE
        (-1.5, 0.5),   # Strong Phantom
        (-0.8, -0.2),  # Near Lambda
        (0.5, 1.0),    # Matter-like + evolving
        (-1.0, 2.0),   # Strong evolution
        (-2.0, -0.5),  # Deep Phantom
        (0.0, 1.0),    # Matter + evolution
    ]
    
    # Redshifts covering low to high z
    z_values = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])

    def _E(a, w0, wa):
        """
        H(a)/H0 for flat CPL model.
        a = 1/(1+z)
        E^2(a) = Om_m * a^-3 + Om_La * a^(3*(1+w0+wa*(1-a)))
        Om_m = 0.3, Om_La = 0.7
        """
        Om_m = 0.3
        Om_La = 0.7
        # Exponent for Dark Energy term
        exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
        
        # Handle a=0 or a=1 carefully to avoid inf/nan
        # a is in (0, 1]
        a_safe = np.clip(a, 1e-10, 1.0)
        
        term_m = Om_m * a_safe**(-3.0)
        term_de = Om_La * a_safe**(exponent_de)
        
        E_sq = term_m + term_de
        # Avoid sqrt of negative due to numerical issues, though E_sq should be positive
        return np.sqrt(np.maximum(E_sq, 1e-30))

    def _true_integral(w0, wa, z):
        """
        Compute the true integral value using high-precision adaptive quadrature.
        dL/(c/H0) = (1+z) * z * \int_0^1 du / E(1/(1+z*u))
        """
        if z < 1e-6:
            # Limit as z->0 is small, use series or just return 0 approx for error calc
            # Actually dL ~ H0^-1 * z for small z? No, dL = c/H0 * (1+z) \int_0^z dz'/E(z')
            # \int_0^z dz'/E(z') ~ z/E(0) = z. So dL ~ (1+z)z.
            # For error calculation, we just need a reference.
            pass
        
        # Use scipy.integrate.quad for high precision reference
        from scipy.integrate import quad
        
        def integrand(u):
            a = 1.0 / (1.0 + z * u)
            E_val = _E(np.array([a]), w0, wa)[0]
            return 1.0 / E_val
            
        # Integrate from 0 to 1
        integral_val, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
        return (1.0 + z) * z * integral_val

    def _quadrature_value(nodes, weights, w0, wa, z):
        """
        Compute the quadrature approximation.
        """
        a = 1.0 / (1.0 + z * nodes)
        E_vals = _E(a, w0, wa)
        integral_approx = np.dot(weights, 1.0 / E_vals)
        return (1.0 + z) * z * integral_approx

    def _compute_errors(nodes, weights):
        """
        Compute relative errors for all model-redshift pairs.
        Returns a vector of relative errors.
        """
        errors = []
        for w0, wa in cpl_models:
            for z in z_values:
                true_val = _true_integral(w0, wa, z)
                quad_val = _quadrature_value(nodes, weights, w0, wa, z)
                if true_val > 1e-10:
                    rel_err = abs(quad_val - true_val) / true_val
                    errors.append(rel_err)
        return np.array(errors)

    def _objective_worst(x):
        """
        Objective function for optimization: Maximize the negative of the worst-case error?
        No, minimize the worst-case error.
        However, max() is not smooth. We use a soft-max approximation or just max for DE.
        For L-BFGS-B, we use sum of squares to get gradients.
        """
        nodes, weights = _decode(x)
        if not _is_valid(nodes, weights):
            return 1e10
        
        errors = _compute_errors(nodes, weights)
        return np.max(errors)

    def _objective_smooth(x):
        """
        Smooth objective for gradient-based methods: Sum of squared errors.
        This helps find a local minimum quickly.
        """
        nodes, weights = _decode(x)
        if not _is_valid(nodes, weights):
            return 1e10
        
        errors = _compute_errors(nodes, weights)
        return np.sum(errors**2)

    def _decode(x):
        """
        Decode parameters x into nodes and weights.
        x = [8 log-spacings, 8 log-weights]
        """
        log_spacings = x[:8]
        spacings = np.exp(log_spacings)
        total = np.sum(spacings)
        nodes = np.cumsum(spacings) / total
        
        log_weights = x[8:16]
        # Normalize weights
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        return nodes, weights

    def _encode(nodes, weights):
        """
        Encode nodes and weights into parameter vector x.
        """
        spacings = np.diff(nodes)
        # Add small epsilon to avoid log(0)
        spacings = np.maximum(spacings, 1e-15)
        log_spacings = np.log(spacings)
        
        log_weights = np.log(np.maximum(weights, 1e-15))
        
        return np.concatenate([log_spacings, log_weights])

    def _is_valid(nodes, weights):
        """
        Check if nodes and weights satisfy constraints.
        """
        if len(nodes) != 8 or len(weights) != 8:
            return False
        if not np.all(nodes > 1e-8) or not np.all(nodes < 1 - 1e-8):
            return False
        if not np.all(np.diff(nodes) > 1e-8):
            return False
        if not np.all(weights > 0):
            return False
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return False
        return True

    # --- Optimization Strategy ---
    
    # 1. Initial Guess: Gauss-Legendre
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
    
    best_nodes = nodes_gl.copy()
    best_weights = weights_gl.copy()
    best_worst_err = _objective_worst(_encode(best_nodes, best_weights))
    
    # 2. Global Search with Differential Evolution
    # Bounds for log-spacings and log-weights
    # Spacings sum to 1, so individual spacings are roughly 1/8 = 0.125, log ~ -2
    # Range [-5, 2] covers 0.006 to 7.4
    bounds = [(-5, 2)] * 8 + [(-5, 2)] * 8
    
    if time.time() - start_time < deadline - 20:
        try:
            # Use DE to find a good global candidate
            de_res = differential_evolution(
                _objective_worst, 
                bounds, 
                maxiter=30, 
                popsize=30, 
                tol=1e-4, 
                seed=42, 
                polish=False,
                init='latinhypercube',
                updateinterval=10
            )
            
            candidate_nodes, candidate_weights = _decode(de_res.x)
            if _is_valid(candidate_nodes, candidate_weights):
                err = _objective_worst(de_res.x)
                if err < best_worst_err:
                    best_worst_err = err
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception as e:
            pass

    # 3. Local Refinement with L-BFGS-B
    # Start from the best candidate found so far
    x0 = _encode(best_nodes, best_weights)
    
    if time.time() - start_time < deadline - 10:
        try:
            # Use smooth objective (L2) for gradient steps
            res_lbfgs = minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            
            candidate_nodes, candidate_weights = _decode(res_lbfgs.x)
            if _is_valid(candidate_nodes, candidate_weights):
                # Check worst-case error for this smooth-optimum
                err_worst = _objective_worst(res_lbfgs.x)
                if err_worst < best_worst_err:
                    best_worst_err = err_worst
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception as e:
            pass

    # 4. Final Polish with Nelder-Mead on the Worst-Case Objective
    # Start from the L-BFGS result
    x0_nm = _encode(best_nodes, best_weights)
    
    if time.time() - start_time < deadline - 2:
        try:
            res_nm = minimize(
                _objective_worst, 
                x0_nm, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            candidate_nodes, candidate_weights = _decode(res_nm.x)
            if _is_valid(candidate_nodes, candidate_weights):
                err_worst = res_nm.fun
                if err_worst < best_worst_err:
                    best_worst_err = err_worst
                    best_nodes = candidate_nodes
                    best_weights = candidate_weights
        except Exception as e:
            pass

    # Final Validation and Normalization
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strict constraints
    if not (len(best_nodes) == 8 and len(best_weights) == 8 and
            np.all(best_nodes > 1e-10) and np.all(best_nodes < 1 - 1e-10) and
            np.all(np.diff(best_nodes) > 1e-10) and
            np.all(best_weights > 1e-10) and
            abs(np.sum(best_weights) - 1.0) < 1e-12):
        # Fallback to GL
        best_nodes = nodes_gl.copy()
        best_weights = weights_gl.copy()
        best_weights = best_weights / np.sum(best_weights)
        
    return {"nodes": best_nodes, "weights": best_weights}
