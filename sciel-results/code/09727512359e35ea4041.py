"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard: 90s wall-clock limit, aim to finish well before.
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # 1. Define representative cosmologies for optimization
    # We need to cover the space of "spatially flat, time-varying dark-energy models".
    # Common parameterizations:
    # E(z) = sqrt(Om_m * (1+z)^3 + Om_r * (1+z)^4 + Om_L * E_L(z)^2)
    # For simplicity and speed, we'll focus on Omega_m and w0/wa (CPL) or simple power laws.
    # The task mentions "disjoint public validation cosmologies", implying a specific set.
    # Since we don't know the exact set, we optimize against a broad, representative set.
    
    # Representative models: (Omega_m, w0, wa) for CPL: E_L(z) = (1+z)^(3(1+w0+wa)/2) ... wait, CPL is usually:
    # E(z) = H(z)/H0 = sqrt(Om_m(1+z)^3 + Om_L * (1+z)^(3(1+w0+wa)) * exp(-3wa*z/(1+z)))
    # Or simpler: E(z)^2 = Om_m(1+z)^3 + Om_L(1+z)^{3(1+w)}
    
    # Let's use a set of (Om_m, w) pairs covering reasonable ranges.
    # Om_m in [0.2, 0.4], w in [-1.5, 0.0]
    # Also include standard LCDM (Om_m=0.3, w=-1)
    
    models = []
    om_m_vals = [0.2, 0.3, 0.4]
    w_vals = [-1.5, -1.2, -1.0, -0.8, -0.5]
    
    for om_m in om_m_vals:
        om_l = 1.0 - om_m
        for w in w_vals:
            # Define E(z) for this model
            def _E(z, om_m=om_m, om_l=om_l, w=w):
                # E(z)^2 = Om_m(1+z)^3 + Om_L(1+z)^(3(1+w))
                # Note: if w=-1, (1+z)^0 = 1
                term_m = om_m * (1+z)**3
                term_l = om_l * (1+z)**(3*(1+w))
                return np.sqrt(term_m + term_l)
            models.append(_E)
            
    # Add some extreme cases to ensure robustness
    # Phantom energy w < -1
    models.append(lambda z: np.sqrt(0.3*(1+z)**3 + 0.7*(1+z)**(3*(1-1.2)))) # w=-1.2, Om_m=0.3
    # Quintessence w > -1
    models.append(lambda z: np.sqrt(0.3*(1+z)**3 + 0.7*(1+z)**(3*(1-(-0.5))))) # w=-0.5, Om_m=0.3

    # 2. Define the integrand for luminosity distance
    # dL/(c/H0) = (1+z) * z * integral_0^z 1/E(z') dz'
    # The task formula: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i) where a_i = 1/(1+z*nodes[i])
    # Wait, the task says: sum_i weights[i]/E(a_i).
    # Let's check the variable change.
    # Standard: d_L = (c/H0) (1+z) int_0^z dz'/E(z')
    # The task formula uses a_i = 1/(1+z*nodes[i]).
    # If nodes are in [0,1], let u = nodes[i].
    # Then a_i = 1/(1+z*u).
    # The integral is over z' from 0 to z.
    # Let's assume the quadrature is approximating int_0^1 du / E(z(u)) or similar?
    # The task says: "sum_i weights[i]/E(a_i)".
    # If we change variable to a, da = -dz/(1+z)^2.
    # int_0^z dz'/E(z') = int_1^(1/(1+z)) da / (E(a) * a^2 * (1/a - 1)) ... this is getting complicated.
    # Let's stick to the task's explicit formula:
    # Approx = (1+z)*z * sum_i weights[i] / E(a_i)
    # where a_i = 1/(1+z*nodes[i]).
    # This implies the nodes are in the variable 'u' such that a = 1/(1+zu).
    # When u=0, a=1. When u=1, a=1/(1+z).
    # The integral being approximated is likely int_0^1 du / E(1/(1+zu)) ?
    # Let's verify the exact integral for a specific model to calibrate the weights.
    
    def exact_dL(z, E_func):
        # Integrate 1/E(z') from 0 to z using high precision quadrature
        # Use scipy.integrate.quad
        from scipy.integrate import quad
        integrand = lambda zp: 1.0 / E_func(zp)
        val, _ = quad(integrand, 0, z, limit=100)
        return (1+z) * z * val

    def quad_dL(z, E_func, nodes, weights):
        # Compute a_i
        a_i = 1.0 / (1.0 + z * nodes)
        # Sum weights / E(a_i)
        s = np.sum(weights / E_func(a_i))
        return (1+z) * z * s

    # 3. Optimization Objective
    # We want to minimize the worst-case relative error across models and z values.
    # Test z values: [0.1, 0.5, 1.0, 2.0, 5.0]
    z_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    
    # Initial guess: Gauss-Legendre nodes and weights for [0,1]
    # GL nodes are in [-1,1], transform to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (x_gl + 1.0)
    weights_init = 0.5 * w_gl
    
    # Ensure nodes are strictly increasing
    nodes_init = np.sort(nodes_init)
    
    # Ensure weights are positive (GL weights are positive)
    
    def objective(params):
        # params: 16 values (8 nodes, 8 weights)
        # We need to enforce constraints: nodes in [0,1] strictly increasing, weights > 0, sum(weights)=1.
        # To handle constraints in optimization, we can use a transformation or penalty.
        # Simpler: optimize nodes and raw weights, then normalize weights and check constraints.
        # If constraints violated, return large penalty.
        
        nodes = params[:8]
        weights_raw = params[8:]
        
        # Check nodes: strictly increasing in [0,1]
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
            
        # Weights: must be positive and sum to 1.
        # Let's use softmax or just normalize if positive.
        # If any weight <= 0, penalty.
        if np.any(weights_raw <= 0):
            return 1e6
        
        weights = weights_raw / np.sum(weights_raw)
        
        # Compute worst-case relative error
        max_err = 0.0
        for E_func in models:
            for z in z_test:
                exact = exact_dL(z, E_func)
                approx = quad_dL(z, E_func, nodes, weights)
                if exact == 0:
                    continue
                rel_err = abs((approx - exact) / exact)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # 4. Optimization Loop
    # Use scipy.optimize.minimize with Nelder-Mead or Powell (derivative-free, handles noise)
    # Since we have 16 parameters, Nelder-Mead might be slow. Powell is often better for this size.
    # We have ~15 models * 5 z-values = 75 evaluations per objective call.
    # Each evaluation involves a quad integration (expensive).
    # We need to be careful with the time budget.
    
    # Pre-compute exact values to speed up objective?
    # The models are fixed, z_test is fixed.
    # We can pre-compute exact_dL for all (model, z) pairs.
    
    print("Pre-computing exact values...")
    exact_vals = np.zeros((len(models), len(z_test)))
    for i, E_func in enumerate(models):
        for j, z in enumerate(z_test):
            exact_vals[i, j] = exact_dL(z, E_func)
    print("Pre-computation done.")
    
    def objective_fast(params):
        nodes = params[:8]
        weights_raw = params[8:]
        
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(weights_raw <= 0):
            return 1e6
            
        weights = weights_raw / np.sum(weights_raw)
        
        max_err = 0.0
        for i, E_func in enumerate(models):
            a_i = 1.0 / (1.0 + z_test * nodes) # shape (8, 5) -> nodes (8,), z_test (5,)
            # E_func takes array
            E_vals = E_func(a_i) # shape (8, 5)
            # Sum weights / E_vals over axis 0
            # weights (8,), E_vals (8, 5) -> weights[:, None] / E_vals -> sum axis 0 -> (5,)
            sums = np.sum(weights[:, None] / E_vals, axis=0)
            approxs = (1+z_test) * z_test * sums
            errs = np.abs((approxs - exact_vals[i]) / exact_vals[i])
            max_err = max(max_err, np.max(errs))
        return max_err

    # Start with GL nodes/weights
    x0 = np.concatenate([nodes_init, weights_init])
    
    # Optimization
    best_params = x0.copy()
    best_score = objective_fast(x0)
    
    # Time check
    if time.time() > deadline:
        return {"nodes": best_params[:8], "weights": best_params[8:]/np.sum(best_params[8:])}
    
    # Use Nelder-Mead with maxiter limit
    # Or Powell
    try:
        from scipy.optimize import minimize
        # Powell is good for this
        res = minimize(objective_fast, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8})
        
        if res.success and res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial if optimization fails or takes too long?
        # The loop above has maxiter, so it should finish.
        
    # Final check and formatting
    nodes_final = best_params[:8]
    weights_final = best_params[8:]
    
    # Ensure strict constraints for the final output
    # Nodes strictly increasing in [0,1]
    # If optimization pushed them out, we might need to clamp or revert.
    # Given the penalty, they should be inside.
    
    # Normalize weights
    weights_final = weights_final / np.sum(weights_final)
    
    # Double check positivity
    if np.any(weights_final <= 0) or np.any(nodes_final <= 0) or np.any(nodes_final >= 1) or np.any(np.diff(nodes_final) <= 0):
        # Revert to initial GL if constraints broken
        nodes_final = nodes_init
        weights_final = weights_init
        weights_final = weights_final / np.sum(weights_final)
        
    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
