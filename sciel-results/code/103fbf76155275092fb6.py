"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Deadline guard: return best result found before 90s wall-clock limit
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin
    
    # We need to find an 8-node positive quadrature rule for the integral
    # I(z) = z * sum_i w_i / E(a_i)  where a_i = 1/(1+z*t_i), t_i in [0,1]
    # Actually the integral is: dL/(c/H0) = (1+z) * integral_0^z dz'/E(a')
    # where E(a) = H(a)/H0. Let's re-read the task.
    #
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # Wait, that's not right. Let me re-read.
    #
    # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
    # Hmm, that seems odd. Let me think about the luminosity distance formula.
    #
    # In a flat universe, the comoving distance is:
    # D_C = (c/H0) * integral_0^z dz'/E(z')
    # where E(z) = H(z)/H0.
    #
    # The luminosity distance is:
    # D_L = (1+z) * D_C = (c/H0) * (1+z) * integral_0^z dz'/E(z')
    #
    # So dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    #
    # The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # That doesn't match. Let me re-read more carefully.
    #
    # "The evaluator computes integrands itself. Minimize worst relative validation error in percent; exactly eight integrand samples per distance."
    # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i]) and E(a)=H(a)/H0"
    #
    # Wait, I think there might be a change of variables. Let me think...
    #
    # If we change variables from z' to a' = 1/(1+z'), then when z'=0, a'=1, and when z'=z, a'=1/(1+z).
    # da'/dz' = -1/(1+z')^2 = -a'^2
    # So dz' = -da'/a'^2
    #
    # integral_0^z dz'/E(z') = integral_1^{1/(1+z)} (-da'/a'^2) / E(1/a' - 1)
    # = integral_{1/(1+z)}^1 da'/(a'^2 * E(1/a' - 1))
    #
    # Hmm, that's getting complicated. Let me re-read the task statement again.
    #
    # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    #
    # I think the formula is:
    # D_L/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    #
    # And they're using a quadrature rule on the integral. The nodes t_i are in [0,1],
    # and the actual z' values are z*t_i. So a_i = 1/(1+z*t_i).
    #
    # The integral integral_0^z dz'/E(z') can be approximated as:
    # sum_i w_i * z / E(z*t_i)  ... no wait, that's not right either.
    #
    # Actually, if we use a quadrature rule on [0,1] with nodes t_i and weights w_i,
    # then integral_0^1 f(t) dt ≈ sum_i w_i f(t_i)
    #
    # So integral_0^z dz'/E(z') = z * integral_0^1 f(u) du where f(u) = 1/E(z*u)
    # ≈ z * sum_i w_i / E(z*t_i)
    #
    # So D_L/(c/H0) = (1+z) * z * sum_i w_i / E(z*t_i)
    #
    # And a_i = 1/(1+z*t_i), so E(a_i) = E(1/(1+z*t_i)) = H(a_i)/H0.
    # But E(z*t_i) = H(z*t_i)/H0, not H(a_i)/H0.
    #
    # Wait, I'm confusing myself. Let me re-read: "a_i=1/(1+z*nodes[i]) and E(a)=H(a)/H0"
    #
    # So E(a) is defined as H(a)/H0 where a is the scale factor. But in the integral,
    # we have E(z') = H(z')/H0 where z' is redshift. These are different!
    #
    # Actually, in cosmology, E(a) = H(a)/H0 where a is the scale factor, and
    # E(z) = H(z)/H0 where z is redshift, and a = 1/(1+z).
    #
    # So E(z') = H(z')/H0 and E(a) = H(a)/H0 where a = 1/(1+z').
    # So E(z') = E(a) where a = 1/(1+z').
    #
    # So the integral integral_0^z dz'/E(z') = integral_0^z dz' * H0/H(z')
    #
    # And H(z') = H0 * E(z') where E(z') is the usual Hubble parameter normalized.
    #
    # I think the task is using E(a) to mean the Hubble parameter at scale factor a,
    # normalized by H0. So E(a) = H(a)/H0.
    #
    # And a_i = 1/(1+z*nodes[i]), so E(a_i) = H(a_i)/H0.
    #
    # But in the integral, we need H(z')/H0 where z' is redshift. And H(z') = H(1/(1+z')).
    # So H(z')/H0 = H(1/(1+z'))/H0 = E(1/(1+z')).
    #
    # So the integral is integral_0^z dz' / E(1/(1+z'))
    #
    # And with the change of variables u = z'/z, dz' = z du:
    # = z * integral_0^1 du / E(1/(1+z*u))
    #
    # So D_L/(c/H0) = (1+z) * z * integral_0^1 du / E(1/(1+z*u))
    # ≈ (1+z) * z * sum_i w_i / E(1/(1+z*t_i))
    # = (1+z) * z * sum_i w_i / E(a_i)
    #
    # OK so that matches the task formula! Great.
    #
    # So we need to find nodes t_i in [0,1] and weights w_i > 0, sum w_i = 1,
    # such that the quadrature sum_i w_i f(t_i) approximates integral_0^1 f(t) dt
    # for f(t) = 1/E(1/(1+z*t)) across various cosmological models and redshifts.
    #
    # The key insight from the knowledge: "Spectral initialization provides a globally
    # superior starting point in the weight/node space by ensuring the rule is accurate
    # for a broad class of smooth functions, preventing the local minimax optimizer
    # from getting stuck in local minima caused by poor initial weights/nodes. Measured:
    # 17.71 -> 76.66."
    #
    # So the strategy is:
    # 1. Start with a spectral (Gauss-Legendre) quadrature rule as initialization
    # 2. Use a local minimax optimizer to refine the nodes and weights
    # 3. Minimize the worst-case relative error across a set of test cosmologies
    
    # Step 1: Get Gauss-Legendre nodes and weights as initialization
    # For 8 nodes on [0,1]:
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    nodes = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0
    
    # Ensure nodes are strictly increasing
    nodes = np.sort(nodes)
    
    # Step 2: Define test cosmologies
    # We'll use a few standard CPL (Chester-Pearson-Linder) models
    # E(a) = (a^-3 + O_m*(1+q0 + s0*ln(a))^3 * a^-3)^0.5 ... 
    # Actually, for CPL: w = w0 + w1*(1-a), and E(a) = (a^-3 * (1 + Omega_m*a^-3 + ...))
    # Let me use the standard form.
    
    def make_E(Om, w0, w1):
        """Create E(a) function for CPL model."""
        def E(a):
            # For flat universe with CPL dark energy:
            # E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^(-3*(1+w0+w1)) * a^(-3*w1)
            # = Omega_m * a^-3 + (1-Omega_m) * exp(-3*w1*(ln(a))) * a^(-3*(1+w0))
            # Actually, the standard form is:
            # E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^(-3*(1+w0)) * exp(-3*w1*(1-a))
            # Wait, let me use the more standard form:
            # E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^(-3*(1+w0+w1)) * a^(-3*w1)
            # Hmm, let me just use: E(a)^2 = Omega_m/a^3 + (1-Omega_m)/a^(3*(1+w0+w1*(1-a)))
            # That's not quite right either. Let me use the integral form.
            #
            # For CPL: w(a) = w0 + w1*(1-a)
            # E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * a^(-3) * a^(-3*integral_1^a w(a')/a' da')
            # integral_1^a w(a')/a' da' = integral_1^a (w0 + w1*(1-a'))/a' da'
            # = w0*ln(a) + w1*integral_1^a (1/a' - 1) da'
            # = w0*ln(a) + w1*(ln(a) - (a-1))
            # = (w0+w1)*ln(a) - w1*(a-1)
            #
            # So E(a)^2 = Omega_m/a^3 + (1-Omega_m)/a^(3 + 3*(w0+w1)*ln(a) - 3*w1*(a-1))
            #
            # That's getting messy. Let me use a simpler parameterization.
            # I'll use: E(a)^2 = Omega_m/a^3 + (1-Omega_m)/a^(3*(1+w0)) * exp(-3*w1*(1-a))
            # This is a common approximation.
            
            a = np.asarray(a, dtype=float)
            a_safe = np.maximum(a, 1e-10)
            term_m = Om / a_safe**3
            term_de = (1 - Om) / a_safe**(3 * (1 + w0)) * np.exp(-3 * w1 * (1 - a_safe))
            return np.sqrt(term_m + term_de)
        return E
    
    # Define a set of test cosmologies for optimization
    # These should be representative but not the validation set
    test_cosmos = [
        (0.3, 0.0, 0.0),   # LCDM
        (0.3, -1.0, 0.0),  # Cosmological constant
        (0.3, -0.5, -0.5), # CPL with w0=-0.5, w1=-0.5
        (0.3, -1.5, 0.5),  # CPL with w0=-1.5, w1=0.5
        (0.25, -0.8, 0.3), # Another CPL
        (0.35, -1.2, 0.2), # Another CPL
    ]
    
    # Redshifts to test
    z_vals = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Step 3: Objective function - worst-case relative error
    def objective(params):
        """Compute worst-case relative error for given nodes and weights."""
        t = time.time()
        if t > deadline:
            return None  # Signal timeout
        
        # Unpack parameters: 8 nodes, 8 weights
        n = 8
        nodes_p = params[:n]
        weights_p = params[n:]
        
        # Enforce constraints: nodes in [0,1] strictly increasing, weights positive sum to 1
        # Soft constraints via penalty
        # Nodes: enforce 0 < t1 < t2 < ... < t8 < 1
        # Use tanh to map to (0,1) and sort
        nodes_raw = np.tanh(nodes_p)  # maps to (-1, 1)
        nodes_mapped = (nodes_raw + 1.0) / 2.0  # maps to (0, 1)
        nodes_mapped = np.sort(nodes_mapped)
        
        # Weights: use softmax to ensure positive and sum to 1
        weights_raw = weights_p
        weights_mapped = np.exp(weights_raw - np.max(weights_raw))
        weights_mapped = weights_mapped / np.sum(weights_mapped)
        
        # Compute worst-case relative error
        max_rel_err = 0.0
        for (Om, w0, w1) in test_cosmos:
            E = make_E(Om, w0, w1)
            for z in z_vals:
                # True integral using adaptive quadrature
                from scipy.integrate import quad
                def integrand(u):
                    a_val = 1.0 / (1.0 + z * u)
                    return 1.0 / E(a_val)
                
                true_integral, _ = quad(integrand, 0, 1, limit=200)
                true_value = (1 + z) * z * true_integral
                
                # Quadrature approximation
                a_i = 1.0 / (1.0 + z * nodes_mapped)
                E_vals = E(a_i)
                quad_value = (1 + z) * z * np.sum(weights_mapped / E_vals)
                
                if abs(true_value) > 1e-15:
                    rel_err = abs(quad_value - true_value) / abs(true_value)
                    max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err * 100.0  # Convert to percent
    
    # Step 4: Optimization using scipy
    from scipy.optimize import minimize
    
    # Initial parameters: inverse transform of initial nodes and weights
    # nodes_mapped = (tanh(nodes_p) + 1)/2, so nodes_p = arctanh(2*nodes_mapped - 1)
    nodes_init = nodes
    weights_init = weights
    
    # Inverse transform for nodes
    nodes_params_init = np.arctanh(np.clip(2 * nodes_init - 1, -0.999, 0.999))
    # Inverse transform for weights (log)
    weights_params_init = np.log(weights_init)
    
    params_init = np.concatenate([nodes_params_init, weights_params_init])
    
    # Run optimization
    best_params = params_init.copy()
    best_err = objective(best_params)
    
    if best_err is None:
        # Timeout, return initial
        return {"nodes": nodes_init, "weights": weights_init}
    
    # Try a few optimization methods
    methods_to_try = ['Nelder-Mead', 'Powell']
    
    for method in methods_to_try:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(
                objective,
                params_init,
                method=method,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun is not None and result.fun < best_err:
                best_err = result.fun
                best_params = result.x
        except Exception as e:
            pass
    
    # Extract final nodes and weights
    nodes_final_raw = np.tanh(best_params[:8])
    nodes_final = np.sort((nodes_final_raw + 1.0) / 2.0)
    
    weights_final_raw = best_params[8:]
    weights_final = np.exp(weights_final_raw - np.max(weights_final_raw))
    weights_final = weights_final / np.sum(weights_final)
    
    # Ensure nodes are strictly in (0,1) and strictly increasing
    nodes_final = np.clip(nodes_final, 1e-6, 1 - 1e-6)
    nodes_final = np.sort(nodes_final)
    # Ensure strict increase
    for i in range(1, len(nodes_final)):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-10
    
    # Ensure weights sum to 1 within 1e-12
    weights_final = weights_final / np.sum(weights_final)
    
    # Final check
    if time.time() > deadline:
        return {"nodes": nodes_init, "weights": weights_init}
    
    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
