"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule using basis-function projection.
    
    Strategy:
    1. Model the luminosity distance integrand shape using physical basis functions
       that capture the behavior of 1/E(a) in CPL dark energy models
    2. Fit a representative set of cosmologies to determine the effective integrand
    3. Derive optimal 8-node quadrature that exactly integrates the fitted basis
    4. Use Gauss-Legendre-like nodes as starting point, then optimize for the
       specific integrand family
    """
    n = 8
    deadline = time.time() + 85.0  # 85s deadline, 5s margin
    
    # The integrand in z-space is: f(z) = z * (1+z) / E(a) where a = 1/(1+z)
    # For CPL model: E(a) = a^(-3(1+w0+wa)) * (1+z)^(-3*wa)
    #           = a^(-3-3w0-3wa) * a^(-3wa) = a^(-3-3w0-6wa)
    # Wait, let me be more careful:
    # E(z) = (1+z)^(3(1+w0+wa)) * (1+z)^(-3*wa) = (1+z)^(3+3w0)
    # Actually for flat universe: H^2/H0^2 = Omega_m(1+z)^3 + Omega_de(1+z)^(3(1+w0+wa))
    # with Omega_m + Omega_de = 1
    # E(z) = sqrt(Omega_m(1+z)^3 + Omega_de(1+z)^(3(1+w0+wa)))
    
    # The integrand is z(1+z)/E(z)
    # For small z: E(z) ≈ 1 + (3/2)Omega_m*z + ...
    # For large z: matter dominates, E(z) ≈ sqrt(Omega_m)(1+z)^(3/2)
    # So integrand ≈ z(1+z)/(sqrt(Omega_m)(1+z)^(3/2)) = z/sqrt(Omega_m)/sqrt(1+z)
    
    # Key insight: the integrand has a specific shape that can be approximated
    # by a few basis functions. Let's use a physics-motivated basis.
    
    # Define a set of representative cosmologies for fitting
    # (Omega_m, w0, wa)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.5),    # CPL with positive wa
        (0.3, -1.0, -0.5),   # CPL with negative wa
        (0.2, -1.0, 0.0),    # Lower matter density
        (0.4, -1.0, 0.0),    # Higher matter density
        (0.3, -0.9, 0.0),    # Slightly different w0
        (0.3, -1.1, 0.0),    # Slightly different w0
        (0.3, -1.0, 1.0),    # Large positive wa
        (0.3, -1.0, -1.0),   # Large negative wa
        (0.3, -0.5, 0.0),    # Quintessence-like
        (0.3, -1.5, 0.0),    # Phantom-like
        (0.25, -1.0, 0.3),   # Mixed
        (0.35, -1.0, -0.3),  # Mixed
    ]
    
    def E_z(z, Omega_m, w0, wa):
        """H(z)/H0 for flat CPL model"""
        Omega_de = 1.0 - Omega_m
        z = np.asarray(z)
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_de * (1.0 + z)**(3.0*(1.0 + w0 + wa*z))
        return np.sqrt(term_m + term_de)
    
    def integrand(z, Omega_m, w0, wa):
        """The integrand z(1+z)/E(z)"""
        z = np.asarray(z)
        return z * (1.0 + z) / E_z(z, Omega_m, w0, wa)
    
    # For the quadrature, we need to integrate f(z) = z(1+z)/E(z) from 0 to z_max
    # But the task says nodes in [0,1], and the formula uses z*nodes[i]
    # So we're integrating from 0 to 1 in the variable t = z/z_max, or
    # the nodes are in [0,1] and represent z directly?
    
    # Re-reading: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # So nodes[i] are in [0,1] and z*nodes[i] is the actual redshift
    # This means we're integrating: integral_0^1 (1+z*t)*z * (1/E(1/(1+z*t))) * dt
    # = integral_0^1 z(1+z*t)/E(1/(1+z*t)) dt
    
    # Let u = z*t, then t = u/z, dt = du/z
    # integral_0^z z(1+u)/E(1/(1+u)) * du/z = integral_0^z (1+u)/E(1/(1+u)) du
    
    # So the quadrature approximates: integral_0^1 (1+u)/E(1/(1+u)) du * z
    # But wait, that's not quite right either. Let me re-read.
    
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # a_i = 1/(1+z*nodes[i])
    # E(a_i) = H(a_i)/H0
    
    # The exact formula is: dL = (c/H0) * (1+z) * integral_0^z du/E(u)
    # where E(u) = H(u)/H0
    
    # So (1+z)*z*sum_i weights[i]/E(a_i) should approximate (1+z)*integral_0^z du/E(u)
    
    # This means: z*sum_i weights[i]/E(a_i) ≈ integral_0^z du/E(u)
    # Or: sum_i weights[i]/E(1/(1+z*nodes[i])) ≈ (1/z)*integral_0^z du/E(u)
    
    # Let me define g(t) = 1/E(1/(1+z*t)) for t in [0,1]
    # Then we want: sum_i weights[i]*g(nodes[i]) ≈ (1/z)*integral_0^z du/E(u)
    # = (1/z)*integral_0^1 z*dt/E(1/(1+z*t)) = integral_0^1 dt/E(1/(1+z*t))
    
    # So the quadrature approximates: integral_0^1 1/E(1/(1+z*t)) dt
    # And the full dL is (1+z)*z times this integral
    
    # The integrand is: h(t) = 1/E(1/(1+z*t))
    # For different z and cosmologies, h(t) has different shapes
    
    # Key: we need a universal quadrature that works for all z and all cosmologies
    # The challenge is that h(t) depends on z
    
    # For small z: h(t) ≈ 1 (since 1/(1+z*t) ≈ 1 for small z)
    # For large z: h(t) varies significantly
    
    # Strategy: find nodes/weights that minimize worst-case error over a range
    # of z values and cosmologies
    
    # Let's use a different approach: optimize nodes and weights directly
    # using a composite objective over multiple test cases
    
    # Initial guess: Gauss-Legendre nodes on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes = (x_gl + 1.0) / 2.0  # map to [0,1]
    weights = w_gl / 2.0  # scale weights for [0,1]
    weights = weights / np.sum(weights)  # normalize
    
    # Define the objective: worst-case relative error over test cases
    def compute_error(nodes, weights, Omega_m, w0, wa, z_test):
        """Compute relative error for a given cosmology and redshift"""
        # Quadrature approximation
        a_vals = 1.0 / (1.0 + z_test * nodes)
        E_vals = E_z(z_test * nodes, Omega_m, w0, wa)  # E at redshift z*nodes
        quad_approx = np.sum(weights / E_vals)
        
        # Exact integral using adaptive quadrature
        from scipy.integrate import quad
        exact_integral, _ = quad(lambda t: 1.0 / E_z(z_test * t, Omega_m, w0, wa), 0, 1)
        
        if abs(exact_integral) < 1e-15:
            return 0.0
        return abs(quad_approx - exact_integral) / abs(exact_integral)
    
    # Test z values and cosmologies
    z_test_values = np.logspace(-2, 1, 15)  # z from 0.01 to 10
    test_cosmologies = cosmologies[:8]  # Use subset for speed
    
    def objective(params):
        """Objective function: worst-case error"""
        if time.time() > deadline:
            return np.inf
        
        # Unpack parameters: first n are nodes (logit-transformed), next n are weights (log-transformed)
        node_logits = params[:n]
        weight_logs = params[n:2*n]
        
        # Convert to actual nodes in (0,1) using sigmoid
        nodes = 1.0 / (1.0 + np.exp(-node_logits))
        
        # Sort nodes and corresponding weights
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weight_logs = weight_logs[sort_idx]
        
        # Check strict increase
        if np.any(np.diff(nodes) <= 1e-10):
            return np.inf
        
        # Convert to actual weights
        weights = np.exp(weight_logs)
        weights = weights / np.sum(weights)
        
        # Compute worst-case error
        max_err = 0.0
        for Omega_m, w0, wa in test_cosmologies:
            for z_test in z_test_values:
                err = compute_error(nodes, weights, Omega_m, w0, wa, z_test)
                if err > max_err:
                    max_err = err
                if time.time() > deadline:
                    return np.inf
        
        return max_err * 100.0  # in percent
    
    # Use Nelder-Mead optimization
    from scipy.optimize import minimize
    
    # Initial parameters
    init_nodes = (x_gl + 1.0) / 2.0
    init_weights = w_gl / 2.0
    init_weights = init_weights / np.sum(init_weights)
    
    # Convert to logit and log space
    init_node_logits = np.log(init_nodes / (1.0 - init_nodes))
    init_weight_logs = np.log(init_weights)
    
    init_params = np.concatenate([init_node_logits, init_weight_logs])
    
    best_params = init_params.copy()
    best_score = objective(init_params)
    
    # Run Nelder-Mead
    try:
        result = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Extract final nodes and weights
    node_logits = best_params[:n]
    weight_logs = best_params[n:2*n]
    
    nodes = 1.0 / (1.0 + np.exp(-node_logits))
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weight_logs = weight_logs[sort_idx]
    
    weights = np.exp(weight_logs)
    weights = weights / np.sum(weights)
    
    # Validate
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 1e-10)):
        # Fallback to Gauss-Legendre
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        weights = weights / np.sum(weights)
    
    if not np.all(weights > 0):
        # Fallback to midpoint
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
