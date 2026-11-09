"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    The integrand is f(z) = z * E(a)^(-1) where a = 1/(1+z) and E(a) = H(a)/H0.
    The luminosity distance is dL/(c/H0) = (1+z) * integral_0^z f(z') dz'.
    
    We seek nodes t_i in [0,1] and weights w_i > 0, sum w_i = 1, such that
    sum_i w_i * f(z_i) approximates integral_0^z f(z') dz' for all z.
    
    We use Gauss-Legendre quadrature on [0,1] as the starting point, then
    optimize using a set of cosmological models to minimize worst-case error.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Define cosmological models to train on
    # E(a) = H(a)/H0 for different dark energy models
    def E_cdm(a, w0=-1.0, wa=0.0):
        """CPL parameterization: E^2 = (1-a)^3 * (1 + w0*(1-a) + wa*(1-a)^2/2) + (1-a)^3 * ... 
        Actually for flat universe: E^2 = (1-a)^3 + (1-a)^3 * exp(3*(1+w0)*(1-a) - 1.5*wa*(1-a)^2)
        More standard: E^2(a) = (1-a)^3 + (1-a)^3 * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        Let me use: E^2(a) = (1-a)^3 * (1 + (1-a)^(-3*(1+w0)) * exp(-3*wa*(1-a)))
        Actually the standard CPL is:
        w(a) = w0 + wa*(1-a)
        E^2(a) = (1-a)^3 + (1-a)^3 * exp(3*(1+w0)*(1-a) - 1.5*wa*(1-a)^2)
        Wait, let me be more careful. For a flat universe with matter and dark energy:
        E^2(a) = (1-a)^3 + (1-a)^3 * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        This is for w(a) = w0 + wa*(1-a).
        """
        if a <= 0:
            return 0.0
        w = w0 + wa * (1.0 - a)
        # Integral of 3(1+w(a'))/a' da' from a to 1
        # = 3*(1+w0)*ln(a) - 1.5*wa*(1-a)^2... let me just use the standard form
        # E^2(a) = (1-a)^3 + (1-a)^3 * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        a_safe = max(a, 1e-10)
        de_term = a_safe ** (-3.0 * (1.0 + w0)) * np.exp(-3.0 * wa * (1.0 - a_safe))
        return np.sqrt((1.0 - a_safe) ** 3 + (1.0 - a_safe) ** 3 * de_term)
    
    # Actually, let me use a cleaner formulation
    def E_model(a, w0, wa):
        """E(a) for CPL dark energy in flat universe"""
        a = np.maximum(a, 1e-15)
        # Matter term: (1-a)^3
        # Dark energy term: (1-a)^3 * exp(3*(1+w0)*(1-a) - 1.5*wa*(1-a)^2)
        # Wait, this isn't quite right either. Let me use:
        # E^2(a) = (1-a)^3 + (1-a)^3 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
        # This corresponds to w(a) = w0 + wa*(1-a)
        matter = (1.0 - a) ** 3
        de = (1.0 - a) ** 3 * np.exp(3.0 * (1.0 + w0) * (1.0 - a) - 1.5 * wa * (1.0 - a) ** 2)
        return np.sqrt(matter + de)
    
    # Actually the correct CPL form is:
    # E^2(a) = (1-a)^3 + (1-a)^3 * exp(3*(1+w0)*(1-a) - 3/2*wa*(1-a)^2)
    # No wait. Let me just use a simple parametric form.
    
    def E_cpl(a, w0, wa):
        """E(a)/H0 for flat universe with CPL dark energy"""
        a = np.asarray(a, dtype=float)
        a_safe = np.maximum(a, 1e-15)
        one_minus_a = 1.0 - a_safe
        # Standard CPL: w(a) = w0 + wa*(1-a)
        # E^2(a) = (1-a)^3 + (1-a)^3 * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        # Let me verify: for w0=-1, wa=0, this gives E^2 = (1-a)^3 + (1-a)^3 = 2(1-a)^3? No.
        # For cosmological constant w=-1: E^2 = (1-a)^3 + (1-a)^3 * a^0 * exp(0) = 2(1-a)^3? 
        # That's wrong. The dark energy density is constant, so rho_DE/rho_crit = (1-a)^3 * a^(-3(1+w))
        # For w=-1: rho_DE/rho_crit = (1-a)^3 * a^0 = (1-a)^3? No, that's not right either.
        # 
        # Let me think again. Omega_m = 0.3, Omega_DE = 0.7.
        # E^2(a) = Omega_m * a^(-3) + Omega_DE * a^(-3(1+w_eff))
        # For CPL: a^(-3(1+w_eff)) where w_eff is the effective equation of state.
        # Actually: rho_DE(a) = rho_DE(0) * a^(-3(1+w0)) * exp(-3*wa*(1-a))
        # So E^2(a) = Omega_m * a^(-3) + Omega_DE * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        # = (1-a)^3 * [Omega_m + Omega_DE * a^(-3*(1+w0)) * exp(-3*wa*(1-a))]
        # Wait no. E^2(a) = H^2(a)/H0^2 = Omega_m * a^(-3) + Omega_DE * (rho_DE(a)/rho_DE(0))
        # = Omega_m * a^(-3) + Omega_DE * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
        # 
        # Let's use Omega_m = 0.3, Omega_DE = 0.7
        omega_m = 0.3
        omega_de = 0.7
        term_m = omega_m * a_safe ** (-3.0)
        term_de = omega_de * a_safe ** (-3.0 * (1.0 + w0)) * np.exp(-3.0 * wa * (1.0 - a_safe))
        return np.sqrt(term_m + term_de)
    
    # Training cosmologies
    models = [
        (-1.0, 0.0),   # Lambda CDM
        (-1.2, 0.5),   # Phantom with evolution
        (-0.9, -0.3),  # Quintessence with evolution
        (-1.0, 0.8),   # Lambda with strong evolution
        (-0.8, -0.5),  # Quintessence
        (-1.1, 0.3),   # Mild phantom
        (-1.0, -0.6),  # Lambda with negative evolution
        (-0.95, 0.4),  # Near lambda
    ]
    
    # Integration points for reference (adaptive quadrature)
    z_eval = np.linspace(0.01, 3.0, 50)
    
    def compute_error(nodes, weights, w0, wa):
        """Compute worst-case relative error for a given quadrature rule and model"""
        worst_err = 0.0
        for z in z_eval:
            # Reference: adaptive integration
            a_vals = 1.0 / (1.0 + z_eval)
            # For the integral from 0 to z, we need to integrate f(z') from 0 to z
            # f(z') = z' / E(a') where a' = 1/(1+z')
            # Let's use scipy.integrate.quad for reference
            
            from scipy.integrate import quad
            
            def integrand(zp):
                ap = 1.0 / (1.0 + zp)
                Ep = E_cpl(ap, w0, wa)
                return zp / Ep
            
            ref_val, _ = quad(integrand, 0.0, z, limit=100)
            ref_dL = (1.0 + z) * ref_val
            
            # Quadrature approximation
            a_nodes = 1.0 / (1.0 + z * nodes)
            E_nodes = E_cpl(a_nodes, w0, wa)
            quad_val = np.sum(weights * z * nodes / E_nodes)
            quad_dL = (1.0 + z) * quad_val
            
            if abs(ref_dL) > 1e-10:
                rel_err = abs(quad_dL - ref_dL) / abs(ref_dL)
                worst_err = max(worst_err, rel_err)
        
        return worst_err
    
    # Initial guess: Gauss-Legendre on [0,1]
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    nodes0 = (x_gl + 1.0) / 2.0
    weights0 = w_gl / 2.0
    
    # Optimize using differential evolution
    def objective(params):
        """Objective function for optimization"""
        if time.time() > deadline:
            return 1e10
        
        nodes = params[:8]
        weights = params[8:16]
        
        # Check constraints
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10
        
        # Compute worst-case error across models
        worst_err = 0.0
        for w0, wa in models:
            err = compute_error(nodes, weights, w0, wa)
            worst_err = max(worst_err, err)
        
        return worst_err
    
    # Bounds for differential evolution
    # Nodes must be in (0,1) and strictly increasing
    # We'll parameterize as: node_i = cumulative sum of positive spacings
    # Let s_i > 0, sum s_i = 1, node_i = s_1 + ... + s_i
    # This automatically ensures nodes in (0,1) and strictly increasing
    
    def params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights"""
        # First 8 params: log-spacings for nodes
        log_spacings = params[:8]
        spacings = np.exp(log_spacings)
        spacings = spacings / np.sum(spacings)
        nodes = np.cumsum(spacings)
        
        # Next 8 params: log-weights
        log_weights = params[8:16]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        return nodes, weights
    
    def objective_constrained(params):
        """Constrained objective using reparameterization"""
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = params_to_nodes_weights(params)
        
        worst_err = 0.0
        for w0, wa in models:
            err = compute_error(nodes, weights, w0, wa)
            worst_err = max(worst_err, err)
        
        return worst_err
    
    # Bounds for the log-parameterization
    # log_spacings: range [-10, 10] (very wide)
    # log_weights: range [-10, 10]
    bounds = [(-5.0, 5.0)] * 16
    
    # Run differential evolution
    result = differential_evolution(
        objective_constrained,
        bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        polish=False,
        init='sobol',
        workers=1
    )
    
    if time.time() > deadline - 5:
        # Use the initial Gauss-Legendre rule
        return {"nodes": nodes0, "weights": weights0}
    
    best_params = result.x
    nodes_opt, weights_opt = params_to_nodes_weights(best_params)
    
    # Verify constraints
    if not (np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and 
            np.all(np.diff(nodes_opt) > 0) and np.all(weights_opt > 0) and
            abs(np.sum(weights_opt) - 1.0) < 1e-10):
        return {"nodes": nodes0, "weights": weights0}
    
    # Compare with Gauss-Legendre
    err_opt = result.fun
    err_gl = objective_constrained(np.log(np.array([
        nodes0[0], nodes0[1]-nodes0[0], nodes0[2]-nodes0[1], nodes0[3]-nodes0[2],
        nodes0[4]-nodes0[3], nodes0[5]-nodes0[4], nodes0[6]-nodes0[5], nodes0[7]-nodes0[6],
        weights0[0], weights0[1], weights0[2], weights0[3],
        weights0[4], weights0[5], weights0[6], weights0[7]
    ])))
    
    if err_opt < err_gl:
        return {"nodes": nodes_opt, "weights": weights_opt}
    else:
        return {"nodes": nodes0, "weights": weights0}

# EVOLVE-BLOCK-END
