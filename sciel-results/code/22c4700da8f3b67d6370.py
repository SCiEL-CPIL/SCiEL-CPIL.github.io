# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Discover an 8-node positive quadrature rule for dimensionless luminosity distance.
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), a_i=1/(1+z*nodes[i])
    """
    t0 = time.time()
    deadline = t0 + 80.0  # 80s safety margin within 90s limit

    # Representative cosmologies: (w0, wa) for CPL dark energy, flat universe
    # Omega_m = 0.3, Omega_DE = 0.7, Omega_k = 0
    omega_m = 0.3
    omega_de = 0.7

    # Range of (w0, wa) values to cover the validation space
    w0_vals = np.array([-0.5, -0.8, -1.0, -1.2, -1.5, -2.0, 0.0, 0.5, 1.0])
    wa_vals = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 2.0])

    # Redshift values to evaluate over (the integral is over z in [0,1] via nodes)
    # The nodes are in [0,1], so z goes from 0 to 1
    # We need to evaluate the integrand at the nodes for each cosmology

    # Gauss-Legendre nodes and weights as initial guess
    n = 8
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    # Transform from [-1,1] to [0,1]
    nodes_init = 0.5 * (x_gl + 1.0)
    weights_init = 0.5 * w_gl

    # Sort nodes to be strictly increasing
    sort_idx = np.argsort(nodes_init)
    nodes_init = nodes_init[sort_idx]
    weights_init = weights_init[sort_idx]

    # Ensure weights are positive and sum to 1
    weights_init = np.abs(weights_init)
    weights_init /= weights_init.sum()

    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_score = np.inf

    # Compute the reference integral for each cosmology using fine quadrature
    # and then compute the quadrature approximation to get the error

    def compute_errors(nodes, weights, w0, wa):
        """Compute relative error for a given cosmology."""
        # Nodes are in [0,1], representing z values
        z = nodes
        a = 1.0 / (1.0 + z)

        # E(a) = H(a)/H0
        # For CPL: w(a) = w0 + wa*(1-a), E^2 = Om*(1/a)^3 + Ol + Ok + Ode*(1/a)^3*(1+w0*wa*(1-a))^(-3*wa)
        # Simplified: E^2 = Om*(1+z)^3 + Ode*(1+z)^3*(1+w0*wa*z)^(3*wa)  (flat, Ok=0)
        # Actually: E^2 = Om*(1+z)^3 + Ode * (1+z)^3 * (1+w0*wa*z)^(-3*wa)
        # Wait, let me be more careful.
        # H^2/H0^2 = Om*(1+z)^3 + Ol + Ok + Ode*(1+z)^3*(1+w0*wa*z)^(3*wa)
        # For flat: Ok=0, Ol=Ode
        # E^2 = Om*(1+z)^3 + Ode*(1+z)^3*(1+w0*wa*z)^(3*wa)
        # = (1+z)^3 * [Om + Ode*(1+w0*wa*z)^(3*wa)]

        # The integrand for dL/(c/H0) = (1+z) * ∫_0^z dz'/E(z')
        # But the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # Wait, that's not quite right. Let me re-read.
        # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
        # Hmm, this seems like the quadrature is applied to the integral ∫_0^z dz'/E(z')
        # and the result is multiplied by (1+z)*z? That doesn't seem standard.
        # 
        # Actually, re-reading: the integral is over [0,1] in some variable.
        # Let me re-interpret: the quadrature approximates ∫_0^1 f(t) dt
        # where the full expression is (1+z)*z * [quadrature result]
        # 
        # Actually I think the formula means:
        # dL/(c/H0) = (1+z) * ∫_0^z dz'/E(z')
        # and the quadrature with nodes in [0,1] approximates ∫_0^1 g(t)dt
        # where g(t) = t/E(t) or something similar, scaled by z.
        #
        # Let me just implement what's stated: 
        # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        # where a_i = 1/(1+z*nodes[i])
        # So E(a_i) = E(1/(1+z*nodes[i]))

        # For a given target redshift z_target, the quadrature gives:
        # dL/(c/H0) = (1+z_target)*z_target*sum_i weights[i]/E(a_i)
        # where a_i = 1/(1+z_target*nodes[i])

        # But we're optimizing the nodes/weights, and the "error" is measured
        # against adaptive integration. The evaluator will test at various z.
        # 
        # For optimization, I'll evaluate at several target redshifts.

        # Actually, I think the nodes parameterize the integration variable,
        # and the formula is for a specific z. Let me think about this differently.
        
        # The integral to approximate is: I(z) = ∫_0^z dz'/E(z')
        # and dL/(c/H0) = (1+z)*I(z)
        
        # The quadrature rule with nodes t_i in [0,1] approximates:
        # I(z) ≈ z * sum_i w_i / E(z*t_i)
        # because we substitute u = z*t, so du = z*dt, and ∫_0^z dz'/E(z') = z*∫_0^1 dt/E(z*t)
        
        # So dL/(c/H0) = (1+z) * z * sum_i w_i / E(z*t_i)
        # where a_i = 1/(1+z*t_i), so E(a_i) = E(1/(1+z*t_i))
        
        # This matches the formula! Good.
        
        # Now for optimization, I need to compare this quadrature against
        # the true integral I(z) = ∫_0^z dz'/E(z')
        
        # Let me pick several target redshifts to evaluate
        z_targets = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0])
        
        errors = []
        for zt in z_targets:
            # True integral using fine quadrature
            z_fine = np.linspace(0, zt, 10000)
            a_fine = 1.0 / (1.0 + z_fine)
            # E^2 = Om*(1+z)^3 + Ode*(1+z)^3*(1+w0*wa*z)^(3*wa)
            # Handle the case where (1+w0*wa*z) might be negative
            base = 1.0 + w0 * wa * z_fine
            # If base is negative and exponent is non-integer, we get NaN
            # For physical models, base should stay positive
            with np.errstate(divide='ignore', invalid='ignore'):
                if wa == 0:
                    de_factor = np.ones_like(z_fine)
                else:
                    de_factor = np.power(np.maximum(base, 1e-30), 3.0 * wa)
                E2_fine = omega_m * (1.0 + z_fine)**3 + omega_de * (1.0 + z_fine)**3 * de_factor
                E_fine = np.sqrt(np.maximum(E2_fine, 1e-30))
                integrand_fine = 1.0 / E_fine
            
            # Integrate using trapezoidal rule
            true_I = np.trapz(integrand_fine, z_fine)
            
            # Quadrature approximation
            a_quad = 1.0 / (1.0 + zt * nodes)
            z_quad = zt * nodes
            base_quad = 1.0 + w0 * wa * z_quad
            with np.errstate(divide='ignore', invalid='ignore'):
                if wa == 0:
                    de_factor_quad = np.ones_like(z_quad)
                else:
                    de_factor_quad = np.power(np.maximum(base_quad, 1e-30), 3.0 * wa)
                E2_quad = omega_m * (1.0 + z_quad)**3 + omega_de * (1.0 + z_quad)**3 * de_factor_quad
                E_quad = np.sqrt(np.maximum(E2_quad, 1e-30))
                integrand_quad = 1.0 / E_quad
            
            quad_I = zt * np.sum(weights * integrand_quad)
            
            # Relative error
            if abs(true_I) > 1e-15:
                rel_err = abs(quad_I - true_I) / abs(true_I)
            else:
                rel_err = 0.0
            errors.append(rel_err)
        
        return errors

    def smooth_max(errors, beta=50.0):
        """Smooth maximum using log-sum-exp."""
        errors = np.array(errors)
        max_e = np.max(errors)
        if max_e < 1e-15:
            return 0.0
        scaled = beta * errors
        # Use log-sum-exp for numerical stability
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        return lse / beta

    # Optimization using Nelder-Mead or similar
    # Parameters: 8 nodes + 8 weights = 16 parameters
    # Constraints: nodes strictly increasing in [0,1], weights positive, sum=1
    
    # Use reparameterization:
    # nodes: use cumulative sum of positive values, normalized to [0,1]
    # weights: use softmax or positive normalization
    
    def params_to_nodes_weights(params):
        """Convert unconstrained parameters to valid nodes and weights."""
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: cumulative sum of softplus, normalized
        node_shifts = np.exp(node_params)  # positive
        nodes = np.cumsum(node_shifts)
        nodes = nodes / nodes[-1]  # normalize to [0,1]
        # Ensure strictly increasing by adding small epsilon
        # The cumsum of positive values is strictly increasing, so this is fine
        # But we need nodes in [0,1] and strictly increasing
        # The first node should be > 0, last < 1
        # Let's add a small offset
        eps = 1e-10
        nodes = np.maximum(nodes, eps)
        nodes = np.minimum(nodes, 1.0 - eps)
        # Ensure strictly increasing
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + eps
        
        # Weights: softmax
        w = np.exp(weight_params - np.max(weight_params))
        weights = w / w.sum()
        
        return nodes, weights

    def objective(params):
        """Objective function to minimize."""
        nodes, weights = params_to_nodes_weights(params)
        
        # Check if still within deadline
        if time.time() > deadline:
            return 1e10
        
        # Compute errors across all cosmologies
        all_errors = []
        for w0 in w0_vals:
            for wa in wa_vals:
                errs = compute_errors(nodes, weights, w0, wa)
                all_errors.extend(errs)
        
        all_errors = np.array(all_errors)
        return smooth_max(all_errors, beta=50.0)

    # Initial parameters
    nodes_init, weights_init = params_to_nodes_weights(np.zeros(16))
    
    # Actually, let me start from Gauss-Legendre
    # Invert the reparameterization approximately
    # For nodes: node_shifts = diff(nodes), first element = nodes[0]
    # For weights: weight_params = log(weights)
    
    node_shifts = np.diff(np.concatenate([[0], nodes_init]))
    node_params = np.log(np.maximum(node_shifts, 1e-10))
    weight_params = np.log(np.maximum(weights_init, 1e-10))
    init_params = np.concatenate([node_params, weight_params])

    # Use scipy.optimize if available, else simple gradient descent
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False

    if use_scipy:
        # Try Nelder-Mead first (no gradient needed)
        result = minimize(
            objective, init_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        if time.time() < deadline:
            # Refine with L-BFGS-B if possible
            best_params = result.x
            best_obj = result.fun
            
            # Try a few more restarts with different initializations
            np.random.seed(42)
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                
                # Perturb the best solution
                perturbed = best_params + np.random.randn(16) * 0.1
                result2 = minimize(
                    objective, perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if result2.fun < best_obj:
                    best_obj = result2.fun
                    best_params = result2.x

            final_params = best_params
        else:
            final_params = result.x
    else:
        # Simple coordinate descent
        final_params = init_params.copy()
        best_obj = objective(final_params)
        
        step_size = 0.1
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(16):
                for direction in [-1, 1]:
                    trial = final_params.copy()
                    trial[i] += direction * step_size
                    trial_obj = objective(trial)
                    
                    if trial_obj < best_obj:
                        best_obj = trial_obj
                        final_params = trial
                        improved = True
                        break
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-6:
                    break

    # Extract final nodes and weights
    final_nodes, final_weights = params_to_nodes_weights(final_params)
    
    # Ensure constraints are met
    # Nodes strictly increasing in [0,1]
    final_nodes = np.sort(final_nodes)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-10
    final_nodes = np.clip(final_nodes, 1e-10, 1.0 - 1e-10)
    
    # Weights positive and sum to 1
    final_weights = np.abs(final_weights)
    final_weights = final_weights / final_weights.sum()
    
    # Verify sum is within 1e-12
    wsum = final_weights.sum()
    final_weights = final_weights / wsum
    
    return {"nodes": final_nodes, "weights": final_weights}
# EVOLVE-BLOCK-END
