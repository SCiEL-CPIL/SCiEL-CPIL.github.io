"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Structural change: Instead of non-convex global optimization of nodes and weights
    with fragile parameterizations, we:
    1. Start with Gauss-Legendre nodes (optimal for smooth functions).
    2. Use a convex reweighting approach: fix nodes, optimize weights via linear
       least-squares against a dense set of reference integrals.
    3. This is a well-posed, fast problem that avoids the instability of node
       optimization while still adapting weights to the cosmological integrand family.
    4. If the optimized weights are valid (positive, sum to 1), use them; otherwise
       fall back to Gauss-Legendre weights.
    
    The key insight: for a fixed set of nodes, the optimal weights for minimizing
    L2 error over a set of integrands are obtained by solving a linear system.
    This is much more stable than joint node-weight optimization.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    # 1. Gauss-Legendre nodes mapped to [0, 1]
    gl_nodes, gl_weights = roots_legendre(8)
    nodes = 0.5 * (gl_nodes + 1.0)
    
    # 2. Define representative CPL cosmologies
    Om_m = 0.3
    Om_de = 0.7
    
    # Dense set of (w0, wa) pairs and redshifts for reference
    w0_values = np.linspace(-1.2, 0.5, 8)
    wa_values = np.linspace(-0.8, 1.0, 7)
    z_values = np.array([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=5001):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                        4*np.sum(integrand[1:-1:2]) + 
                        2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    # 3. Pre-compute reference values and node integrand values
    # For each (w0, wa, z), compute:
    #   - reference integral value
    #   - 1/E(a_i) at the 8 fixed nodes
    
    ref_vals = []
    node_integrands = []  # shape: (n_cases, 8)
    
    for w0 in w0_values:
        for wa in wa_values:
            for z in z_values:
                if time.time() > deadline:
                    break
                try:
                    # Reference integral
                    rv = reference_integral(z, w0, wa)
                    if rv <= 1e-12:
                        continue
                    
                    # Node integrand values: 1/E(a_i) where a_i = 1/(1+z*nodes[i])
                    a_vals = 1.0 / (1.0 + z * nodes)
                    E_vals = E_of_a(a_vals, w0, wa)
                    node_vals = 1.0 / E_vals
                    
                    ref_vals.append(rv)
                    node_integrands.append(node_vals)
                except Exception:
                    continue
    
    if not ref_vals:
        # Fallback to Gauss-Legendre weights
        weights = 0.5 * gl_weights
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}
    
    ref_vals = np.array(ref_vals)
    node_integrands = np.array(node_integrands)  # shape: (n_cases, 8)
    
    # 4. Solve for weights: minimize sum_i (sum_j w_j * f_ij - ref_i)^2
    # This is a linear least-squares problem: A @ w = b
    # where A = node_integrands, b = ref_vals
    # The unconstrained solution minimizes L2 error.
    
    # Use np.linalg.lstsq for the unconstrained solution
    try:
        w_sol, residuals, rank, sv = np.linalg.lstsq(node_integrands, ref_vals, rcond=None)
    except Exception:
        w_sol = 0.5 * gl_weights
    
    # Check if the solution is valid (all positive)
    if np.all(w_sol > 1e-10):
        # Normalize to sum to 1
        weights = w_sol / np.sum(w_sol)
        
        # Verify the solution is good
        if time.time() < deadline - 5:
            # Compute the actual error to verify
            quad_vals = node_integrands @ weights
            rel_errors = np.abs(quad_vals - ref_vals) / ref_vals
            max_err = np.max(rel_errors)
            
            # If the error is reasonable, accept
            if max_err < 0.5:  # Less than 50% relative error
                pass  # Use these weights
            else:
                # Fall back to GL weights
                weights = 0.5 * gl_weights
    else:
        # The unconstrained solution has negative weights.
        # Try a constrained optimization: minimize L2 error subject to w > 0, sum(w) = 1
        # Use scipy.optimize.minimize with SLSQP
        
        def objective_constrained(w):
            if time.time() > deadline:
                return 1e10
            quad_vals = node_integrands @ w
            errors = quad_vals - ref_vals
            return np.sum(errors**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: w - 1e-10}
        ]
        
        w_init = 0.5 * gl_weights
        w_init = w_init / np.sum(w_init)
        
        try:
            result = minimize(objective_constrained, w_init, method='SLSQP',
                             constraints=constraints,
                             bounds=[(1e-10, None)]*8,
                             options={'maxiter': 100, 'ftol': 1e-12})
            
            if result.success and np.all(result.x > 1e-10):
                weights = result.x / np.sum(result.x)
            else:
                weights = 0.5 * gl_weights
        except Exception:
            weights = 0.5 * gl_weights
    
    # 5. Final sanity checks
    if not np.all(np.diff(nodes) > 0):
        nodes = 0.5 * (roots_legendre(8)[0] + 1.0)
        weights = 0.5 * roots_legendre(8)[1]
    
    if not np.all(weights > 0):
        weights = 0.5 * roots_legendre(8)[1]
    
    weights = weights / np.sum(weights)
    
    # Verify sum is within tolerance
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = 0.5 * roots_legendre(8)[1]
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
