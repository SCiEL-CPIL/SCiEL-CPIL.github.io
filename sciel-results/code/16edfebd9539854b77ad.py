"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    The rule minimizes worst-case relative error across a set of representative
    CPL (Chevallier-Polarski-Linder) cosmologies. We use a smooth surrogate objective
    (sum of squared log-errors) optimized with L-BFGS-B, then refine with Powell.
    
    The integrand is f(a) = 1/E(a) where a = 1/(1+z*node), and E(a) = H(a)/H0.
    The full distance is dL/(c/H0) = (1+z)*z * integral_0^z dz'/E(a(z'))
    
    But in the quadrature form given: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i])
    
    So the quadrature approximates: integral_0^1 du / E(1/(1+zu))  (with u as dummy variable)
    Actually, let me re-read: the formula is (1+z)*z*sum_i w_i/E(a_i) with a_i = 1/(1+z*nodes[i])
    
    This means we're approximating the integral ∫_0^z dz'/E(1/(1+z'))
    By substituting u = z'/z, we get z * ∫_0^1 du / E(1/(1+zu))
    And the quadrature is sum_i w_i / E(1/(1+z*nodes[i]))
    So we're approximating ∫_0^1 du / E(1/(1+zu)) with nodes in [0,1].
    
    The nodes must be strictly increasing in [0,1], weights positive, sum to 1.
    """
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Representative CPL cosmologies: (w0, wa) pairs
    # E(a) = a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(3*wa/(1+w0+wa))
    # For flat universe with matter and dark energy:
    # E(a) = sqrt(Om_m/a^3 + Om_de * a^(-3(1+w0+wa)) * (1+wa*(1-a))^(3*wa/(1+w0+wa)))
    # With Om_m = 0.3, Om_de = 0.7 for most models
    
    # Let's define a set of representative (w0, wa) values
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
    wa_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    cosmologies = []
    for w0 in w0_values:
        for wa in wa_values:
            cosmologies.append((w0, wa))
    
    Om_m = 0.3
    Om_de = 0.7
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        # Matter term
        matter = Om_m / a**3
        # Dark energy term
        # rho_de(a) = rho_de(1) * a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(3*wa/(1+w0+wa))
        # E^2 = Om_m/a^3 + Om_de * a^(-3(1+w0+wa)) * (1+wa*(1-a))^(3*wa/(1+w0+wa))
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def reference_integral(z, w0, wa, n_ref=10000):
        """High-accuracy reference integral using many points."""
        # Integral from 0 to z of dz'/E(1/(1+z'))
        # Use substitution u = z'/z, integral = z * ∫_0^1 du/E(1/(1+zu))
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        # Simpson's rule for high accuracy
        if n_ref % 2 == 0:
            n_ref = n_ref - 1
            u = np.linspace(0, 1, n_ref)
            a_vals = 1.0 / (1.0 + z * u)
            E_vals = E_of_a(a_vals, w0, wa)
            integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 4*np.sum(integrand[1:-1:2]) + 2*np.sum(integrand[2:-1:2]))
        return z * integral
    
    def quadrature_error(nodes, weights, z, w0, wa):
        """Compute relative error of quadrature vs reference for given cosmology and z."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        quad_val = np.sum(weights / E_vals)
        ref_val = reference_integral(z, w0, wa)
        # The full distance is (1+z)*z*quad_val vs (1+z)*z*ref_val
        # Relative error is |quad_val - ref_val| / ref_val
        return np.abs(quad_val - ref_val) / ref_val
    
    def objective(params):
        """Objective: sum of squared log-errors across test points."""
        if time.time() > deadline:
            return 1e10
        
        # Unpack nodes and weights
        nodes_raw = params[:8]
        weights_raw = params[8:16]
        
        # Enforce strictly increasing nodes in (0, 1) using sigmoid transformation
        # Use a parameterization that guarantees ordering
        # Let nodes be cumulative sums of positive values
        # Or use: node_i = sigmoid(x_i) with x_i strictly increasing
        # Simpler: use softplus to ensure positivity of differences
        # d_i = softplus(y_i) > 0, node_0 = softplus(y_0)/S, etc.
        
        # Alternative: parameterize nodes as cumulative fractions
        # Let t_i = exp(params[i]) for i=0..7, then normalize
        # nodes[i] = cumsum(t)[i] / cumsum(t)[-1]
        # This ensures nodes are in (0,1) and strictly increasing
        
        t = np.exp(nodes_raw)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        # Ensure nodes are strictly in (0,1)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Weights: use softmax to ensure positive and sum to 1
        w_exp = np.exp(weights_raw - np.max(weights_raw))
        weights = w_exp / np.sum(w_exp)
        
        # Compute objective: sum of squared relative errors across test z and cosmologies
        total_error = 0.0
        z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for w0, wa in cosmologies:
            for z in z_test:
                err = quadrature_error(nodes, weights, z, w0, wa)
                total_error += np.log1p(err * 1000)**2  # log-scaled error
        
        return total_error
    
    # Initial guess: Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = roots_legendre(8)
    # Map from [-1,1] to [0,1]
    gl_nodes_01 = 0.5 * (gl_nodes + 1)
    
    # Invert the parameterization
    # nodes = cumsum(exp(nodes_raw)) / cumsum(exp(nodes_raw))[-1]
    # So we need to find nodes_raw such that this holds
    # This is tricky; let's just start with a reasonable initial guess
    # For the t parameterization: t_i such that cumsum(t)/cumsum(t)[-1] = gl_nodes_01
    # t_0 = gl_nodes_01[0]
    # t_i = gl_nodes_01[i] - gl_nodes_01[i-1] for i > 0
    t_init = np.zeros(8)
    t_init[0] = gl_nodes_01[0]
    for i in range(1, 8):
        t_init[i] = gl_nodes_01[i] - gl_nodes_01[i-1]
    nodes_raw_init = np.log(t_init + 1e-10)
    
    # Weights: softmax inverse
    weights_raw_init = np.log(gl_weights + 1e-10) - np.log(gl_weights + 1e-10).max()
    
    params_init = np.concatenate([nodes_raw_init, weights_raw_init])
    
    # Optimize with L-BFGS-B
    try:
        result = minimize(objective, params_init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 5000})
        if time.time() > deadline:
            raise TimeoutError
    except Exception:
        result = None
    
    # Refine with Powell if time allows
    if result is not None and time.time() < deadline - 5:
        try:
            result2 = minimize(objective, result.x, method='Powell',
                              options={'maxiter': 100, 'maxfev': 5000})
            if result2.fun < result.fun:
                result = result2
        except Exception:
            pass
    
    if result is None:
        # Fallback to Gauss-Legendre
        nodes = gl_nodes_01
        weights = gl_weights
    else:
        params_opt = result.x
        t = np.exp(params_opt[:8])
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        w_exp = np.exp(params_opt[8:16] - np.max(params_opt[8:16]))
        weights = w_exp / np.sum(w_exp)
    
    # Ensure weights sum to 1 within tolerance
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing
    if not np.all(np.diff(nodes) > 0):
        nodes = gl_nodes_01
        weights = gl_weights
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
