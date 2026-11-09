"""Rewritten program: adaptive minimax quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless
    luminosity-distance integrand.

    The integrand for a cosmology is f(t) = (1 + z*t) / E(a(t)) where
    t = z_node (the quadrature abscissa in z-space mapped to [0,1]),
    a(t) = 1/(1+t), and E(a) = H(a)/H0.

    The luminosity distance is dL/(c/H0) = (1+z) * z * integral_0^1 f(t) dt.

    Since (1+z)*z is a prefactor independent of the quadrature, we need
    to approximate integral_0^1 f(t) dt well for all cosmologies.

    Strategy: The integrand f(t) = (1+t)/E(1/(1+t)) varies in shape depending
    on the dark energy model. For a flat universe with CPL dark energy:
    E(z) = (1+z)^3 * [omega_m + omega_lam*(1+z)^(-3*(1+w0+wa)) - omega_lam*wa*(1+z)^(-3*(w0+wa))*ln(1+z)]
    (for wa != 0), or simpler forms for wa=0.

    We design the quadrature to minimize the worst-case relative error over
    a family of representative integrands. We use a minimax optimization
    over the nodes and weights, with the constraint that weights > 0 and
    sum to 1.

    We parameterize nodes as strictly increasing values in (0,1) and weights
    as positive values summing to 1 (using a softplus/log-exp reparameterization
    for optimization stability).

    We test against a set of representative cosmologies covering a range of
    omega_m, omega_lam, w0, wa parameters, and minimize the maximum relative
    error across these models and a range of redshifts.
    """

    # Representative set of cosmological parameters for design
    # (omega_m, w0, wa) - spatially flat, so omega_lam = 1 - omega_m
    # We include a range of models to be robust
    design_models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.1, 0.0),
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.30, -0.8, 0.5),
        (0.30, -1.2, -0.3),
        (0.20, -1.0, 0.3),
        (0.40, -1.0, -0.3),
    ]

    # Redshifts at which to evaluate (the quadrature is applied per-distance)
    # The integral is over t in [0,1] representing the scaled variable,
    # but the actual z can vary. The key is that the shape of f(t)
    # depends on z and the cosmology. For a given z, the integrand is:
    # f(t) = (1 + z*t) / E(z*t)  where a = 1/(1+z*t)
    #
    # Wait, let me re-read the task. The formula is:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    #
    # So the integral being approximated is:
    # integral_0^1 [1/E(1/(1+z*u))] du
    # and the quadrature approximates this with sum_i weights[i] / E(1/(1+z*nodes[i]))
    #
    # The (1+z)*z is a prefactor. So we need to approximate:
    # I(z) = integral_0^1 1/E(1/(1+z*u)) du
    #
    # The integrand g_z(u) = 1/E(1/(1+z*u)) where u in [0,1]
    #
    # For different z and cosmologies, g_z has different shapes.
    # We want a single quadrature rule (nodes, weights) that works well
    # for all z > 0 and all cosmologies.

    def E_of_z(z, omega_m, w0, wa):
        """Compute E(z) = H(z)/H0 for flat CPL model."""
        if abs(wa) < 1e-10:
            # E(z) = (1+z)^{3/2} * sqrt(omega_m*(1+z)^3 + omega_lam)
            omega_lam = 1.0 - omega_m
            return (1.0 + z)**1.5 * np.sqrt(omega_m*(1.0 + z)**3 + omega_lam)
        else:
            omega_lam = 1.0 - omega_m
            z1 = 1.0 + z
            term1 = omega_m * z1**3
            term2 = omega_lam * z1**(-3*(1+w0+wa))
            term3 = -omega_lam * wa * z1**(-3*(w0+wa)) * np.log(z1)
            return z1**1.5 * np.sqrt(term1 + term2 + term3)

    def integrand(u, z, omega_m, w0, wa):
        """g_z(u) = 1/E(z*u) for u in [0,1]"""
        zz = z * u
        # Avoid E(0) issues; E(0) = 1 always
        if zz < 1e-15:
            return 1.0
        e_val = E_of_z(zz, omega_m, w0, wa)
        if e_val < 1e-30:
            return 1e30
        return 1.0 / e_val

    def high_accuracy_integral(z, omega_m, w0, wa, n_quad=10000):
        """Compute integral_0^1 1/E(z*u) du using fine trapezoidal rule."""
        u = np.linspace(0.0, 1.0, n_quad)
        vals = np.array([integrand(ui, z, omega_m, w0, wa) for ui in u])
        return np.trapz(vals, u)

    # Precompute high-accuracy integrals for design set
    z_evals = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

    # We'll define the objective function
    n_nodes = 8

    def nodes_from_params(params):
        """Convert optimization params to strictly increasing nodes in (0,1)."""
        # Use a parameterization that ensures strict increase and bounds in (0,1)
        # params: 7 values for differences, 1 for offset
        # Let nodes[0] = sigmoid(params[0]) * 0.9 + 0.05
        # nodes[i+1] = nodes[i] + 0.05 + 0.9*sigmoid(params[i])
        # This ensures nodes[0] > 0.05 and nodes[i+1] - nodes[i] > 0.05
        # Max node = 0.05 + 7*0.95 = 0.05 + 6.65 = 6.7 which is too large
        # Let's use a different approach: use the fact that we need 8 nodes in (0,1)
        # Use logit parameterization: nodes[i] = sigmoid(params[i])
        # But we need them sorted. So use a different approach.
        # Let's use: nodes[i] = (i+1 + params[i]) / (n_nodes+1) but that's not flexible enough.
        
        # Better: use the difference parameterization with scaling
        # d[0] = nodes[0], d[i] = nodes[i] - nodes[i-1] for i=1..7
        # We need sum(d) = nodes[7] < 1, and each d[i] > 0
        # Let d[i] = exp(params[i]) * scale, where scale is chosen so sum = 0.9
        # Then nodes[0] = d[0], nodes[1] = d[0]+d[1], etc.
        # This ensures all nodes in (0, 0.9) which is subset of (0,1)
        
        d = np.exp(params[:n_nodes])
        d = d / d.sum() * 0.9  # sum to 0.9, leaving margin
        nodes = np.cumsum(d)
        # nodes is now in (0, 0.9), strictly increasing
        return nodes

    def weights_from_params(params):
        """Convert optimization params to positive weights summing to 1."""
        w_raw = np.exp(params[n_nodes:])
        return w_raw / w_raw.sum()

    def objective(params):
        nodes = nodes_from_params(params)
        weights = weights_from_params(params)
        
        max_rel_err = 0.0
        for (om, w0, wa) in design_models:
            for z in z_evals:
                # Compute quadrature estimate
                quad_est = 0.0
                for i in range(n_nodes):
                    quad_est += weights[i] * integrand(nodes[i], z, om, w0, wa)
                
                # Compute reference
                ref = high_accuracy_integral(z, om, w0, wa, n_quad=5000)
                
                if ref < 1e-30:
                    rel_err = 0.0
                else:
                    rel_err = abs(quad_est - ref) / ref
                
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
        
        return max_rel_err * 100.0  # in percent

    # Initial guess: midpoint rule
    init_nodes = (np.arange(n_nodes) + 0.5) / n_nodes
    init_weights = np.full(n_nodes, 1.0 / n_nodes)
    
    # Invert parameterization for initial guess
    # nodes = cumsum(d), d = d_raw / sum(d_raw) * 0.9
    # So d_i = nodes[i] - nodes[i-1] (with nodes[-1]=0)
    d_init = np.diff(np.concatenate(([0.0], init_nodes)))
    # d_init sums to init_nodes[-1] = 0.9375
    # We want d = exp(params)/sum(exp(params))*0.9
    # So exp(params) proportional to d_init
    params_init = np.log(d_init)
    params_init = np.concatenate([params_init, np.log(init_weights)])
    
    # Optimize
    result = minimize(objective, params_init, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    final_params = result.x
    nodes = nodes_from_params(final_params)
    weights = weights_from_params(final_params)
    
    # Ensure nodes are strictly increasing and in (0,1)
    # They should be by construction, but let's verify
    if not np.all(np.diff(nodes) > 0):
        # Fallback to Gauss-Legendre
        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        # Clip and adjust
        nodes = np.clip(nodes, 0.01, 0.99)
        # Re-sort and fix
        nodes, idx = np.sort(nodes), None
        if len(np.unique(nodes)) < n_nodes:
            nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
            nodes = (nodes + 1) / 2
            weights = weights / 2
            return {"nodes": nodes, "weights": weights}
    
    if np.any(weights <= 0):
        weights, nodes = np.polynomial.legendre.leggauss(n_nodes)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
