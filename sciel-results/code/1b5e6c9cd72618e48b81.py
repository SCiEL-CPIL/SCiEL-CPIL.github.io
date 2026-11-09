"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    Uses a rational-function (Padé-like) insight: the integrand 1/E(a) for CPL models
    has structure that can be approximated by a rational function. We use this to
    guide node placement and optimize nodes/weights via min-max relative error.
    """
    n = 8

    def E_of_a(a, w0, wa):
        """Hubble parameter normalized by H0 for CPL model."""
        a = np.maximum(a, 1e-15)
        if abs(w0 + 1.0) < 1e-10:
            return np.sqrt(a**(-3) + (1.0 + wa) * np.exp(-3.0 * wa * (1.0 - a)))
        else:
            exponent = -3.0 * (1.0 + w0)
            prefactor = (1.0 + w0 + wa)
            exp_term = np.exp(-3.0 * wa * (1.0 - a) / (1.0 + w0))
            return np.sqrt(a**(-3) + prefactor * a**exponent * exp_term)

    def exact_luminosity_distance(z, w0, wa):
        """Compute exact dL/(c/H0) via adaptive quadrature."""
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return (1.0 + zp) * zp / E_of_a(a, w0, wa)
        result, _ = quad(integrand, 0.0, z, limit=200, epsabs=1e-12, epsrel=1e-12)
        return result

    def quadrature_luminosity_distance(z, nodes, weights, w0, wa):
        """Compute quadrature approximation of dL/(c/H0)."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)

    # Broader set of training cosmologies
    training_cosmos = [
        (0.0, 0.0),    # Lambda CDM
        (-0.5, 0.5),   # quintessence
        (-1.5, 0.5),   # phantom
        (-1.0, 0.5),   # w0 = -1
        (-1.0, -0.5),  # w0 = -1, negative wa
        (-0.5, -0.5),
        (-1.5, -0.5),
        (0.0, 0.5),
        (-0.2, 0.3),
        (-0.8, 0.2),
        (-1.2, 0.8),
        (-0.3, -0.3),
    ]

    # Test redshifts - include more for better coverage
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    # Precompute exact values
    exact_values = {}
    for (w0, wa) in training_cosmos:
        exact_values[(w0, wa)] = np.array([exact_luminosity_distance(z, w0, wa) for z in test_zs])

    # Use a rational function insight to initialize nodes.
    # For CPL models, 1/E(a) ~ combination of a^3 and exponential terms.
    # The integrand in z-space: (1+z)*z/E(a(z)) where a=1/(1+z)
    # This suggests clustering nodes at low z (where curvature of integrand is high)
    # and spreading at high z.

    # Initial nodes: Gauss-Legendre-like but biased toward low z
    # Use nodes from a (8,8) Gauss quadrature on [0,1] mapped via a power law
    # to cluster more at low z
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(n)
    # Map from [-1,1] to [0,1]
    x_gl = 0.5 * (x_gl + 1.0)
    # Apply a power-law transformation to cluster nodes at low z
    # z_new = z^alpha with alpha < 1 clusters at low z
    alpha = 0.7
    x_gl = x_gl ** alpha
    # Sort and normalize to [0,1]
    x_gl = np.sort(x_gl)
    x_gl = x_gl / x_gl[-1]  # ensure max is 1
    # Shift slightly inside (0,1)
    x_gl = 1e-4 + x_gl * (1.0 - 2e-4)

    # Initial weights: from Gauss-Legendre, adjusted
    w_gl = w_gl / np.sum(w_gl)

    # Convert to logit/log params for optimization
    # nodes = sigmoid(t) => t = logit(nodes)
    t_init = np.log(x_gl / (1.0 - x_gl))
    log_w_init = np.log(w_gl + 1e-15)

    x0 = np.concatenate([t_init, log_w_init])

    def objective(params):
        t = params[:n]
        log_w = params[n:]

        nodes = 1.0 / (1.0 + np.exp(-t))
        w = np.exp(log_w)

        # Enforce increasing nodes: penalty if not
        diffs = np.diff(nodes)
        if np.any(diffs <= 0):
            return 1e10 + np.sum(np.minimum(diffs, 0)**2)

        # Enforce nodes in (0,1)
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10

        # Normalize weights
        w = w / np.sum(w)

        # Compute worst-case relative error
        max_rel_err = 0.0
        for (w0, wa) in training_cosmos:
            exact = exact_values[(w0, wa)]
            for i_z, z in enumerate(test_zs):
                approx = quadrature_luminosity_distance(z, nodes, w, w0, wa)
                rel_err = abs(approx - exact[i_z]) / max(exact[i_z], 1e-15)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err

        return max_rel_err

    # Optimize with Nelder-Mead, then refine with L-BFGS-B
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 50000})

    t_opt = result.x[:n]
    log_w_opt = result.x[n:]

    nodes = 1.0 / (1.0 + np.exp(-t_opt))
    weights = np.exp(log_w_opt)
    weights = weights / np.sum(weights)

    # Ensure nodes are strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)

    # Ensure strict increase
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8

    # Final normalization
    weights = weights / np.sum(weights)

    # Sanity check: weights positive and sum to 1
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes must be in (0,1)"

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
