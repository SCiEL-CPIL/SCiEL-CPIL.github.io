"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def E_func(z, Omega_m, w0, w1):
        Omega_L = 1.0 - Omega_m
        exponent = 3.0 * (1.0 + w0 + w1 * z)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L * (1.0 + z)**exponent)

    def true_integral(z, Omega_m, w0, w1):
        integrand = lambda u: 1.0 / E_func(z * u, Omega_m, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
        return res

    # Representative cosmologies - broader coverage
    cosmo_params = [
        (0.3, -1.0, 0.0),
        (0.3, -1.2, 0.0),
        (0.3, -0.8, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.1, 0.3),
        (0.3, -0.9, -0.3),
        (0.25, -1.0, 0.2),
        (0.35, -1.0, -0.2),
        (0.3, -1.3, 0.1),
        (0.3, -0.7, -0.1),
        (0.2, -1.1, 0.4),
        (0.4, -0.9, -0.4),
    ]

    z_test = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    num_cosmo = len(cosmo_params)
    num_z = len(z_test)

    # Precompute true integrals
    true_vals = np.zeros((num_cosmo, num_z))
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)

    if time.time() > deadline:
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # Moment-matching initialization:
    # For each cosmology and z, compute the integral of u^k / E(z*u) for k=0..7
    # Then solve for nodes/weights that match these moments
    # But since nodes must be universal, we average the moment requirements

    def moment_matching_init():
        """Initialize nodes using Gauss-Legendre, then adjust weights via moment matching."""
        # Start with Gauss-Legendre nodes
        n = 8
        x_gl, w_gl = np.polynomial.legendre.leggauss(n)
        x_gl = 0.5 * (x_gl + 1.0)
        w_gl = 0.5 * w_gl

        # For each test case, compute moments m_k = ∫ u^k / E(z*u) du
        # We want Σ w_i x_i^k ≈ average over all cases
        # This is a linear system in weights given fixed nodes
        # But we also want to optimize nodes

        # Use Gauss-Legendre as starting point
        return x_gl.copy(), w_gl.copy()

    nodes_init, weights_init = moment_matching_init()

    def evaluate_params(params):
        """params: 16 floats - 8 node params, 8 weight params"""
        node_params = params[:8]
        weight_params = params[8:]

        # Map to nodes in (0,1) strictly increasing
        # Use log-spaced parameterization for better conditioning
        # nodes[i] = sigmoid(node_params[i]) but ensure ordering
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0]))
        for i in range(1, 8):
            step = 1.0 / (1.0 + np.exp(-node_params[i]))
            nodes[i] = nodes[i-1] + step * (1.0 - nodes[i])

        # Weights via softmax
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)

        # Compute worst-case relative error
        max_err = 0.0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
        return max_err

    # Convert initial nodes/weights to parameter space
    def nodes_to_params(nodes):
        params = np.zeros(8)
        params[0] = np.log(nodes[0] / (1.0 - nodes[0]))
        for i in range(1, 8):
            # nodes[i] = nodes[i-1] + sigmoid(p_i) * (1 - nodes[i-1])
            # sigmoid(p_i) = (nodes[i] - nodes[i-1]) / (1 - nodes[i-1])
            s = (nodes[i] - nodes[i-1]) / (1.0 - nodes[i-1] + 1e-15)
            s = np.clip(s, 1e-10, 1.0 - 1e-10)
            params[i] = np.log(s / (1.0 - s))
        return params

    def weights_to_params(weights):
        return np.log(weights + 1e-15)

    init_params = np.concatenate([nodes_to_params(nodes_init), weights_to_params(weights_init)])

    bounds = [(-6.0, 6.0)] * 16

    # Run differential evolution
    try:
        result = differential_evolution(
            evaluate_params,
            bounds,
            maxiter=150,
            maxfev=2000,
            tol=1e-10,
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.8,
            popsize=15
        )
    except Exception:
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    best_params = result.x

    # Local refinement
    if time.time() < deadline - 15:
        try:
            def local_obj(params):
                return evaluate_params(params)

            local_result = minimize(local_obj, best_params, method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                             'initial_simplex': None})
            if local_result.fun < result.fun:
                best_params = local_result.x
        except Exception:
            pass

    # Second round of local refinement if time allows
    if time.time() < deadline - 10:
        try:
            def local_obj2(params):
                return evaluate_params(params)

            local_result2 = minimize(local_obj2, best_params, method='Powell',
                                     options={'maxiter': 1000, 'ftol': 1e-14})
            if local_result2.fun < result.fun:
                best_params = local_result2.x
        except Exception:
            pass

    # Reconstruct nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:]

    nodes = np.zeros(8)
    nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0]))
    for i in range(1, 8):
        step = 1.0 / (1.0 + np.exp(-node_params[i]))
        nodes[i] = nodes[i-1] + step * (1.0 - nodes[i])

    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)

    # Validate
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1:
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights

    weights = weights / weights.sum()

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
