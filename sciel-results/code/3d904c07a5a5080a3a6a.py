"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10)
    exponent = wa / (1.0 + w0)
    return a**(-3.0) * base**exponent


def _integrand_1_over_E(z, w0, wa):
    """1/E(z) for CPL model, where z is redshift."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)


def _exact_integral_fast(z, w0, wa, n_quad=256):
    """Fast high-accuracy integral of 1/E(z') from 0 to z using Gauss-Legendre."""
    if z <= 0:
        return 0.0
    # Use Gauss-Legendre on [0, z]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1,1] to [0, z]
    mapped = 0.5 * z * (gl_nodes + 1.0)
    w_mapped = 0.5 * z * gl_weights
    vals = _integrand_1_over_E(mapped, w0, wa)
    return float(np.sum(w_mapped * vals))


def _quad_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of integral_0^z dz'/E(z')."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_nodes, w0, wa)
    return z * np.sum(weights / E_vals)


def _relative_error(z, nodes, weights, w0, wa):
    """Compute relative error of quadrature for given z and cosmology."""
    exact = _exact_integral_fast(z, w0, wa)
    approx = _quad_value(z, nodes, weights, w0, wa)
    if abs(exact) < 1e-15:
        return 0.0
    return abs((approx - exact) / exact)


def _worst_error(nodes, weights, models, z_test):
    """Compute worst-case relative error across all models and redshifts."""
    max_err = 0.0
    for w0, wa in models:
        for z in z_test:
            err = _relative_error(z, nodes, weights, w0, wa)
            if err > max_err:
                max_err = err
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Representative CPL models covering a broad range
    models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.0, 0.5),
        (-0.8, -0.5),
        (-1.2, 0.3),
        (-0.3, -0.3),
        (-1.0, -0.5),
        (-0.7, 0.2),
        (-1.1, -0.2),
        (-0.9, 0.4),
        (-0.6, -0.8),
    ]

    # Dense test redshifts
    z_test = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])

    # Initial guess: Gauss-Legendre nodes on [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    nodes0 = np.clip(nodes0, 1e-6, 1.0 - 1e-6)
    nodes0 = np.sort(nodes0)

    # Parameterization: use logit transform for nodes to enforce (0,1) and ordering
    # x_i = logit(node_i) = log(node_i / (1 - node_i))
    # To enforce ordering, use x_1 < x_2 < ... < x_8
    # Parameterize as: x_1 = p_0, x_{i+1} = x_i + exp(p_i) for i=1..7
    # This ensures strict ordering automatically
    # Weights: use log-weights, normalize at end

    def _params_to_nodes_weights(params):
        """Convert unconstrained params to valid nodes and weights."""
        # params[0:8] = node parameters (logit-space with cumulative gaps)
        # params[8:16] = log weights
        x0 = params[0]
        log_gaps = params[1:8]
        gaps = np.exp(log_gaps)  # positive gaps
        # Build x values: x_1 = x0, x_{i+1} = x_i + gap_i
        x = np.zeros(8)
        x[0] = x0
        for i in range(1, 8):
            x[i] = x[i-1] + gaps[i-1]
        
        # Convert logit to nodes: node = sigmoid(x) = 1/(1+exp(-x))
        nodes = 1.0 / (1.0 + np.exp(-x))
        
        # Ensure nodes are strictly in (0,1) - sigmoid guarantees this
        # But clip to avoid numerical issues
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Weights
        log_w = params[8:16]
        weights = np.exp(log_w)
        weights = weights / np.sum(weights)
        
        return nodes, weights

    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        return _worst_error(nodes, weights, models, z_test)

    # Initial params
    logit_nodes0 = np.log(nodes0 / (1.0 - nodes0))
    gaps0 = np.diff(logit_nodes0)
    x0 = logit_nodes0[0]
    log_gaps0 = np.log(np.maximum(gaps0, 1e-6))
    log_w0 = np.log(weights0)
    params0 = np.concatenate([[x0], log_gaps0, log_w0])

    best_params = params0.copy()
    best_score = _objective(params0)

    # Phase 1: Nelder-Mead on reduced problem
    try:
        result = minimize(
            _objective,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Phase 2: Local refinement with smaller step
    if time.time() < deadline - 10.0:
        try:
            result2 = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass

    # Phase 3: Random restarts if time permits
    if time.time() < deadline - 5.0:
        np.random.seed(42)
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            # Perturb best solution
            perturbation = np.random.randn(16) * 0.1
            trial_params = best_params + perturbation
            trial_score = _objective(trial_params)
            if trial_score < best_score:
                best_score = trial_score
                best_params = trial_params.copy()
            else:
                # Try smaller perturbation
                perturbation = np.random.randn(16) * 0.01
                trial_params = best_params + perturbation
                trial_score = _objective(trial_params)
                if trial_score < best_score:
                    best_score = trial_score
                    best_params = trial_params.copy()

    # Final extraction
    nodes_opt, weights_opt = _params_to_nodes_weights(best_params)

    # Ensure validity
    if np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and np.all(np.diff(nodes_opt) > 0):
        # Normalize weights to sum to 1
        weights_opt = weights_opt / np.sum(weights_opt)
        return {"nodes": nodes_opt, "weights": weights_opt}

    # Fallback
    nodes0 = np.clip(nodes0, 1e-10, 1.0 - 1e-10)
    weights0 = weights0 / np.sum(weights0)
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END
