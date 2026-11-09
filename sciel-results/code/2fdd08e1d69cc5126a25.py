"""Model-and-optimize: Chebyshev moment-matching with Remez-like refinement for luminosity distance quadrature."""
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0


def _time_left():
    return _DEADLINE - time.time()


def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)


def _true_dL(z, w0, wa):
    def integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, w0, wa)
    val, _ = quad(integrand, 0, z, limit=200, epsabs=1e-14, epsrel=1e-14)
    return (1.0 + z) * val


def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx


def _chebyshev_moments(n_nodes=8, z_max=10.0, n_cheb=20):
    """
    Compute Chebyshev moment integrals:
    For the integral ∫_0^z 1/E(a) dz', we want nodes/weights that match
    Chebyshev polynomial moments of the integration kernel.
    
    We use the fact that for a quadrature with nodes t_i in [0,1] and weights w_i,
    the approximation is: ∑ w_i f(t_i) ≈ ∫_0^1 f(t) dt
    where f(t) = 1/E(1/(1+zt)) * z / (1+zt) ... actually let's reformulate.
    
    The luminosity distance integral in the variable t = z'/z (so z' = zt, t ∈ [0,1]):
    dL/(c/H0) = (1+z) * ∫_0^1 z/(1+zt) * 1/E(1/(1+zt)) dt
    
    So the integrand is g_z(t) = z/(1+zt) * 1/E(1/(1+zt))
    
    We want ∑ w_i g_z(t_i) ≈ ∫_0^1 g_z(t) dt for all z and all (w0,wa).
    
    For moment matching, we expand g_z(t) in Chebyshev polynomials and match moments.
    But g_z depends on z and cosmology, so we can't match all moments exactly.
    
    Instead, we use a different approach: optimize for equal error (Remez-like)
    across a grid of (z, w0, wa) points.
    """
    pass


def _compute_rel_errors(nodes, weights, test_cosmos, test_zs):
    """Compute relative errors on a grid of cosmologies and redshifts."""
    errors = []
    meta = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
                meta.append((w0, wa, z))
    return np.array(errors), meta


def _solve_weights_lp(nodes, test_cosmos, test_zs, max_iter=50):
    """
    Given fixed nodes, find optimal positive weights that minimize worst-case relative error.
    Uses iterative reweighting / LP-like approach.
    """
    n = len(nodes)
    # Precompute E values for all (cosmo, z, node) combinations
    E_data = []
    true_vals = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            E_data.append(1.0 / E_vals)
            true_val = _true_dL(z, w0, wa)
            true_vals.append(true_val)
    
    E_data = np.array(E_data)  # shape: (n_points, n_nodes)
    true_vals = np.array(true_vals)
    
    # Initial weights: uniform
    weights = np.ones(n) / n
    
    for iteration in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Compute quadrature values
        quad_vals = E_data @ weights  # shape: (n_points,)
        quad_vals = (1.0 + test_zs_grid[test_zs_idx]) * test_zs_grid[test_zs_idx] * quad_vals
        # Actually need to be more careful with the (1+z)*z factor
        # Let's restructure: for each point (w0, wa, z), quad = (1+z)*z * sum(w_i / E_i)
        
        # Recompute properly
        quad_vals = np.zeros(len(true_vals))
        idx = 0
        for w0, wa in test_cosmos:
            for z in test_zs:
                quad_vals[idx] = (1.0 + z) * z * np.sum(weights / _E_cpl(1.0/(1.0 + z*nodes), w0, wa))
                idx += 1
        
        rel_errors = np.abs(quad_vals - true_vals) / np.maximum(true_vals, 1e-30)
        
        # Find worst case
        max_idx = np.argmax(rel_errors)
        max_err = rel_errors[max_idx]
        
        # Update weights using projected gradient on the constraint sum(w)=1, w>0
        # Gradient of worst-case error w.r.t. weights
        # d/dw_i |quad - true|/true * sign(quad - true)
        w0_wa = test_cosmos[max_idx // len(test_zs)]
        z_worst = test_zs[max_idx % len(test_zs)]
        sign = np.sign(quad_vals[max_idx] - true_vals[max_idx])
        grad = sign * (1.0 + z_worst) * z_worst / E_data[max_idx] / true_vals[max_idx]
        
        # Projected gradient step
        step_size = 0.01 / (max_err + 1e-15)
        weights -= step_size * grad
        # Project to simplex
        weights = np.maximum(weights, 1e-10)
        weights /= np.sum(weights)
        
        if max_err < 1e-14:
            break
    
    return weights


# Precompute test data
_TEST_COSMOS = [
    (0.0, 0.0),
    (-1.0, 0.0),
    (0.5, -1.0),
    (-1.5, 0.5),
    (-0.5, 0.5),
    (0.3, -0.3),
    (-2.0, 1.0),
    (1.0, -2.0),
    (-0.8, -0.2),
    (0.2, 0.1),
]
_TEST_ZS = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])


def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights


def _moment_match_init():
    """
    Initialize nodes using moment matching against a representative integrand.
    We match the first 8 moments of a typical 1/E(a) profile.
    """
    # Use LCDM as reference
    def ref_integrand(z_prime):
        a = 1.0 / (1.0 + z_prime)
        return 1.0 / _E_cpl(a, -1.0, 0.0)
    
    # We want nodes t_i in [0,1] and weights w_i such that
    # sum w_i * t_i^k ≈ ∫_0^1 t^k * (1/(1+t)) * 1/E(1/(1+t)) dt for k=0,...,7
    # (The 1/(1+t) comes from the change of variables z' = z*t)
    
    # Actually, let's just use Gauss-Legendre as a good start
    return _gauss_legendre_init()


def _remez_refine(nodes, weights, n_iter=30):
    """
    Remez-like refinement: iteratively adjust nodes and weights to equalize
    the maximum relative error across the test grid.
    """
    for it in range(n_iter):
        if _time_left() < 5.0:
            break
        
        errors, meta = _compute_rel_errors(nodes, weights, _TEST_COSMOS, _TEST_ZS)
        max_err = np.max(errors)
        
        if max_err < 1e-13:
            break
        
        # Find the worst-case point
        max_idx = np.argmax(errors)
        w0, wa, z = meta[max_idx]
        
        # Compute gradient of error w.r.t. nodes and weights
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = _E_cpl(a_vals, w0, wa)
        inv_E = 1.0 / E_vals
        
        quad_val = (1.0 + z) * z * np.sum(weights * inv_E)
        true_val = _true_dL(z, w0, wa)
        err = abs(quad_val - true_val)
        sign = np.sign(quad_val - true_val)
        
        # d(err)/d(w_j) = sign * (1+z)*z * inv_E[j] / true_val
        grad_w = sign * (1.0 + z) * z * inv_E / true_val
        
        # d(err)/d(t_j): t_j appears in a_j = 1/(1+z*t_j)
        # da_j/dt_j = -z/(1+z*t_j)^2 = -z*a_j^2
        # d(1/E)/dt_j = -d(1/E)/da * da/dt = dE/da / E^2 * (-z*a^2) ... this is complex
        # For now, just adjust weights
        step = 0.005
        weights -= step * grad_w
        weights = np.maximum(weights, 1e-10)
        weights /= np.sum(weights)
        
        # Small node perturbation toward equalizing error
        # Move nodes slightly toward where the integrand varies most
        for j in range(8):
            # Perturb node j slightly in direction that reduces local error
            delta_t = 0.001 * sign * inv_E[j] / np.sum(inv_E * weights)
            nodes[j] += delta_t
            nodes[j] = np.clip(nodes[j], 1e-10, 1.0 - 1e-10)
        
        # Enforce ordering
        for j in range(1, 8):
            if nodes[j] <= nodes[j-1] + 1e-12:
                nodes[j] = nodes[j-1] + 1e-12
    
    return nodes, weights


def _nelder_mead_optimize(nodes, weights):
    """Full Nelder-Mead on 16 params."""
    def decode(params):
        raw = params[:8]
        sig = 1.0 / (1.0 + np.exp(-10.0 * (raw - 3.5)))
        sig = np.sort(sig)
        for i in range(1, 8):
            if sig[i] <= sig[i-1] + 1e-12:
                sig[i] = sig[i-1] + 1e-12
        sig = np.clip(sig, 1e-10, 1.0 - 1e-10)
        nodes_d = sig
        
        log_w = params[8:16]
        w_raw = np.exp(log_w)
        weights_d = w_raw / np.sum(w_raw)
        return nodes_d, weights_d
    
    def objective(params):
        if _time_left() < 3.0:
            return 1e10
        nd, wd = decode(params)
        errors, _ = _compute_rel_errors(nd, wd, _TEST_COSMOS, _TEST_ZS)
        # Use soft max
        beta = 50.0
        max_e = np.max(errors)
        if max_e < 1e-15:
            return 0.0
        log_sum = np.log(np.sum(np.exp(beta * (errors - max_e)))) / beta + max_e
        return log_sum
    
    def encode(nd, wd):
        pn = np.log(nd / (1.0 - nd)) / 10.0 + 3.5
        pw = np.log(wd)
        return np.concatenate([pn, pw])
    
    x0 = encode(nodes, weights)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < 1e-3:
            nd, wd = decode(res.x)
            return nd, wd
    except Exception:
        pass
    
    return nodes, weights


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Start from Gauss-Legendre
    nodes, weights = _gauss_legendre_init()
    
    # Phase 1: Solve optimal weights for GL nodes
    weights = _solve_weights_lp(nodes, _TEST_COSMOS, _TEST_ZS, max_iter=30)
    nodes, weights = _decode_fix(nodes, weights)
    
    # Phase 2: Remez-like refinement
    if _time_left() > 20.0:
        nodes, weights = _remez_refine(nodes, weights, n_iter=20)
    
    # Phase 3: Nelder-Mead fine-tuning
    if _time_left() > 25.0:
        nodes, weights = _nelder_mead_optimize(nodes, weights)
    
    # Final validation and cleanup
    nodes, weights = _decode_fix(nodes, weights)
    
    return {"nodes": nodes, "weights": weights}


def _decode_fix(nodes, weights):
    nodes = np.array(nodes, dtype=float)
    weights = np.array(weights, dtype=float)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    nodes.sort()
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-12
    weights = np.abs(weights)
    if np.sum(weights) < 1e-15:
        weights = np.ones(8) / 8.0
    weights = weights / np.sum(weights)
    return nodes, weights
