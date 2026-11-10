"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_legendre
from scipy.optimize import minimize
from scipy.integrate import quad


def _cosmology_E(z, w0, wa, Om_m=0.3):
    """CPL dark energy: E(z)^2 = Om_m*(1+z)^3 + (1-Om_m)*(1+z)^(3*(1+w0)) * (1 - wa*(1-1/(1+z)))^(1.5*wa)"""
    z = np.atleast_1d(z)
    one_plus_z = 1.0 + z
    matter = Om_m * one_plus_z**3
    de_base = (1.0 - Om_m) * one_plus_z**(3.0 * (1.0 + w0))
    de_arg = 1.0 - wa * (1.0 - 1.0 / one_plus_z)
    de_arg = np.maximum(de_arg, 1e-30)
    de = de_base * de_arg**(1.5 * wa)
    return np.sqrt(matter + de)


def _integrand_z(z, w0, wa, Om_m):
    """f(z) = z / E(z) for the integral I(zmax) = ∫_0^zmax z'/E(z') dz'"""
    if z < 1e-15:
        return 0.0
    E = _cosmology_E(z, w0, wa, Om_m)
    return z / E


def _true_integral(zmax, w0, wa, Om_m):
    """Adaptive quadrature reference."""
    if zmax < 1e-15:
        return 0.0
    result, _ = quad(_integrand_z, 0, zmax, args=(w0, wa, Om_m), limit=200, epsabs=1e-13, epsrel=1e-13)
    return result


def _rel_error(zmax, nodes, weights, w0, wa, Om_m):
    """Relative error in percent for quadrature vs true integral."""
    # Quadrature: I ≈ zmax * sum(wi * f(zmax*ti)) where f(z) = z/E(z)
    z_i = zmax * nodes
    E_i = _cosmology_E(z_i, w0, wa, Om_m)
    f_i = z_i / E_i
    I_quad = zmax * np.sum(weights * f_i)
    I_true = _true_integral(zmax, w0, wa, Om_m)
    dL_quad = (1.0 + zmax) * I_quad
    dL_true = (1.0 + zmax) * I_true
    if abs(dL_true) < 1e-15:
        return 0.0
    return abs(dL_quad - dL_true) / abs(dL_true) * 100.0


def _objective(params):
    """Worst-case relative error across cosmologies and z values."""
    # Decode: nodes from log-spacings, weights from log-weights
    s = params[:8]
    w = params[8:16]
    
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-15):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    
    # Use a diverse set of CPL cosmologies
    cosmologies = [
        (0.0, 0.0, 0.3),      # LCDM
        (-1.0, 0.0, 0.3),     # Cosmological constant
        (-0.5, 0.5, 0.3),     # Quintessence
        (0.3, -0.6, 0.3),     # Phantom-ish
        (-0.8, 0.3, 0.3),     # Another quintessence
        (0.0, -0.5, 0.3),     # Phantom
        (-0.5, 0.0, 0.3),     # Simple quintessence
        (-1.0, 0.5, 0.3),     # CC with evolution
        (0.0, 0.3, 0.3),      # Matter-dominated-ish
        (-0.3, -0.3, 0.3),    # Another phantom
    ]
    
    # Range of redshifts to test
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    
    max_err = 0.0
    for w0, wa, Om in cosmologies:
        for z in z_values:
            err = _rel_error(z, nodes, weights, w0, wa, Om)
            if err > max_err:
                max_err = err
            if max_err > 100.0:
                return max_err
    return max_err


def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Starts from Gauss-Legendre and refines via optimization to minimize worst-case
    relative error across diverse CPL cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Start from Gauss-Legendre
    x, w = roots_legendre(8)
    nodes_gl = 0.5 * (x + 1.0)
    weights_gl = 0.5 * w
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Encode GL as parameters
    spacings_gl = np.diff(np.concatenate([[0], nodes_gl]))
    s_init = np.log(spacings_gl)
    w_init = np.log(weights_gl)
    x0 = np.concatenate([s_init, w_init])
    
    best_params = x0.copy()
    best_score = _objective(x0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        # Not enough time, return GL
        pass
    else:
        # Nelder-Mead optimization
        try:
            result = minimize(
                _objective, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            elapsed = time.time() - start_time
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Try a few random restarts if time allows
        elapsed = time.time() - start_time
        rng = np.random.default_rng(42)
        n_restarts = 3
        for _ in range(n_restarts):
            if time.time() - start_time > deadline - 20:
                break
            x_rand = x0 + rng.normal(0, 0.1, size=16)
            try:
                result = minimize(
                    _objective, x_rand, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
    
    # Decode best parameters
    s = best_params[:8]
    w = best_params[8:16]
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    # Final safety: if anything went wrong, fall back to GL
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0) and np.all(weights > 0)):
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes must be in (0,1)"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
