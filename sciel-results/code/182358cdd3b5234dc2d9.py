"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 10.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA via state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: cost Hamiltonian
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: mixer (X rotations)
    if abs(beta1) > 1e-12:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_r = psi.reshape(shape)
            p0 = psi_r[:, 0, :].copy()
            p1 = psi_r[:, 1, :].copy()
            psi_r[:, 0, :] = c * p0 - 1j * s * p1
            psi_r[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_r.reshape(-1)
    
    # Layer 2: cost Hamiltonian
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: mixer (X rotations)
    if abs(beta2) > 1e-12:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_r = psi.reshape(shape)
            p0 = psi_r[:, 0, :].copy()
            p1 = psi_r[:, 1, :].copy()
            psi_r[:, 0, :] = c * p0 - 1j * s * p1
            psi_r[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_r.reshape(-1)
    
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using physics-informed initialization
    and L-BFGS-B local optimization.
    """
    
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    # Physics-informed starting points based on asymptotic QAOA for MaxCut
    # For p=2, literature suggests:
    # gamma1 ~ 0.5-0.7, beta1 ~ 0.3-0.5
    # gamma2 ~ 0.3-0.5, beta2 ~ 0.2-0.4
    # Also include some spread for small graphs
    inits = [
        [0.55, 0.40, 0.35, 0.25],   # asymptotic-ish
        [0.60, 0.35, 0.40, 0.20],
        [0.50, 0.45, 0.30, 0.30],
        [0.65, 0.30, 0.45, 0.15],
        [0.45, 0.50, 0.25, 0.35],
        [0.70, 0.25, 0.50, 0.10],
        [0.40, 0.55, 0.20, 0.40],
        [0.55, 0.40, 0.35, 0.25],  # repeat center
        [0.62, 0.38, 0.38, 0.22],
        [0.48, 0.48, 0.32, 0.28],
    ]
    
    best_x = None
    best_val = np.inf
    
    # Try each starting point with L-BFGS-B
    for init in inits:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                obj,
                init,
                method='L-BFGS-B',
                bounds=[(0.0, np.pi), (0.0, np.pi/2), (0.0, np.pi), (0.0, np.pi/2)],
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
    
    # If L-BFGS-B didn't find anything good, try Nelder-Mead as fallback
    if best_x is None:
        for init in inits[:4]:
            if _remaining() < 5.0:
                break
            try:
                res = minimize(
                    obj,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                continue
    
    # Final high-precision polish with very tight tolerance
    if best_x is not None and _remaining() > 3.0:
        try:
            res_polish = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=[(0.0, np.pi), (0.0, np.pi/2), (0.0, np.pi), (0.0, np.pi/2)],
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        best_x = np.array([0.55, 0.40, 0.35, 0.25])
    
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.55, 0.40], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
