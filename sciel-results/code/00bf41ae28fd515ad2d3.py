"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

_DEADLINE = 145.0
_SAFETY = 5.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    edges = [(int(u), int(v)) for u, v in edges]
    num_states = 2 ** n
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
    Compute expected cut for 2-layer QAOA.
    Optimized state-vector simulation.
    """
    psi = np.ones(2 ** n, dtype=np.complex128) / np.sqrt(2 ** n)
    
    # Layer 1: Cost unitary
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: Mixer unitary
    if abs(beta1) > 1e-12:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2 ** (n - k - 1), 2, 2 ** k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)
    
    # Layer 2: Cost unitary
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: Mixer unitary
    if abs(beta2) > 1e-12:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2 ** (n - k - 1), 2, 2 ** k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * cuts))

def _local_search_1d(n, cuts, g1, b1, g2, b2, param_idx, bounds, n_steps=40):
    """
    1D local search along one parameter with adaptive step size.
    param_idx: 0=gamma1, 1=beta1, 2=gamma2, 3=beta2
    """
    params = [g1, b1, g2, b2]
    lo, hi = bounds[param_idx]
    best_val = _expected_cut(n, cuts, *params)
    best_param = params[param_idx]
    
    step = (hi - lo) / 10.0
    for _ in range(n_steps):
        if _remaining() < 0.5:
            break
        # Try forward
        params_new = list(params)
        new_val = params[param_idx] + step
        if new_val > hi:
            new_val = hi - 1e-10
        params_new[param_idx] = new_val
        val_f = _expected_cut(n, cuts, *params_new)
        
        # Try backward
        new_val_b = params[param_idx] - step
        if new_val_b < lo:
            new_val_b = lo + 1e-10
        params_new[param_idx] = new_val_b
        val_b = _expected_cut(n, cuts, *params_new)
        
        if val_f > best_val and val_f >= val_b:
            best_val = val_f
            params[param_idx] = new_val
        elif val_b > best_val:
            best_val = val_b
            params[param_idx] = new_val_b
        else:
            # Shrink step if no improvement
            step *= 0.7
            if step < 1e-8:
                break
    
    return params, best_val

def _polish_4d(n, cuts, best_x, bounds):
    """
    High-precision polish using L-BFGS-B followed by coordinate descent.
    """
    def obj(x):
        if _remaining() < 0.5:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    best_val = -obj(best_x)
    
    # L-BFGS-B polish
    try:
        res = minimize(
            obj,
            np.array(best_x),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.fun < 0:
            candidate_val = -res.fun
            if candidate_val > best_val:
                best_val = candidate_val
                best_x = res.x
    except Exception:
        pass
    
    # Coordinate descent polish
    for _ in range(3):
        if _remaining() < 1.0:
            break
        for idx in range(4):
            if _remaining() < 0.5:
                break
            params, val = _local_search_1d(n, cuts, *best_x, param_idx=idx, bounds=bounds, n_steps=30)
            if val > best_val:
                best_val = val
                best_x = params
    
    return best_x, best_val

def angles(n, edges, p):
    n = int(n)
    edges = [(int(u), int(v)) for u, v in edges]
    
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Objective function for minimizers (we want to maximize expected cut)
    def obj(x):
        if _remaining() < 0.5:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi / 2, np.pi, np.pi / 2]
    bounds = list(zip(lower, upper))
    
    # Initial best from a reasonable starting point
    best_x = [0.5, 0.5, 0.5, 0.5]
    best_val = _expected_cut(n, cuts, *best_x)
    
    # Use Differential Evolution for global search
    time_per_eval_est = 0.01  # Estimate ~10ms per eval for n=10
    remaining = _remaining()
    max_iters = max(10, int(remaining / (20 * 4 * time_per_eval_est)))
    
    try:
        res = differential_evolution(
            obj,
            bounds,
            seed=42,
            maxiter=max_iters,
            popsize=20,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if res.fun < 0:
            candidate_x = res.x
            candidate_val = -res.fun
            if candidate_val > best_val:
                best_val = candidate_val
                best_x = candidate_x
    except Exception:
        pass
    
    # Final high-precision polish on the best found
    if _remaining() > 2.0:
        try:
            best_x, best_val = _polish_4d(n, cuts, best_x, bounds)
        except Exception:
            pass
    
    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
