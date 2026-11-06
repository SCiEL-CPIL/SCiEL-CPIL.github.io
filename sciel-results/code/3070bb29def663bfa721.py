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
    Compute expected cut for 2-layer QAOA.
    Optimized state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: Cost unitary
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: Mixer unitary
    if abs(beta1) > 1e-9:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)
            
    # Layer 2: Cost unitary
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: Mixer unitary
    if abs(beta2) > 1e-9:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _local_search_1d(n, cuts, g1, b1, g2, b2, param_idx, bounds, n_steps=50):
    """
    1D local search along one parameter with adaptive step size.
    param_idx: 0=gamma1, 1=beta1, 2=gamma2, 3=beta2
    """
    params = [g1, b1, g2, b2]
    lo, hi = bounds[param_idx]
    best_val = _expected_cut(n, cuts, *params)
    best_param = params[param_idx]
    
    # Adaptive step: start with 10% of range, shrink if improving
    step = (hi - lo) / 10.0
    for _ in range(n_steps):
        if _remaining() < 0.5:
            break
        new_val = _expected_cut(n, cuts, *params)
        # Try forward
        params_new = params.copy()
        params_new[param_idx] = min(hi, params_new[param_idx] + step)
        if abs(params_new[param_idx] - params[param_idx]) < 1e-10:
            # Try backward
            params_new[param_idx] = max(lo, params[param_idx] - step)
        val_new = _expected_cut(n, cuts, *params_new)
        if val_new > best_val + 1e-12:
            params = params_new
            best_val = val_new
            best_param = params_new[param_idx]
            # Keep step same or slightly larger
        else:
            # Shrink step
            step *= 0.7
            if step < 1e-8:
                break
            # Try backward
            params_new[param_idx] = max(lo, params[param_idx] - step)
            val_new = _expected_cut(n, cuts, *params_new)
            if val_new > best_val + 1e-12:
                params = params_new
                best_val = val_new
                best_param = params_new[param_idx]
    
    return params, best_val

def _polish_4d(n, cuts, x0, bounds, max_iter=200):
    """
    High-precision 4D polish using Nelder-Mead with tight tolerances.
    """
    def obj(x):
        if _remaining() < 0.5:
            return 1e10
        g1, b1, g2, b2 = x
        # Clamp to bounds
        g1 = np.clip(g1, bounds[0][0], bounds[0][1])
        b1 = np.clip(b1, bounds[1][0], bounds[1][1])
        g2 = np.clip(g2, bounds[2][0], bounds[2][1])
        b2 = np.clip(b2, bounds[3][0], bounds[3][1])
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    try:
        res = minimize(
            obj,
            np.array(x0, dtype=np.float64),
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        if res.fun < 0:
            return list(np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds]))
    except Exception:
        pass
    return list(x0)

def angles(n, edges, p):
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
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    bounds = list(zip(lower, upper))
    
    best_x = [0.5, 0.5, 0.5, 0.5]
    best_val = _expected_cut(n, cuts, *best_x)
    
    # Use Differential Evolution for global search
    # Use a smaller population but more iterations for better coverage
    time_per_eval_est = 0.01  # Estimate ~10ms per eval for n=10
    remaining = _remaining()
    # DE evaluates popsize * 4 candidates per iteration roughly
    max_iters = max(15, int(remaining / (25 * 4 * time_per_eval_est)))
    
    try:
        res = differential_evolution(
            obj, 
            bounds, 
            seed=713, 
            maxiter=max_iters, 
            popsize=25, 
            tol=1e-8, 
            polish=False,  # We do our own polish
            init='sobol',
            strategy='best1bin',
            mutation=(0.5, 1.5),
            recombination=0.8
        )
        
        if res.fun < 0:  # Valid result
            candidate_x = list(res.x)
            candidate_val = -res.fun
            if candidate_val > best_val:
                best_val = candidate_val
                best_x = candidate_x
    except Exception:
        pass
    
    # Local 1D search on each parameter from the best point
    if _remaining() > 3.0:
        for i in range(4):
            if _remaining() < 1.0:
                break
            best_x, best_val = _local_search_1d(n, cuts, *best_x, param_idx=i, bounds=bounds, n_steps=60)
    
    # Final high-precision polish using Nelder-Mead
    if _remaining() > 2.0:
        polished_x = _polish_4d(n, cuts, best_x, bounds, max_iter=300)
        polished_val = _expected_cut(n, cuts, *polished_x)
        if polished_val > best_val:
            best_val = polished_val
            best_x = polished_x
    
    # One more round of 1D + polish if time permits
    if _remaining() > 3.0:
        for i in range(4):
            if _remaining() < 1.0:
                break
            best_x, best_val = _local_search_1d(n, cuts, *best_x, param_idx=i, bounds=bounds, n_steps=40)
        
        if _remaining() > 2.0:
            polished_x = _polish_4d(n, cuts, best_x, bounds, max_iter=200)
            polished_val = _expected_cut(n, cuts, *polished_x)
            if polished_val > best_val:
                best_val = polished_val
                best_x = polished_x
    
    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
