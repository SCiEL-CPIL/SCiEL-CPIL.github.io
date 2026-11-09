"""Rewritten program with vectorized QAOA simulation and Nelder-Mead multi-restart optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 135s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 135.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        # Vectorized: create array of bit values for all states
        states = np.arange(dim)
        bi = ((states >> i) & 1).astype(np.float64)
        bj = ((states >> j) & 1).astype(np.float64)
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=200, tol=1e-10):
    """Run Nelder-Mead optimization with time guard."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _polish(x0, n, cut_sizes, p, maxiter=300, tol=1e-12):
    """High-precision polish pass using Nelder-Mead with tighter tolerance."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 135.0
    
    # 1. Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # 2. Generate diverse initial guesses
    np.random.seed(42)
    initial_guesses = []
    
    # Standard QAOA angles
    initial_guesses.append(np.array([0.35, 0.25] * p))
    
    # Additional heuristic starts
    initial_guesses.append(np.array([0.5, 0.3] * p))
    initial_guesses.append(np.array([0.2, 0.4] * p))
    initial_guesses.append(np.array([0.7, 0.2] * p))
    
    # Random guesses
    for _ in range(30):
        gammas = np.random.uniform(0, np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # 3. Multi-restart Nelder-Mead optimization
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
        
        x_opt, score = _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=150, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # 4. Final high-precision polish on the best result
    if best_angles is not None and _time_left() > 20.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=300, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
