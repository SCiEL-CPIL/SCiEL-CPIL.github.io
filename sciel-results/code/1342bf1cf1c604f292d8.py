"""Rewritten program with optimized QAOA angle design using fast simulation and multi-restart optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer_fast(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        # Move axis i to 0 to apply the 2x2 rotation
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        # Apply rotation matrix [[c, -is], [-is, c]]
        # new_0 = c * old_0 - is * old_1
        # new_1 = -is * old_0 + c * old_1
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
        state = _apply_x_layer_fast(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """
    Compute objective (negative expected cut) and its gradient using finite differences
    for robustness, or analytic if available. Here we use analytic for speed if possible,
    but finite difference is safer if analytic is buggy.
    Given the low dimensionality (4 params), finite difference is very fast.
    """
    angles = params.reshape(p, 2)
    exp_cut = _expected_cut(n, cut_sizes, angles)
    
    # Gradient via central differences
    eps = 1e-6
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[k] += eps
        params_minus[k] -= eps
        
        f_plus = _expected_cut(n, cut_sizes, params_plus.reshape(p, 2))
        f_minus = _expected_cut(n, cut_sizes, params_minus.reshape(p, 2))
        
        grad[k] = (f_plus - f_minus) / (2 * eps)
        
    # Minimize -exp_cut
    return -exp_cut, -grad


def _optimize_local(x0, n, cut_sizes, p, maxiter=500):
    """
    Optimize using L-BFGS-B with finite difference gradients.
    """
    try:
        res = minimize(
            lambda x: _objective(x, n, cut_sizes, p)[0],
            x0,
            jac=lambda x: _objective(x, n, cut_sizes, p)[1],
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-10
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _generate_initial_guesses(n, edges, p):
    """
    Generate a diverse set of initial guesses.
    """
    num_edges = len(edges)
    max_possible_edges = n * (n - 1) // 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0.5
    
    inits = []
    
    # 1. Density-based heuristic
    g1 = np.pi * density / 2
    b1 = np.pi * (1 - density) / 4
    inits.append(np.array([g1, b1, g1, b1]))
    
    # 2. Common optimal regions for p=2
    inits.append(np.array([0.4, 0.2, 0.4, 0.2]))
    inits.append(np.array([0.5, 0.3, 0.5, 0.3]))
    inits.append(np.array([0.3, 0.1, 0.3, 0.1]))
    inits.append(np.array([0.6, 0.4, 0.6, 0.4]))
    
    # 3. Randomized starts
    np.random.seed(42)
    for _ in range(10):
        g = np.random.uniform(0.1, 0.9, p)
        b = np.random.uniform(0.05, 0.6, p)
        inits.append(np.column_stack((g, b)).flatten())
        
    return inits


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Generate initial guesses
    initial_guesses = _generate_initial_guesses(n, edges, p)
    
    best_angles = None
    best_score = -np.inf
    
    # Sort initial guesses by a quick heuristic score to prioritize good starts
    # We can just evaluate them all quickly
    scored_guesses = []
    for x0 in initial_guesses:
        score = _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        scored_guesses.append((score, x0))
    
    # Sort by score descending
    scored_guesses.sort(key=lambda x: x[0], reverse=True)
    
    # Try optimizing from the best starts first
    for score, x0 in scored_guesses:
        if _time_left() < 5.0:
            break
            
        x_opt, opt_score = _optimize_local(x0, n, cut_sizes, p, maxiter=200)
        
        if opt_score > best_score:
            best_score = opt_score
            best_angles = x_opt.reshape(p, 2)
            
    # Final polish with tighter tolerance if time permits
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _optimize_local(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
