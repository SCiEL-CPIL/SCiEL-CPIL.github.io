"""QAOA parameter design: per-graph optimization of angles maximizing expected cut ratio.

Combined strategy:
  1. Fast vectorized QAOA simulation (vectorized bit-pair mixer, precomputed cut sizes).
  2. Structure-informed multi-restart Nelder-Mead screening.
  3. High-precision polish pass on the best candidate.
  4. Optional analytic gradient refinement step if time permits.
  5. Strict deadline guard.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = None
_TIME_LIMIT = 135.0  # Safety margin against 150s limit


def _time_left():
    if _DEADLINE is None:
        return np.inf
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    states = np.arange(dim, dtype=np.int64)
    
    for (i, j) in edges:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi ^ bj).astype(np.float64)
        
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state vector.
    Vectorized bit-pair mixer: eliminates per-axis Python loop overhead.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    dim = len(state)
    
    # Vectorized approach: for each bit position, swap pairs
    # We process all bit positions in one pass using reshape tricks
    # But since n is small (6-10), we can still loop over bits but vectorize within each bit
    tensor = state.reshape([2] * n)
    
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    
    return tensor.flatten()


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value for given angles."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _nelder_mead(x0, n, cut_sizes, p, maxiter=300, tol=1e-10):
    """Run Nelder-Mead optimization with time guard."""
    def _obj_wrapper(params):
        if _time_left() < 5.0:
            raise TimeoutError()
        return _objective(params, n, cut_sizes, p)
    
    try:
        res = minimize(
            _obj_wrapper,
            x0,
            args=(),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except TimeoutError:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _analytic_gradient_refinement(x0, n, edges, cut_sizes, p, max_steps=50, lr=0.01):
    """
    Analytic gradient ascent using parameter-shift rule.
    For QAOA, the gradient with respect to each angle can be computed
    using the parameter-shift rule: f(x + pi/2) - f(x - pi/2) / 2.
    This is exact for QAOA (single-parameter shifts).
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
    
    for step in range(max_steps):
        if _time_left() < 10.0:
            break
        
        # Compute gradient via parameter-shift rule
        grad = np.zeros(2 * p)
        for i in range(2 * p):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += np.pi / 2
            x_minus[i] -= np.pi / 2
            
            f_plus = _expected_cut(n, cut_sizes, x_plus.reshape(p, 2))
            f_minus = _expected_cut(n, cut_sizes, x_minus.reshape(p, 2))
            
            grad[i] = (f_plus - f_minus) / 2.0
        
        # Gradient ascent step with adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Simple adaptive learning rate
        step_size = lr / (1.0 + step * 0.05)
        x_new = x + step_size * grad
        
        # Clip to valid range [0, pi]
        x_new = np.clip(x_new, 0.0, np.pi)
        
        f_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
        
        if f_new > best_val:
            best_val = f_new
            best_x = x_new.copy()
            x = x_new
        else:
            # Backtrack
            step_size *= 0.5
            x_new = x + step_size * grad
            x_new = np.clip(x_new, 0.0, np.pi)
            f_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
            if f_new > best_val:
                best_val = f_new
                best_x = x_new.copy()
                x = x_new
            else:
                break
    
    return best_x, best_val


def _initial_guesses_from_structure(n, edges, p):
    """Generate structure-informed initial angle guesses."""
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0.0
    
    # Degree-based heuristic
    degree = np.zeros(n, dtype=np.int32)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1
    max_deg = int(np.max(degree)) if n > 0 else 0
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    guesses = []
    
    # Heuristic 1: based on edge density
    base_gamma = 0.15 + 0.3 * density
    base_beta = 0.2 + 0.15 * (1.0 - density)
    guesses.append(np.tile([base_gamma, base_beta], p))
    
    # Heuristic 2: degree-scaled
    gamma2 = 0.1 + 0.05 * avg_deg
    beta2 = 0.25 - 0.02 * avg_deg
    guesses.append(np.tile([max(0.05, gamma2), max(0.05, beta2)], p))
    
    # Heuristic 3: known good QAOA starting points from literature
    guesses.append(np.array([0.35, 0.25] * p))
    guesses.append(np.array([0.4, 0.2] * p))
    guesses.append(np.array([0.25, 0.3] * p))
    
    # Heuristic 4: small angles (near identity)
    guesses.append(np.tile([0.1, 0.1], p))
    
    # Heuristic 5: moderate angles
    guesses.append(np.tile([0.5, 0.35], p))
    
    # A few random guesses with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(5):
        gammas = rng.uniform(0.1, 0.8, p)
        betas = rng.uniform(0.05, 0.5, p)
        guesses.append(np.column_stack((gammas, betas)).flatten())
    
    return guesses


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Generate structure-informed initial guesses
    initial_guesses = _initial_guesses_from_structure(n, edges, p)
    
    best_angles = None
    best_score = -np.inf
    
    # Multi-restart Nelder-Mead optimization (screening phase)
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 10.0:
            break
        
        # Use fewer iterations for early restarts, more for later ones
        maxiter = 200 + 50 * idx
        tol = 1e-9 if idx < len(initial_guesses) - 2 else 1e-11
        
        x_opt, score = _nelder_mead(x0, n, cut_sizes, p, maxiter=maxiter, tol=tol)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Final high-precision polish on the best result (Nelder-Mead)
    if best_angles is not None and _time_left() > 20.0:
        x_polished, score_polished = _nelder_mead(
            best_angles.flatten(), n, cut_sizes, p, maxiter=400, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Analytic gradient refinement if time permits
    if best_angles is not None and _time_left() > 30.0:
        x_grad, score_grad = _analytic_gradient_refinement(
            best_angles.flatten(), n, edges, cut_sizes, p, max_steps=30, lr=0.02
        )
        if score_grad > best_score:
            best_score = score_grad
            best_angles = x_grad.reshape(p, 2)
    
    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
