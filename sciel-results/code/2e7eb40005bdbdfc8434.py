"""
Optimized QAOA Parameter Design for MaxCut.

Strategy:
1. Fast vectorized simulation of the QAOA circuit.
2. Two-stage optimization:
   a. Screening: Quick Nelder-Mead from diverse initial points to identify promising basins.
   b. Polish: High-precision Nelder-Mead on the best candidate(s) found in screening.
3. Strict deadline guard to ensure valid return before 150s.
"""
# EVOLVE-BLOCK-START
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
    
    # Vectorized cut size calculation
    # For each edge (i, j), we check if bit i and bit j differ.
    # (states >> i) & 1 gives the bit value.
    # XOR gives 1 if they differ, 0 if same.
    for (i, j) in edges:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi ^ bj).astype(np.float64)
        
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state vector.
    This is a product of single-qubit rotations exp(-i beta X_i).
    exp(-i beta X) = cos(beta) I - i sin(beta) X.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    
    # Reshape to tensor [2, 2, ..., 2]
    tensor = state.reshape([2] * n)
    
    for i in range(n):
        # Move axis i to front
        moved = np.moveaxis(tensor, i, 0)
        # Flatten remaining axes
        flat = moved.reshape(2, -1)
        
        # Apply rotation matrix [[c, -is], [-is, c]]
        # new[0] = c*old[0] - is*old[1]
        # new[1] = -is*old[0] + c*old[1]
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        
        # Reshape and move axis back
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
        
    return tensor.flatten()


def _expected_cut(n, cut_sizes, angles):
    """
    Compute expected cut value for given angles.
    angles: array of shape (p, 2) where row is [gamma, beta].
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for k in range(angles.shape[0]):
        gamma = angles[k, 0]
        beta = angles[k, 1]
        
        # Cost unitary: diagonal in computational basis
        # exp(-i gamma C)
        # C eigenvalues are cut_sizes
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Mixer unitary: exp(-i beta sum X)
        state = _apply_x_layer(state, beta, n)
        
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    try:
        return -_expected_cut(n, cut_sizes, angles)
    except Exception:
        return 1e10


def _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter, tol, timeout_check=True):
    """
    Run Nelder-Mead optimization with time guard.
    Returns (optimized_params, score).
    """
    def _obj_wrapper(params):
        if timeout_check and _time_left() < 2.0:
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
        # Ensure we get the best point found
        best_x = res.x
        best_score = -res.fun
        
        # Sometimes Nelder-Mead returns a point that is worse than initial due to numerical issues
        # We can check initial score
        initial_score = -_objective(x0, n, cut_sizes, p)
        if initial_score > best_score:
            best_x = x0
            best_score = initial_score
            
        return best_x, best_score
    except TimeoutError:
        # Return current best if timed out? Nelder-Mead doesn't expose intermediate best easily in this wrapper.
        # We will just return the initial guess if timeout happens during screening, 
        # but for polish, we should have checked time beforehand.
        return x0, -_objective(x0, n, cut_sizes, p)
    except Exception:
        return x0, -_objective(x0, n, cut_sizes, p)


def _generate_initial_guesses(n, edges, p):
    """
    Generate diverse initial angle guesses based on graph structure and literature.
    """
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0.0
    
    # Degree stats
    degree = np.zeros(n, dtype=np.int32)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1
    max_deg = int(np.max(degree)) if n > 0 else 0
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    guesses = []
    
    # 1. Density-based heuristic
    # Sparse graphs often prefer smaller gammas
    g1 = 0.2 + 0.4 * density
    b1 = 0.2 + 0.2 * (1 - density)
    guesses.append(np.tile([g1, b1], p))
    
    # 2. Literature-inspired (Bravyi et al., Farhi et al.)
    # For p=2, typical good ranges are gamma in [0.2, 0.6], beta in [0.1, 0.4]
    guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    guesses.append(np.array([0.40, 0.20, 0.40, 0.20]))
    guesses.append(np.array([0.25, 0.30, 0.25, 0.30]))
    guesses.append(np.array([0.50, 0.35, 0.50, 0.35]))
    guesses.append(np.array([0.30, 0.30, 0.30, 0.30]))
    
    # 3. Small angles (near identity)
    guesses.append(np.tile([0.1, 0.1], p))
    
    # 4. Randomized diverse starts (fixed seed for reproducibility)
    rng = np.random.RandomState(1234)
    for _ in range(8):
        gammas = rng.uniform(0.05, 0.8, p)
        betas = rng.uniform(0.05, 0.5, p)
        # Ensure some correlation or structure? No, random is good for diversity.
        guesses.append(np.column_stack((gammas, betas)).flatten())
        
    return guesses


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns an array of shape (p, 2).
    """
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Generate initial guesses
    initial_guesses = _generate_initial_guesses(n, edges, p)
    
    best_angles = None
    best_score = -np.inf
    
    # --- Stage 1: Screening ---
    # Run a quick optimization from each initial guess to find the best basin.
    # We limit iterations to keep this stage fast.
    screening_maxiter = 100
    screening_tol = 1e-6
    
    candidates = []
    
    for i, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:  # Keep 15s for polish
            break
            
        # Slightly increase iterations for later guesses if time allows? 
        # No, keep it uniform for screening.
        x_opt, score = _nelder_mead_optimize(
            x0, n, cut_sizes, p, 
            maxiter=screening_maxiter, 
            tol=screening_tol
        )
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
            
        candidates.append((score, x_opt))
        
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # --- Stage 2: Polish ---
    # Take the top 1-2 candidates and refine them with high precision.
    # If we have time, we can polish the top 2.
    
    polish_candidates = []
    for score, x_opt in candidates[:2]:
        polish_candidates.append(x_opt)
        
    # Refine the best candidate
    if _time_left() > 10.0 and len(polish_candidates) > 0:
        x_best = polish_candidates[0]
        # High precision polish
        x_polished, score_polished = _nelder_mead_optimize(
            x_best, n, cut_sizes, p,
            maxiter=500,
            tol=1e-10
        )
        
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
            
    # If time allows, refine the second best candidate
    if _time_left() > 10.0 and len(polish_candidates) > 1:
        x_second = polish_candidates[1]
        x_polished2, score_polished2 = _nelder_mead_optimize(
            x_second, n, cut_sizes, p,
            maxiter=300,
            tol=1e-10
        )
        
        if score_polished2 > best_score:
            best_score = score_polished2
            best_angles = x_polished2.reshape(p, 2)
            
    # Final check: if we have a lot of time left, do another polish on the absolute best
    if best_angles is not None and _time_left() > 20.0:
        x_final = best_angles.flatten()
        x_polished_final, score_final = _nelder_mead_optimize(
            x_final, n, cut_sizes, p,
            maxiter=200,
            tol=1e-12
        )
        if score_final > best_score:
            best_score = score_final
            best_angles = x_polished_final.reshape(p, 2)
            
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
