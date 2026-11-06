"""QAOA p=2 graph-dependent angle optimizer: grid search + Nelder-Mead refinement.

Combines the fast exact-state-vector simulation and c_diag precomputation from the
direct-approach program with a structured grid search followed by high-precision
Nelder-Mead polish to escape local minima.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 15.0  # seconds to spare before 150 s wall-clock kill


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _c_diag(n, edges):
    """Precompute cut value per basis state."""
    num_states = 1 << n
    cd = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut = 0
        for (u, v) in edges:
            if bits[u] != bits[v]:
                cut += 1
        cd[state] = cut
    return cd


def _expected_cut(n, edges, gammas, betas, c_diag):
    """Fast expected cut via exact state simulation.

    Parameters
    ----------
    n : int
    edges : list of (u,v)
    gammas : array (p,)
    betas : array (p,)
    c_diag : array (num_states,) precomputed cut values per basis state
    """
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer: product over i of (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            mask = 1 << i
            idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
            idx1 = idx0 | mask
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _exhaustive_maxcut(n, edges, c_diag):
    return int(np.max(c_diag)) if c_diag.size else 0


def _initial_angles(n, edges, p):
    """Direct explicit initial guess from graph structure."""
    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    gamma1 = 0.25 + 0.55 * d
    beta1 = 0.30 - 0.10 * d
    gamma2 = 0.45 + 0.60 * d
    beta2 = 0.22 - 0.08 * d

    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    for k in range(2, p):
        if (k % 2 == 0):
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _grid_search(n, edges, p, c_diag, opt_cut, deadline):
    """Structured grid search over parameter space.
    
    Uses graph density to scale the search ranges, providing uniform coverage
    before refinement.
    """
    if opt_cut == 0:
        return _initial_angles(n, edges, p)

    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Scale search ranges based on density
    # For low density, parameters tend to be smaller
    # For high density, parameters can be larger
    g1_range = (0.1, 0.3 + 0.7 * d)
    b1_range = (0.05, 0.4 - 0.1 * d)
    g2_range = (0.15, 0.4 + 0.8 * d)
    b2_range = (0.03, 0.35 - 0.05 * d)

    # Grid resolution: balance coverage vs. time
    # Use ~8x8x8x8 = 4096 points, but adapt based on n
    if n <= 8:
        n_pts = 10
    elif n <= 10:
        n_pts = 8
    else:
        n_pts = 6

    g1_vals = np.linspace(g1_range[0], g1_range[1], n_pts)
    b1_vals = np.linspace(b1_range[0], b1_range[1], n_pts)
    g2_vals = np.linspace(g2_range[0], g2_range[1], n_pts)
    b2_vals = np.linspace(b2_range[0], b2_range[1], n_pts)

    best_angles = _initial_angles(n, edges, p)
    best_val = -1.0

    # Include the initial guess in the search
    candidates = []
    candidates.append(best_angles)

    for g1 in g1_vals:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        for b1 in b1_vals:
            if time.time() > deadline - _DEADLINE_MARGIN:
                break
            for g2 in g2_vals:
                if time.time() > deadline - _DEADLINE_MARGIN:
                    break
                for b2 in b2_vals:
                    if time.time() > deadline - _DEADLINE_MARGIN:
                        break
                    a = np.zeros((p, 2))
                    if p >= 1:
                        a[0, 0] = g1
                        a[0, 1] = b1
                    if p >= 2:
                        a[1, 0] = g2
                        a[1, 1] = b2
                    candidates.append(a)

    # Evaluate all candidates
    for a in candidates:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        try:
            v = _expected_cut(n, edges, a[:, 0], a[:, 1], c_diag)
            ratio = v / opt_cut
        except Exception:
            continue
        if ratio > best_val:
            best_val = ratio
            best_angles = a.copy()

    return best_angles, best_val


def _nelder_mead_refine(n, edges, p, angles0, c_diag, opt_cut, deadline):
    """High-precision Nelder-Mead refinement of the best grid point."""
    if opt_cut == 0:
        return angles0

    def objective(params):
        """Minimize negative expected cut ratio."""
        gammas = params[:p]
        betas = params[p:]
        try:
            v = _expected_cut(n, edges, gammas, betas, c_diag)
            return -v / opt_cut
        except Exception:
            return 1.0  # Large penalty for invalid

    # Flatten angles to 1D parameter vector
    params0 = np.concatenate([angles0[:, 0], angles0[:, 1]])

    # Bounds: keep parameters in reasonable range
    bounds = [(-np.pi, np.pi)] * (2 * p)

    try:
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': 500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if result.success:
            best_params = result.x
        else:
            # If Nelder-Mead fails, try BFGS as fallback
            try:
                result2 = minimize(
                    objective,
                    params0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                best_params = result2.x
            except Exception:
                best_params = params0
    except Exception:
        best_params = params0

    # Reconstruct angles
    refined = np.zeros((p, 2))
    refined[:, 0] = best_params[:p]
    refined[:, 1] = best_params[p:]
    return np.clip(refined, -np.pi, np.pi)


def _local_refine(n, edges, p, angles0, c_diag, opt_cut, deadline):
    """Fast coordinate-descent refinement as a fallback or additional polish."""
    if opt_cut == 0:
        return angles0

    best = angles0.copy()
    best_val = -1.0
    try:
        v = _expected_cut(n, edges, best[:, 0], best[:, 1], c_diag)
        best_val = v / opt_cut
    except Exception:
        return best

    base_steps = np.array([0.15, 0.10, 0.20, 0.08])
    n_params = p * 2
    shrink = 1.0
    max_rounds = 100  # Reduced since Nelder-Mead does the heavy lifting
    for rnd in range(max_rounds):
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        improved = False
        for i in range(n_params):
            if time.time() > deadline - _DEADLINE_MARGIN:
                break
            layer = i // 2
            role = i % 2
            step = base_steps[i % 4] * shrink
            for sign in (1.0, -1.0):
                cand = best.copy()
                cand[layer, role] += sign * step
                if cand[layer, role] < -np.pi or cand[layer, role] > np.pi:
                    continue
                try:
                    v = _expected_cut(n, edges, cand[:, 0], cand[:, 1], c_diag)
                    ratio = v / opt_cut
                except Exception:
                    continue
                if ratio > best_val:
                    best = cand
                    best_val = ratio
                    improved = True
        if not improved:
            shrink *= 0.7
            if shrink < 0.05:
                break
    return best


def _multi_start_refine(n, edges, p, c_diag, opt_cut, deadline):
    """Combined grid search + Nelder-Mead + local refinement pipeline."""
    if opt_cut == 0:
        return _initial_angles(n, edges, p)

    # Phase 1: Grid search to find a good starting point
    grid_result, grid_val = _grid_search(n, edges, p, c_diag, opt_cut, deadline)
    
    if time.time() > deadline - _DEADLINE_MARGIN:
        return grid_result

    # Phase 2: Nelder-Mead refinement from the best grid point
    nm_result = _nelder_mead_refine(n, edges, p, grid_result, c_diag, opt_cut, deadline)
    
    if time.time() > deadline - _DEADLINE_MARGIN:
        return nm_result

    # Phase 3: Final local refinement polish
    final_result = _local_refine(n, edges, p, nm_result, c_diag, opt_cut, deadline)
    
    return final_result


def angles(n, edges, p):
    """Entry point. Returns (p, 2) array of (gamma, beta) in radians."""
    t_start = time.time()
    deadline = t_start + 150.0 - _DEADLINE_MARGIN

    # Hard safety: if we somehow can't do anything, return the direct guess
    try:
        init = _initial_angles(n, edges, p)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

    if n == 0 or len(edges) == 0:
        return init

    # Precompute
    try:
        c_diag = _c_diag(n, edges)
        opt_cut = _exhaustive_maxcut(n, edges, c_diag)
    except Exception:
        return init

    if opt_cut == 0:
        return init

    # If n is large (state space too big), just return the direct guess.
    # For n up to 12 (4096 states) simulation is fine; for larger, skip refine.
    if n > 12:
        return init

    try:
        refined = _multi_start_refine(n, edges, p, c_diag, opt_cut, deadline)
        return np.clip(refined, -np.pi, np.pi)
    except Exception:
        return np.clip(init, -np.pi, np.pi)
