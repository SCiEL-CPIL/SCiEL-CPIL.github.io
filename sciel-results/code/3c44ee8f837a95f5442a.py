"""Graph-dependent two-layer QAOA angle design.

Combines fast exact state-vector simulation with density-aware initialization,
structured grid search for global coverage, Nelder-Mead refinement, and
coordinate-descent polish to maximize the expected cut ratio.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 142.0  # safety margin below the 150 s wall-clock kill


def _build_cost_diag(n, edges):
    """Precompute cut value per basis state (diagonal of C in Z basis)."""
    num_states = 1 << n
    cd = np.zeros(num_states, dtype=float)
    for state in range(num_states):
        bits = [(state >> i) & 1 for i in range(n)]
        cut = 0
        for (u, v) in edges:
            if 0 <= u < n and 0 <= v < n and u != v:
                if bits[u] != bits[v]:
                    cut += 1
        cd[state] = cut
    return cd


def _exact_maxcut(c_diag):
    """Exhaustive MaxCut optimum from precomputed cut values."""
    return float(np.max(c_diag)) if c_diag.size else 0.0


def _expected_cut(n, edges, gammas, betas, c_diag):
    """Exact expected cut via efficient state-vector simulation.
    
    Uses in-place rotations for the mixer instead of constructing full matrices.
    """
    num_states = 1 << n
    psi = np.full(num_states, 1.0 / np.sqrt(num_states), dtype=complex)

    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]

        # Cost layer: diagonal phase
        psi = psi * np.exp(-1j * gamma * c_diag)

        # Mixer layer: product over i of (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            mask = 1 << i
            # Split indices by bit i
            idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
            idx1 = idx0 | mask
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


def _initial_angles(n, edges, p):
    """Density-aware initial guess from graph structure."""
    if n < 2:
        return np.tile([0.35, 0.25], (p, 1))
    possible = n * (n - 1) / 2.0
    d = len(edges) / possible if possible > 0 else 0.0
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
        if k % 2 == 0:
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


def _grid_search(n, edges, p, c_diag, opt_cut, deadline):
    """Structured grid search over parameter space with density-scaled ranges."""
    if opt_cut <= 0.0:
        return _initial_angles(n, edges, p), 0.0

    d = len(edges) / (n * (n - 1) / 2.0) if n >= 2 else 0.0
    d = min(max(d, 0.05), 1.0)

    # Scale search ranges based on density
    g1_range = (0.1, 0.3 + 0.7 * d)
    b1_range = (0.05, 0.4 - 0.1 * d)
    g2_range = (0.15, 0.4 + 0.8 * d)
    b2_range = (0.03, 0.35 - 0.05 * d)

    # Grid resolution: balance coverage vs. time
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
    best_val = _expected_cut(n, edges, best_angles[:, 0], best_angles[:, 1], c_diag) / opt_cut

    # Evaluate grid points
    for g1 in g1_vals:
        if time.time() > deadline - 15.0:
            break
        for b1 in b1_vals:
            if time.time() > deadline - 15.0:
                break
            for g2 in g2_vals:
                if time.time() > deadline - 15.0:
                    break
                for b2 in b2_vals:
                    if time.time() > deadline - 15.0:
                        break
                    a = np.zeros((p, 2))
                    if p >= 1:
                        a[0, 0] = g1
                        a[0, 1] = b1
                    if p >= 2:
                        a[1, 0] = g2
                        a[1, 1] = b2
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
    if opt_cut <= 0.0:
        return angles0

    def objective(params):
        """Minimize negative expected cut ratio."""
        gammas = params[:p]
        betas = params[p:]
        try:
            v = _expected_cut(n, edges, gammas, betas, c_diag)
            return -v / opt_cut
        except Exception:
            return 1.0

    params0 = np.concatenate([angles0[:, 0], angles0[:, 1]])
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
        best_params = result.x
    except Exception:
        # Fallback to L-BFGS-B
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

    refined = np.zeros((p, 2))
    refined[:, 0] = best_params[:p]
    refined[:, 1] = best_params[p:]
    return np.clip(refined, -np.pi, np.pi)


def _coordinate_descent_polish(n, edges, p, angles0, c_diag, opt_cut, deadline):
    """Coordinate-descent refinement for final polish."""
    if opt_cut <= 0.0:
        return angles0

    best = angles0.copy()
    try:
        best_val = _expected_cut(n, edges, best[:, 0], best[:, 1], c_diag) / opt_cut
    except Exception:
        return best

    base_steps = np.array([0.15, 0.10, 0.20, 0.08])
    n_params = p * 2
    shrink = 1.0
    max_rounds = 50

    for rnd in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(n_params):
            if time.time() > deadline - 10.0:
                break
            step = base_steps[i % 4] * shrink
            for sign in (1.0, -1.0):
                trial = best.copy()
                layer = i // 2
                role = i % 2
                trial[layer, role] += sign * step
                try:
                    v = _expected_cut(n, edges, trial[:, 0], trial[:, 1], c_diag)
                    ratio = v / opt_cut
                except Exception:
                    continue
                if ratio > best_val:
                    best_val = ratio
                    best = trial.copy()
                    improved = True
        if not improved:
            shrink *= 0.5
            if shrink < 0.01:
                break

    return best


def _optimize_for_graph(n, edges, p, deadline):
    """Optimize QAOA angles for a single graph."""
    c_diag = _build_cost_diag(n, edges)
    opt = _exact_maxcut(c_diag)

    if opt <= 0.0:
        return np.tile([0.35, 0.25], (p, 1))

    # Phase 1: Grid search for global coverage
    best_angles, best_val = _grid_search(n, edges, p, c_diag, opt, deadline)

    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 15.0:
        best_angles = _nelder_mead_refine(n, edges, p, best_angles, c_diag, opt, deadline)

    # Phase 3: Coordinate-descent polish
    if time.time() < deadline - 10.0:
        best_angles = _coordinate_descent_polish(n, edges, p, best_angles, c_diag, opt, deadline)

    # Final high-precision L-BFGS-B polish if time permits
    if time.time() < deadline - 5.0:
        def objective(params):
            gammas = params[:p]
            betas = params[p:]
            try:
                v = _expected_cut(n, edges, gammas, betas, c_diag)
                return -v
            except Exception:
                return 0.0

        params0 = np.concatenate([best_angles[:, 0], best_angles[:, 1]])
        bounds = [(0.0, np.pi)] * (2 * p)
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            refined = np.zeros((p, 2))
            refined[:, 0] = result.x[:p]
            refined[:, 1] = result.x[p:]
            # Verify improvement
            v_new = _expected_cut(n, edges, refined[:, 0], refined[:, 1], c_diag)
            v_old = _expected_cut(n, edges, best_angles[:, 0], best_angles[:, 1], c_diag)
            if v_new >= v_old:
                best_angles = refined
        except Exception:
            pass

    return best_angles


def angles(n, edges, p):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    try:
        return _optimize_for_graph(n, edges, p, deadline)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
