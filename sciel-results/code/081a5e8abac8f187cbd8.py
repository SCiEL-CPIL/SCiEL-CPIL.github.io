"""Direct-approach QAOA parameter design.

Builds explicit (gamma, beta) parameters from graph structure (density,
degree statistics) and refines them with a fast local search.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE_MARGIN = 12.0  # seconds to spare before 150 s wall-clock kill


def _density(n, edges):
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible


def _initial_angles(n, edges, p):
    """Direct explicit initial guess from graph structure."""
    d = _density(n, edges)
    # Clamp density to a useful range
    d = min(max(d, 0.05), 1.0)

    # Base parameters tuned from known QAOA p=2 behavior.
    # For a graph of density d:
    #   gamma1: moderate, scales with d
    #   beta1:  small
    #   gamma2: larger, scales with d
    #   beta2:  small
    gamma1 = 0.25 + 0.55 * d
    beta1 = 0.30 - 0.10 * d
    gamma2 = 0.45 + 0.60 * d
    beta2 = 0.22 - 0.08 * d

    # Build (p,2) array; for p>2 we would need more structure, but task has p=2.
    # If p is not 2, fall back to repeating the pattern.
    result = np.zeros((p, 2))
    if p >= 1:
        result[0, 0] = gamma1
        result[0, 1] = beta1
    if p >= 2:
        result[1, 0] = gamma2
        result[1, 1] = beta2
    # For p > 2 (unlikely), alternate
    for k in range(2, p):
        if (k % 2 == 0):
            result[k, 0] = gamma2
            result[k, 1] = beta2
        else:
            result[k, 0] = gamma1
            result[k, 1] = beta1
    return result


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
            # Vectorize the pairwise update
            # For all states with bit i = 0, update with partner
            # Build index arrays
            idx0 = np.array([s0 for s0 in range(num_states) if (s0 & mask) == 0], dtype=np.int64)
            idx1 = idx0 | mask
            v0 = psi[idx0].copy()
            v1 = psi[idx1].copy()
            psi[idx0] = c * v0 - 1j * s * v1
            psi[idx1] = -1j * s * v0 + c * v1

    probs = np.abs(psi) ** 2
    return float(np.sum(probs * c_diag))


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


def _exhaustive_maxcut(n, edges, c_diag):
    return int(np.max(c_diag)) if c_diag.size else 0


def _nelder_mead_refine(n, edges, p, angles0, c_diag, opt_cut, deadline):
    """Nelder-Mead refinement of the (gamma, beta) parameters using scipy."""
    if opt_cut == 0:
        return angles0

    x0 = angles0.flatten().copy()
    
    def objective(x):
        if time.time() > deadline - _DEADLINE_MARGIN:
            return -1.0
        try:
            a = x.reshape(p, 2)
            v = _expected_cut(n, edges, a[:, 0], a[:, 1], c_diag)
            return -v / opt_cut
        except Exception:
            return -1.0

    # Initial evaluation
    try:
        best_val = -objective(x0)
    except Exception:
        return angles0

    # Nelder-Mead with adaptive settings
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True,
            }
        )
        x_opt = res.x
        val_opt = -res.fun
        if val_opt > best_val:
            best_val = val_opt
            x0 = x_opt
    except Exception:
        pass

    # Final high-precision polish: run Nelder-Mead again with tighter tolerances
    if time.time() < deadline - _DEADLINE_MARGIN:
        try:
            res2 = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            x_opt2 = res2.x
            val_opt2 = -res2.fun
            if val_opt2 > best_val:
                best_val = val_opt2
                x0 = x_opt2
        except Exception:
            pass

    return x0.reshape(p, 2)


def _multi_start_refine(n, edges, p, c_diag, opt_cut, deadline, k_starts=6):
    """Try several structured starts, keep the best after Nelder-Mead refinement."""
    if opt_cut == 0:
        return _initial_angles(n, edges, p)

    d = _density(n, edges)
    d = min(max(d, 0.05), 1.0)

    # Structured starts: vary the density-scaling coefficients
    starts = []
    # Base (direct)
    starts.append(_initial_angles(n, edges, p))
    # Variants with different gamma2 emphasis
    for g2_scale in (0.7, 0.85, 1.0, 1.2, 1.4, 1.6):
        g1 = 0.25 + 0.55 * d
        b1 = 0.30 - 0.10 * d
        g2 = (0.45 + 0.60 * d) * g2_scale
        b2 = 0.22 - 0.08 * d
        a = np.zeros((p, 2))
        if p >= 1:
            a[0, 0], a[0, 1] = g1, b1
        if p >= 2:
            a[1, 0], a[1, 1] = min(g2, np.pi), b2
        starts.append(a)
    # Additional diverse starts
    for g1_scale, b1_scale, g2_scale, b2_scale in [(0.8, 1.2, 1.1, 0.9), (1.2, 0.8, 0.9, 1.1)]:
        g1 = (0.25 + 0.55 * d) * g1_scale
        b1 = (0.30 - 0.10 * d) * b1_scale
        g2 = (0.45 + 0.60 * d) * g2_scale
        b2 = (0.22 - 0.08 * d) * b2_scale
        a = np.zeros((p, 2))
        if p >= 1:
            a[0, 0], a[0, 1] = min(g1, np.pi), min(b1, np.pi)
        if p >= 2:
            a[1, 0], a[1, 1] = min(g2, np.pi), min(b2, np.pi)
        starts.append(a)

    best = starts[0]
    best_val = -1.0
    for a0 in starts:
        if time.time() > deadline - _DEADLINE_MARGIN:
            break
        a0 = np.clip(a0, -np.pi, np.pi)
        a_ref = _nelder_mead_refine(n, edges, p, a0, c_diag, opt_cut, deadline)
        try:
            v = _expected_cut(n, edges, a_ref[:, 0], a_ref[:, 1], c_diag)
            ratio = v / opt_cut
        except Exception:
            ratio = -1.0
        if ratio > best_val:
            best_val = ratio
            best = a_ref
    return best


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
    # For n up to 10 (1024 states) simulation is fine; for larger, skip refine.
    if n > 12:
        return init

    try:
        refined = _multi_start_refine(n, edges, p, c_diag, opt_cut, deadline)
        result = np.clip(refined, -np.pi, np.pi)
        return result
    except Exception:
        return np.clip(init, -np.pi, np.pi)
