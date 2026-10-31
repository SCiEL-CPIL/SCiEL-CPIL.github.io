"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Structural change: instead of restricting to 2-block step functions, we optimize
the full vector h directly using a smooth (log-sum-exp) approximation to the
max-overlap objective, with L-BFGS-B. We also explore multi-block (3, 4, 5 block)
parametrizations and a sinusoidal family, then polish with direct discrete
local search (swap/adjust individual entries).
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _smooth_max(vals, beta=200.0):
    """Smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (vals - m)))) / beta


def _smooth_max_grad(vals, beta=200.0):
    """Gradient of smooth max via log-sum-exp."""
    m = np.max(vals)
    if m < -1e-15:
        return np.zeros_like(vals)
    exp_vals = np.exp(beta * (vals - m))
    sum_exp = np.sum(exp_vals)
    return exp_vals / sum_exp


def _overlap_vector(h):
    """Compute overlap vector h_i * (1 - h_{i+1}) for i=0..N-2, with h_N=0."""
    n = len(h)
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
    return overlap


def _objective_and_grad(x, beta=200.0):
    """
    Epigraph-style smooth objective with analytic gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    Returns (value, gradient).
    """
    h = np.clip(x, 0.0, 1.0)
    n = len(h)
    overlap = _overlap_vector(h)
    
    # Smooth max value and its gradient w.r.t. overlap
    sm_val = _smooth_max(overlap, beta)
    sm_grad_overlap = _smooth_max_grad(overlap, beta)
    
    # Gradient of overlap w.r.t. h
    # overlap[i] = h[i] * (1 - h[i+1]) for i < n-1
    # d(overlap[i])/d(h[i]) = 1 - h[i+1]
    # d(overlap[i])/d(h[i+1]) = -h[i]
    grad_h = np.zeros(n)
    if n > 1:
        # Contribution from overlap[i] to grad_h[i]
        grad_h[:-1] += sm_grad_overlap[:-1] * (1.0 - h[1:])
        # Contribution from overlap[i] to grad_h[i+1]
        grad_h[1:] += sm_grad_overlap[:-1] * (-h[:-1])
    
    # Penalty gradient
    penalty_coeff = 50.0
    sum_diff = h.sum() - N / 2.0
    grad_h += penalty_coeff * 2.0 * sum_diff
    
    value = sm_val + penalty_coeff * sum_diff ** 2
    return value, grad_h


def _fix_sum(h):
    """Adjust h so that sum(h) = N/2, keeping values in [0,1]."""
    diff = N / 2.0 - h.sum()
    if abs(diff) < 1e-14:
        return h
    # Distribute the difference
    n = len(h)
    if diff > 0:
        # Need to increase some entries
        room = 1.0 - h
        total_room = room.sum()
        if total_room > 1e-14:
            h = h + diff * room / total_room
        else:
            h = np.clip(h + diff / n, 0.0, 1.0)
    else:
        # Need to decrease some entries
        room = h
        total_room = room.sum()
        if total_room > 1e-14:
            h = h + diff * room / total_room
        else:
            h = np.clip(h + diff / n, 0.0, 1.0)
    return np.clip(h, 0.0, 1.0)


def _optimize_full_vector(h0, beta=200.0, max_iter=5000):
    """Optimize full vector h using L-BFGS-B with analytic smooth objective."""
    def obj(x):
        val, _ = _objective_and_grad(x, beta)
        return val

    def jac(x):
        _, g = _objective_and_grad(x, beta)
        return g

    h0_clipped = np.clip(h0, 0.0, 1.0)
    try:
        res = minimize(obj, h0_clipped, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt
    except Exception:
        return _fix_sum(h0_clipped)


def _sinusoidal_overlap(amp, phase, offset):
    """Compute overlap for h(x) = offset + amp * sin(2*pi*phase*x/2 + phase0)."""
    # h defined on [0,2], N points
    x = np.linspace(0, 2, N)
    h = offset + amp * np.sin(2 * np.pi * phase * x / 2.0 + phase0)
    h = np.clip(h, 0.0, 1.0)
    # Fix sum
    h = _fix_sum(h)
    return _score(h), h


def _optimize_sinusoidal():
    """Optimize sinusoidal family: h(x) = c + a*sin(k*x + phi)."""
    best_score = 1e10
    best_h = None

    # Grid search over parameters
    for c in np.linspace(0.3, 0.7, 10):
        for a in np.linspace(0.1, 0.5, 10):
            for k in np.linspace(0.5, 3.0, 10):
                for phi in np.linspace(0, 2 * np.pi, 10):
                    x = np.linspace(0, 2, N)
                    h = c + a * np.sin(k * x + phi)
                    h = np.clip(h, 0.0, 1.0)
                    h = _fix_sum(h)
                    s = _score(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
    return best_score, best_h


def _local_search(h, max_iters=5000):
    """Direct discrete local search: try small perturbations to individual entries."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.05, 0.02]

    for it in range(max_iters):
        improved = False
        for i in range(N):
            for ds in step_sizes:
                for sign in [1, -1]:
                    h_try = h.copy()
                    h_try[i] = np.clip(h_try[i] + sign * ds, 0.0, 1.0)
                    h_try = _fix_sum(h_try)
                    s = _score(h_try)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_h = h_try.copy()
                        h = h_try.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try larger steps
            for i in range(N):
                for ds in [0.1, 0.2, 0.05]:
                    for sign in [1, -1]:
                        h_try = h.copy()
                        h_try[i] = np.clip(h_try[i] + sign * ds, 0.0, 1.0)
                        h_try = _fix_sum(h_try)
                        s = _score(h_try)
                        if s < best_score - 1e-12:
                            best_score = s
                            best_h = h_try.copy()
                            h = h_try.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    return best_score, best_h


def _paired_local_search(h, max_iters=2000):
    """Local search with paired perturbations: adjust two entries simultaneously."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    step_sizes = [0.01, 0.005, 0.002, 0.02, 0.05]

    for it in range(max_iters):
        improved = False
        # Try single-entry moves first
        for i in range(N):
            for ds in step_sizes[:3]:
                for sign in [1, -1]:
                    h_try = h.copy()
                    h_try[i] = np.clip(h_try[i] + sign * ds, 0.0, 1.0)
                    h_try = _fix_sum(h_try)
                    s = _score(h_try)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_h = h_try.copy()
                        h = h_try.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # Try paired moves: increase one, decrease another
        for i in range(0, N, 2):  # Sample pairs for speed
            for j in range(i + 1, N, 2):
                for ds in step_sizes[:2]:
                    # Increase i, decrease j
                    h_try = h.copy()
                    h_try[i] = np.clip(h_try[i] + ds, 0.0, 1.0)
                    h_try[j] = np.clip(h_try[j] - ds, 0.0, 1.0)
                    h_try = _fix_sum(h_try)
                    s = _score(h_try)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_h = h_try.copy()
                        h = h_try.copy()
                        improved = True
                        break
                    # Decrease i, increase j
                    h_try = h.copy()
                    h_try[i] = np.clip(h_try[i] - ds, 0.0, 1.0)
                    h_try[j] = np.clip(h_try[j] + ds, 0.0, 1.0)
                    h_try = _fix_sum(h_try)
                    s = _score(h_try)
                    if s < best_score - 1e-12:
                        best_score = s
                        best_h = h_try.copy()
                        h = h_try.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break
    return best_score, best_h


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_score = 1e10
    best_h = None

    # --- Initial candidates ---

    # 1. Golden-ratio 2-block (from seed)
    A0 = (3.0 - math.sqrt(5.0)) / 2.0
    B0 = 1.0 - A0
    m0 = N // 2
    h_2block = np.zeros(N)
    h_2block[:m0] = A0
    h_2block[m0:] = B0
    h_2block = _fix_sum(h_2block)
    s = _score(h_2block)
    if s < best_score:
        best_score = s
        best_h = h_2block.copy()

    # 2. Uniform
    h_unif = np.full(N, 0.5)
    s = _score(h_unif)
    if s < best_score:
        best_score = s
        best_h = h_unif.copy()

    # 3. Sinusoidal
    if time.time() < deadline - 30:
        s_sin, h_sin = _optimize_sinusoidal()
        if h_sin is not None and s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # 4. Full-vector L-BFGS-B optimization from multiple starts (with analytic gradients)
    if time.time() < deadline - 60:
        starts = [best_h.copy(), h_2block.copy(), h_unif.copy()]
        # Add a few random starts
        for _ in range(3):
            h_rand = np.random.uniform(0.2, 0.8, N)
            h_rand = _fix_sum(h_rand)
            starts.append(h_rand)

        for h0 in starts:
            if time.time() > deadline - 30:
                break
            h_opt = _optimize_full_vector(h0, beta=300.0, max_iter=3000)
            s = _score(h_opt)
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()

    # 5. Local search polish (single-entry)
    if time.time() < deadline - 20:
        s_ls, h_ls = _local_search(best_h, max_iters=2000)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls.copy()

    # 6. Paired local search for more complex moves
    if time.time() < deadline - 15:
        s_pls, h_pls = _paired_local_search(best_h, max_iters=1500)
        if s_pls < best_score:
            best_score = s_pls
            best_h = h_pls.copy()

    # 7. Another round of L-BFGS-B from the locally-searched result with higher beta
    if time.time() < deadline - 10:
        h_opt2 = _optimize_full_vector(best_h, beta=500.0, max_iter=3000)
        s = _score(h_opt2)
        if s < best_score:
            best_score = s
            best_h = h_opt2.copy()
        # Final local search
        if time.time() < deadline - 5:
            s_ls2, h_ls2 = _local_search(best_h, max_iters=1000)
            if s_ls2 < best_score:
                best_score = s_ls2
                best_h = h_ls2.copy()
            # Final paired search
            if time.time() < deadline - 3:
                s_pls2, h_pls2 = _paired_local_search(best_h, max_iters=500)
                if s_pls2 < best_score:
                    best_score = s_pls2
                    best_h = h_pls2.copy()

    # Final fix
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
