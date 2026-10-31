"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.

Structural change: Use a global Differential Evolution (DE) search to escape
local minima, followed by multi-stage L-BFGS-B polishing with increasing
smoothness (beta) and direct local search. The DE phase explores the full
100-dimensional space, while the L-BFGS-B phase refines promising solutions
using analytic gradients. This combines the global exploration capability of
DE with the fast local convergence of gradient-based methods.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


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
    h = np.asarray(h, dtype=float)
    diff = TARGET_SUM - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        # If clipping broke the sum, do a second pass on non-saturated elements
        diff2 = TARGET_SUM - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                # Third pass if necessary
                diff3 = TARGET_SUM - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _optimize_full_vector(h0, beta=200.0, max_iter=5000, deadline=None):
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
                       bounds=[(0.0, 1.0)] * N,
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50, 'iprint': 0})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return h_opt
    except Exception:
        return _fix_sum(h0_clipped)


def _local_search(h, max_iters=5000, deadline=None):
    """Direct discrete local search: try small perturbations to individual entries."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.05, 0.02]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
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


def _paired_local_search(h, max_iters=2000, deadline=None):
    """Local search with paired perturbations: adjust two entries simultaneously."""
    h = np.clip(h, 0.0, 1.0).copy()
    h = _fix_sum(h)
    best_score = _score(h)
    best_h = h.copy()

    step_sizes = [0.01, 0.005, 0.002, 0.02, 0.05]

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
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


def _de_objective(x):
    """Objective function for Differential Evolution."""
    h = np.clip(x, 0.0, 1.0)
    h = _fix_sum(h)
    return _score(h)


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

    # 3. Linear ramp from 1 to 0 (inspired by observation)
    h_ramp = np.linspace(1.0, 0.0, N)
    h_ramp = _fix_sum(h_ramp)
    s = _score(h_ramp)
    if s < best_score:
        best_score = s
        best_h = h_ramp.copy()

    # 4. Differential Evolution global search
    if time.time() < deadline - 60:
        try:
            de_res = differential_evolution(
                _de_objective,
                bounds=[(0.0, 1.0)] * N,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            h_de = np.clip(de_res.x, 0.0, 1.0)
            h_de = _fix_sum(h_de)
            s_de = _score(h_de)
            if s_de < best_score:
                best_score = s_de
                best_h = h_de.copy()
        except Exception:
            pass

    # 5. Multi-stage L-BFGS-B optimization from multiple starts
    if time.time() < deadline - 40:
        starts = [best_h.copy(), h_2block.copy(), h_unif.copy(), h_ramp.copy()]
        # Add a few random starts
        for _ in range(3):
            h_rand = np.random.uniform(0.2, 0.8, N)
            h_rand = _fix_sum(h_rand)
            starts.append(h_rand)

        for h0 in starts:
            if time.time() > deadline - 30:
                break
            # Stage 1: lower beta for broader exploration
            h_opt1 = _optimize_full_vector(h0, beta=100.0, max_iter=2000, deadline=deadline)
            s1 = _score(h_opt1)
            if s1 < best_score:
                best_score = s1
                best_h = h_opt1.copy()
            
            # Stage 2: higher beta for sharper max approximation
            if time.time() < deadline - 20:
                h_opt2 = _optimize_full_vector(h_opt1, beta=300.0, max_iter=2000, deadline=deadline)
                s2 = _score(h_opt2)
                if s2 < best_score:
                    best_score = s2
                    best_h = h_opt2.copy()

    # 6. Local search polish (single-entry)
    if time.time() < deadline - 20:
        s_ls, h_ls = _local_search(best_h, max_iters=2000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls.copy()

    # 7. Paired local search for more complex moves
    if time.time() < deadline - 15:
        s_pls, h_pls = _paired_local_search(best_h, max_iters=1500, deadline=deadline)
        if s_pls < best_score:
            best_score = s_pls
            best_h = h_pls.copy()

    # 8. Final L-BFGS-B with highest beta
    if time.time() < deadline - 10:
        h_opt3 = _optimize_full_vector(best_h, beta=500.0, max_iter=3000, deadline=deadline)
        s3 = _score(h_opt3)
        if s3 < best_score:
            best_score = s3
            best_h = h_opt3.copy()
        # Final local search
        if time.time() < deadline - 5:
            s_ls2, h_ls2 = _local_search(best_h, max_iters=1000, deadline=deadline)
            if s_ls2 < best_score:
                best_score = s_ls2
                best_h = h_ls2.copy()
            # Final paired search
            if time.time() < deadline - 3:
                s_pls2, h_pls2 = _paired_local_search(best_h, max_iters=500, deadline=deadline)
                if s_pls2 < best_score:
                    best_score = s_pls2
                    best_h = h_pls2.copy()

    # Final fix
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
