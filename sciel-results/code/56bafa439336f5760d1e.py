"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _overlap_and_grad_h(h, alpha=200.0):
    """Return overlap vector and its Jacobian rows (sparse) for smooth-max chain rule.
    overlap_i = h_i*(1-h_{i+1}) for i<N-1; overlap_{N-1}=h_{N-1}.
    Returns (overlap, w) where w = smooth-max weights.
    """
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    w = _smooth_max_grad(overlap, alpha)
    return overlap, w

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        return np.full(N, 0.5)
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _optimize_from(h_start, alpha, maxiter, ftol, deadline):
    """Run SLSQP epigraph optimization from h_start. Returns (h, score) or None."""
    current = _score(h_start)
    def objective(params):
        h = params[:N]
        t = params[N]
        overlap, _ = _overlap_and_grad_h(h, alpha)
        smax = _smooth_max(overlap, alpha)
        return t + 1000.0 * np.maximum(0.0, smax - t)**2

    def constraint_sum(params):
        return np.sum(params[:N]) - N/2.0

    def constraint_epigraph(params):
        h = params[:N]
        t = params[N]
        overlap, _ = _overlap_and_grad_h(h, alpha)
        return t - np.max(overlap)

    x0 = np.concatenate([h_start, [current + 0.05]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_epigraph}
    ]
    try:
        res = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds_ext, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        if res.fun < 100.0:
            h_opt = _project_to_feasible(res.x[:N])
            sc = _score(h_opt)
            if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                return h_opt, sc
    except Exception:
        pass
    return None

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Start with 2-block solution
    h_final = _two_block_init()
    current_score = _score(h_final)

    # Phase 1: SLSQP from 2-block, moderate alpha
    res = _optimize_from(h_final, alpha=200.0, maxiter=2000, ftol=1e-14, deadline=deadline)
    if res and res[1] < current_score:
        h_final, current_score = res

    # Phase 2: High-precision polish, tighter alpha
    if time.time() < deadline - 20.0:
        res = _optimize_from(h_final, alpha=600.0, maxiter=3000, ftol=1e-16, deadline=deadline)
        if res and res[1] < current_score:
            h_final, current_score = res

    # Phase 3: Diverse structured starts
    if time.time() < deadline - 25.0:
        rng = np.random.default_rng(7)
        starts = []
        # Sinusoidal patterns
        for freq in [1, 2, 3, 4]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                x = np.arange(N)
                h_init = 0.5 + 0.45 * np.sin(2*np.pi*freq*x/N + phase)
                h_init = _project_to_feasible(h_init)
                starts.append(h_init)
        # Random smooth
        for _ in range(4):
            h_init = np.clip(0.5 + rng.normal(0, 0.2, N), 0.0, 1.0)
            h_init = _project_to_feasible(h_init)
            starts.append(h_init)
        # Alternating-ish
        for amp in [0.3, 0.45, 0.55]:
            h_init = np.full(N, 0.5)
            h_init[0::2] = min(1.0, 0.5 + amp)
            h_init[1::2] = max(0.0, 0.5 - amp)
            h_init = _project_to_feasible(h_init)
            starts.append(h_init)

        for h_init in starts:
            if time.time() > deadline - 15.0:
                break
            res = _optimize_from(h_init, alpha=300.0, maxiter=1500, ftol=1e-13, deadline=deadline)
            if res and res[1] < current_score:
                h_final, current_score = res

    # Phase 4: Local search refinement
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        h = h_final.copy()
        best_local = current_score
        for iteration in range(8000):
            if time.time() > deadline - 5.0:
                break
            step_size = max(0.0005, 0.04 * (1.0 - iteration / 8000.0))
            h_trial = h.copy()
            n_perturb = int(rng.integers(1, 5))
            idxs = rng.choice(N, size=n_perturb, replace=False)
            for i in idxs:
                h_trial[i] = np.clip(h_trial[i] + rng.uniform(-step_size, step_size), 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            if sc < best_local - 1e-13:
                best_local = sc
                h = h_trial.copy()
                h_final = h_trial.copy()
                current_score = sc

    # Final validation
    h_final = _project_to_feasible(h_final)
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)

    return h_final
