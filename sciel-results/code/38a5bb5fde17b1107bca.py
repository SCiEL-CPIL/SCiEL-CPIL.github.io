"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - h.sum()
    if abs(diff) > 1e-14:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _sawtooth_h(a, b, P, phase):
    """
    Generate step heights from a two-valued sawtooth pattern.
    Period P: first ceil(P/2) entries get value 'a', rest get value 'b'.
    phase shifts the starting position.
    """
    h = np.zeros(N)
    half = max(1, int(np.ceil(P / 2.0)))
    for i in range(N):
        idx = (i + phase) % N
        pos_in_period = idx % P
        if pos_in_period < half:
            h[i] = a
        else:
            h[i] = b
    return h


def _optimize_sawtooth():
    """
    Optimize the sawtooth parameters (a, b, P, phase) via:
    1. Coarse grid search
    2. Local Nelder-Mead refinement
    """
    best_score = float('inf')
    best_params = None
    best_h = None

    def evaluate(a, b, P, phase):
        P = max(2, min(N, int(round(P))))
        phase = int(phase) % N
        h = _sawtooth_h(a, b, P, phase)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        s = _score(h)
        return s, h

    # Phase 1: Coarse grid search
    a_vals = np.linspace(0.2, 0.8, 15)
    b_vals = np.linspace(0.2, 0.8, 15)
    P_vals = np.arange(2, N + 1, 2)  # even periods for symmetry
    phase_vals = np.arange(0, max(1, int(N / 4)))

    for a in a_vals:
        for b in b_vals:
            # Enforce sum constraint approximately: (a*half + b*(P-half))/P * N ≈ N/2
            # So a*half + b*(P-half) ≈ P/2
            for P in P_vals:
                half = max(1, int(np.ceil(P / 2.0)))
                # Check if this (a,b,P) can satisfy sum constraint
                target_per_period = P / 2.0
                actual_per_period = a * half + b * (P - half)
                if abs(actual_per_period - target_per_period) > 0.15:
                    continue
                for phase in phase_vals:
                    s, h = evaluate(a, b, P, phase)
                    if s < best_score:
                        best_score = s
                        best_params = (a, b, P, phase)
                        best_h = h.copy()

    # Phase 2: Local refinement with Nelder-Mead on continuous params
    def obj_saw(params):
        a, b, P, phase = params
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        P = max(2, min(N, P))
        phase = phase % N
        h = _sawtooth_h(a, b, P, int(phase))
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    if best_params is not None:
        x0 = np.array([best_params[0], best_params[1], float(best_params[2]), float(best_params[3])])
        try:
            res = minimize(obj_saw, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = (float(res.x[0]), float(res.x[1]), int(round(res.x[2])), int(round(res.x[3])) % N)
                h = _sawtooth_h(best_params[0], best_params[1], best_params[2], best_params[3])
                h = np.clip(h, 0.0, 1.0)
                h = _fix_sum(h)
                best_h = h.copy()
        except Exception:
            pass

    # Phase 3: Try multiple starting points
    for a_init in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for b_init in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for P_init in [4, 6, 8, 10, 14, 20, 30, 50]:
                for phase_init in [0, 1, 2, 3]:
                    x0 = np.array([a_init, b_init, float(P_init), float(phase_init)])
                    try:
                        res = minimize(obj_saw, x0, method='Nelder-Mead',
                                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res.fun < best_score:
                            best_score = res.fun
                            best_params = (float(res.x[0]), float(res.x[1]), int(round(res.x[2])), int(round(res.x[3])) % N)
                            h = _sawtooth_h(best_params[0], best_params[1], best_params[2], best_params[3])
                            h = np.clip(h, 0.0, 1.0)
                            h = _fix_sum(h)
                            best_h = h.copy()
                    except Exception:
                        pass

    if best_h is None:
        best_h = np.full(N, 0.5)
        best_score = _score(best_h)

    return best_score, best_h, best_params


def _optimize_sinewave():
    """
    Optimize a sinusoidal h: h_i = 0.5 + A*sin(2*pi*i/N + phi) + A2*sin(4*pi*i/N + phi2)
    """
    best_score = float('inf')
    best_h = None

    def obj_sin(params):
        A, phi, A2, phi2 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        i = np.arange(N)
        h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
        if abs(A2) > 1e-12:
            h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    # Grid search
    best_grid = float('inf')
    best_x = None
    for A in np.linspace(0.1, 0.45, 12):
        for phi in np.linspace(0, 2 * np.pi, 20):
            s = obj_sin([A, phi, 0.0, 0.0])
            if s < best_grid:
                best_grid = s
                best_x = [A, phi, 0.0, 0.0]

    if best_x is not None:
        try:
            res = minimize(obj_sin, best_x, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                A, phi, A2, phi2 = [float(v) for v in res.x]
                i = np.arange(N)
                h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
                if abs(A2) > 1e-12:
                    h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
                h = np.clip(h, 0.0, 1.0)
                h = _fix_sum(h)
                best_h = h.copy()
        except Exception:
            pass

    # Multiple starting points
    for A_init in [0.1, 0.2, 0.3, 0.4]:
        for phi_init in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x0 = [A_init, phi_init, 0.0, 0.0]
            try:
                res = minimize(obj_sin, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < best_score:
                    best_score = res.fun
                    A, phi, A2, phi2 = [float(v) for v in res.x]
                    i = np.arange(N)
                    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
                    if abs(A2) > 1e-12:
                        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
                    h = np.clip(h, 0.0, 1.0)
                    h = _fix_sum(h)
                    best_h = h.copy()
            except Exception:
                pass

    if best_h is None:
        best_h = np.full(N, 0.5)
        best_score = _score(best_h)

    return best_score, best_h


def _optimize_direct(h0, max_iter=3000):
    """
    Direct SLSQP optimization with smooth max (log-sum-exp) objective.
    """
    beta = 500.0

    def smooth_overlap(h):
        n = len(h)
        shifted = np.zeros(n)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        m = np.max(overlap)
        if m < 0:
            return m
        return m + np.log(np.sum(np.exp(beta * (overlap - m)))) / beta

    def obj(h):
        h = np.clip(h, 0.0, 1.0)
        return smooth_overlap(h)

    constraints = [{'type': 'eq', 'fun': lambda h: h.sum() - N / 2.0}]

    try:
        res = minimize(obj, h0, method='SLSQP',
                       bounds=[(0.0, 1.0)] * N,
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-15})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return _score(h_opt), h_opt
    except Exception:
        return float('inf'), h0


def _local_polish(h, max_iters=3000):
    """
    Local search: try small adjustments to individual entries.
    """
    h = h.copy()
    current_score = _score(h)
    improved = True
    iters = 0
    deltas = [0.005, -0.005, 0.01, -0.01, 0.002, -0.002, 0.001, -0.001]

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for i in range(N):
            for delta in deltas:
                new_val = h[i] + delta
                if new_val < 0 or new_val > 1:
                    continue
                h_trial = h.copy()
                h_trial[i] = new_val
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-13:
                    h = h_trial
                    current_score = s
                    improved = True
                    break

    return current_score, h


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    best_score = float('inf')
    best_h = None

    # Phase 1: Sawtooth optimization (new approach)
    if time.time() < deadline:
        s_saw, h_saw, _ = _optimize_sawtooth()
        if s_saw < best_score:
            best_score = s_saw
            best_h = h_saw.copy()

    # Phase 2: Sinusoidal optimization
    if time.time() < deadline - 20.0:
        s_sin, h_sin = _optimize_sinewave()
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # Phase 3: Direct SLSQP from best candidate
    if best_h is not None and time.time() < deadline - 25.0:
        s_direct, h_direct = _optimize_direct(best_h, max_iter=3000)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 4: Local polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_polish, h_polish = _local_polish(best_h, max_iters=3000)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 5: Final high-precision SLSQP
    if best_h is not None and time.time() < deadline - 5.0:
        s_final, h_final = _optimize_direct(best_h, max_iter=5000)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
