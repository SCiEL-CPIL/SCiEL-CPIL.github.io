"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target, preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    for _ in range(5):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, beta):
    """Smooth max using log-sum-exp, returns (value, gradient)."""
    x = np.asarray(x, dtype=float)
    x_max = np.max(x)
    shifted = x - x_max
    exp_vals = np.exp(beta * shifted)
    wsum = np.sum(exp_vals)
    if wsum < 1e-300:
        return x_max, np.zeros_like(x)
    value = x_max + math.log(wsum) / beta
    grad = exp_vals / wsum
    return value, grad

def _fourier_to_h(coeffs, N):
    """Convert Fourier coefficients to h values.
    coeffs = [a1, b1, a2, b2, ..., aJ, bJ]
    h_i = 0.5 + sum_{j=1}^{J} (a_j*cos(2*pi*j*i/N) + b_j*sin(2*pi*j*i/N))
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    J = len(coeffs) // 2
    for j in range(1, J + 1):
        a = coeffs[2*(j-1)]
        b = coeffs[2*(j-1)+1]
        h += a * np.cos(2.0 * np.pi * j * i / N) + b * np.sin(2.0 * np.pi * j * i / N)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Start with cosine wave (inspiration 1: score ~1.085)
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    current_score = _score(h_cos)
    h_final = h_cos.copy()

    # Also try the 2-block solution as fallback
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h_block = np.zeros(N)
    half = N // 2
    h_block[:half] = A
    h_block[half:] = B
    h_block = _fix_sum(h_block)
    block_score = _score(h_block)
    if block_score < current_score:
        h_final = h_block.copy()
        current_score = block_score

    # Phase 1: Fourier parameterization optimization
    # Use J harmonics. Start with J=5, then increase if time permits.
    try:
        if time.time() < deadline - 30.0:
            J = 5
            n_coeffs = 2 * J
            
            # Initial coefficients: match cosine wave
            # h_cos = 0.5 + 0.5*cos(2*pi*i/N) => a1=0.5, b1=0, rest=0
            coeffs0 = np.zeros(n_coeffs)
            coeffs0[0] = 0.5  # a1
            # b1 = 0 already
            
            def obj_fourier(coeffs):
                h = _fourier_to_h(coeffs, N)
                # Check bounds with penalty
                viol = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h - 1)**2)
                if viol > 1e-8:
                    return 1e6 * viol + _score(h)
                return _score(h)
            
            # Optimize with Nelder-Mead (no gradient needed for this small space)
            result_f = minimize(
                obj_fourier,
                coeffs0.copy(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result_f.fun < current_score - 1e-10:
                h_f = _fourier_to_h(result_f.x, N)
                h_f = _fix_sum(h_f)
                if np.all(h_f >= -1e-9) and np.all(h_f <= 1 + 1e-9):
                    h_f = np.clip(h_f, 0.0, 1.0)
                    h_f = _fix_sum(h_f)
                    new_score = _score(h_f)
                    if new_score < current_score:
                        h_final = h_f.copy()
                        current_score = new_score

    except Exception:
        pass

    # Phase 2: Direct SLSQP on h with smooth max + epigraph
    try:
        if time.time() < deadline - 20.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax, _ = _smooth_max(overlap, beta=200.0)
                return t + 1000.0 * np.maximum(0.0, smax - t)**2

            x0 = np.concatenate([h_final, [current_score + 0.05]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - TARGET_SUM

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result2 = minimize(
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 1500, 'ftol': 1e-14, 'disp': False}
            )

            if result2.fun < 100.0:
                h_opt = result2.x[:N].copy()
                h_opt = _fix_sum(h_opt)
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt.copy()
                    current_score = new_score

    except Exception:
        pass

    # Phase 3: Higher-precision polish with tighter smooth max
    try:
        if time.time() < deadline - 10.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax, _ = _smooth_max(overlap, beta=500.0)
                return t + 10000.0 * np.maximum(0.0, smax - t)**2

            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
            
            def constraint_sum(params):
                return np.sum(params[:N]) - TARGET_SUM

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return t - np.max(overlap)

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )

            if result3.fun < 100.0:
                h_opt = result3.x[:N].copy()
                h_opt = _fix_sum(h_opt)
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt.copy()
                    current_score = new_score

    except Exception:
        pass

    # Phase 4: Try higher Fourier harmonics if time permits
    try:
        if time.time() < deadline - 15.0:
            J = 8
            n_coeffs = 2 * J
            coeffs0 = np.zeros(n_coeffs)
            coeffs0[0] = 0.5
            
            def obj_fourier2(coeffs):
                h = _fourier_to_h(coeffs, N)
                viol = np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h - 1)**2)
                if viol > 1e-8:
                    return 1e6 * viol + _score(h)
                return _score(h)
            
            result_f2 = minimize(
                obj_fourier2,
                coeffs0.copy(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result_f2.fun < current_score - 1e-10:
                h_f = _fourier_to_h(result_f2.x, N)
                h_f = _fix_sum(h_f)
                if np.all(h_f >= -1e-9) and np.all(h_f <= 1 + 1e-9):
                    h_f = np.clip(h_f, 0.0, 1.0)
                    h_f = _fix_sum(h_f)
                    new_score = _score(h_f)
                    if new_score < current_score:
                        h_final = h_f.copy()
                        current_score = new_score
                        # Polish the Fourier result with SLSQP
                        if time.time() < deadline - 5.0:
                            def objective_smooth2(params):
                                h = params[:N]
                                t = params[N]
                                shifted = np.zeros(N)
                                shifted[:-1] = h[1:]
                                overlap = h * (1.0 - shifted)
                                smax, _ = _smooth_max(overlap, beta=300.0)
                                return t + 5000.0 * np.maximum(0.0, smax - t)**2

                            x0_p = np.concatenate([h_final, [current_score + 0.02]])
                            bounds_p = [(0.0, 1.0)] * N + [(0.0, 1.0)]
                            
                            def cs(params):
                                return np.sum(params[:N]) - TARGET_SUM

                            def ce(params):
                                h = params[:N]
                                t = params[N]
                                shifted = np.zeros(N)
                                shifted[:-1] = h[1:]
                                overlap = h * (1.0 - shifted)
                                return t - np.max(overlap)

                            cons_p = [
                                {'type': 'eq', 'fun': cs},
                                {'type': 'ineq', 'fun': ce}
                            ]

                            res_p = minimize(
                                objective_smooth2,
                                x0_p,
                                method='SLSQP',
                                bounds=bounds_p,
                                constraints=cons_p,
                                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
                            )

                            if res_p.fun < 100.0:
                                h_p = res_p.x[:N].copy()
                                h_p = _fix_sum(h_p)
                                new_sp = _score(h_p)
                                if new_sp < current_score and np.all(h_p >= 0.0) and np.all(h_p <= 1.0):
                                    h_final = h_p.copy()
                                    current_score = new_sp

    except Exception:
        pass

    # Final validation
    h_final = _fix_sum(h_final)
    if not (np.all(h_final >= 0.0) and np.all(h_final <= 1.0)):
        # Fallback to cosine wave
        h_final = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        h_final = _fix_sum(h_final)

    return h_final
