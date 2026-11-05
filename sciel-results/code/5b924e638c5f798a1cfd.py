"""Rewritten program: parameterized chirp+harmonics + differential evolution + Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 400
_DEADLINE = 150.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - 15.0 - (time.time() - _T0)

def _exact_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chirp_harmonics(params):
    """params: [phase0, chirp_rate, amp1, ph1, freq1, amp2, ph2, freq2, amp3, ph3, freq3, amp4, ph4, freq4, dc_offset]
    Returns h of length N."""
    t = np.arange(N, dtype=float)
    # Chirp: h_c(t) = cos(phase0 + 2*pi*(f0*t/N + chirp_rate*(t/N)^2))
    # We fold f0 into phase0 effectively; use chirp_rate as quadratic term coefficient
    phase = params[0] + 2.0 * np.pi * (params[1] * (t / N) ** 2)
    h = np.cos(phase)
    # Add harmonics: amp_j * cos(ph_j + 2*pi*freq_j*t/N)
    for j in range(4):
        base = 2 + 3 * j
        amp = params[base]
        ph = params[base + 1]
        freq = params[base + 2]
        h += amp * np.cos(ph + 2.0 * np.pi * freq * t / N)
    # DC offset
    h += params[14]
    return h

def _objective_params(params):
    h = _chirp_harmonics(params)
    return _exact_value(h)

def _nm_polish(params0, max_iter=500):
    """Nelder-Mead on the parameter space."""
    try:
        res = minimize(
            _objective_params,
            params0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True},
        )
        if res.success or res.fun < 1e18:
            return res.x, res.fun
    except Exception:
        pass
    return params0, _objective_params(params0)

def construct():
    rng = np.random.default_rng(42)
    best_h = np.ones(N)
    best_val = _exact_value(best_h)  # 2.0

    # Bounds for differential evolution
    # [phase0, chirp_rate, amp1, ph1, freq1, amp2, ph2, freq2, amp3, ph3, freq3, amp4, ph4, freq4, dc_offset]
    bounds = [
        (0, 2 * np.pi),       # phase0
        (-10, 10),            # chirp_rate (quadratic term)
        (-1.0, 1.0),          # amp1
        (0, 2 * np.pi),       # ph1
        (0.1, 20.0),          # freq1 (cycles over N)
        (-1.0, 1.0),          # amp2
        (0, 2 * np.pi),       # ph2
        (0.1, 20.0),          # freq2
        (-1.0, 1.0),          # amp3
        (0, 2 * np.pi),       # ph3
        (0.1, 20.0),          # freq3
        (-1.0, 1.0),          # amp4
        (0, 2 * np.pi),       # ph4
        (0.1, 20.0),          # freq4
        (-2.0, 2.0),          # dc_offset
    ]

    # Differential evolution global search
    if _remaining() > 30.0:
        try:
            res_de = differential_evolution(
                _objective_params,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=False,  # we do our own polish
                init='sobol',
                updating='immediate',
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = _chirp_harmonics(res_de.x)
        except Exception:
            pass

    # If DE didn't improve much, try a few random parameter starts + NM
    n_random_starts = 8
    for i in range(n_random_starts):
        if _remaining() < 10.0:
            break
        params_rand = np.array([
            rng.uniform(0, 2 * np.pi),
            rng.uniform(-10, 10),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.1, 20.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.1, 20.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.1, 20.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.1, 20.0),
            rng.uniform(-2.0, 2.0),
        ])
        p_opt, f_opt = _nm_polish(params_rand, max_iter=300)
        if f_opt < best_val:
            best_val = f_opt
            best_h = _chirp_harmonics(p_opt)

    # Final polish: Nelder-Mead directly on the 400-dim h from best parameterization
    if _remaining() > 15.0:
        h0 = best_h.copy()
        # Small perturbation to escape local minima
        h0 += rng.normal(0, 0.01, N)
        try:
            res_nm = minimize(
                _exact_value,
                h0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_h = res_nm.x
        except Exception:
            pass

    # Coordinate-descent polish on exact objective for last bits
    if _remaining() > 5.0:
        h = best_h.copy()
        for _ in range(20):
            if _remaining() < 2.0:
                break
            improved = False
            for i in range(0, N, 4):
                if _remaining() < 2.0:
                    break
                base = h[i]
                for step in [0.001, -0.001, 0.005, -0.005, 0.0001, -0.0001]:
                    h[i] = base + step
                    v = _exact_value(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                        break
                if not improved:
                    h[i] = base
            if not improved:
                break
        best_h = h

    return best_h
