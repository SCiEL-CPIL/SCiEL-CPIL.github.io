"""Combined program: chirp parameterization + low-frequency DFT basis + L-BFGS-B refinement.

Strategy:
- Keep the chirp-based parameterization (h[i] = base + A*cos(phase + rate*i^2/N)) as the global search space.
- Use differential evolution for global search over chirp parameters.
- Then refine using L-BFGS-B on a low-frequency DFT basis representation, initialized from the best chirp.
- Also try a direct Nelder-Mead polish on the chirp parameters with tight tolerances.
- Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

# Low-frequency DFT basis for refinement
_K = 15  # number of frequency components
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K, N))
for k in range(1, _K + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    # coeffs shape: (2*K,), base_val is scalar
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    # Phase 1: Global search via differential evolution on chirp parameters
    # params: [phase, rate, amp, base]
    bounds = [
        (-np.pi, np.pi),      # phase
        (-200.0, 200.0),      # rate (quadratic phase coefficient)
        (-3.0, 3.0),          # amplitude
        (-3.0, 3.0),          # base
    ]
    
    try:
        if deadline_ok():
            res = differential_evolution(
                _chirp_obj,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead polish on chirp parameters
    if deadline_ok():
        try:
            # Start from the best found, or from a reasonable default
            start_params = [0.0, 10.0, 1.0, 1.0]
            # If we have a better best_h, try to estimate chirp params from it
            # For simplicity, just run Nelder-Mead from a few starting points
            for start in [
                [0.0, 10.0, 1.0, 1.0],
                [0.0, -10.0, 1.0, 1.0],
                [np.pi/2, 50.0, 0.5, 1.0],
                [0.0, 100.0, 1.5, 0.5],
            ]:
                if not deadline_ok():
                    break
                res2 = minimize(
                    _chirp_obj,
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300},
                )
                if res2.x is not None:
                    h = _chirp_h(res2.x)
                    v = _value(h)
                    if v < best_v:
                        best_v = v
                        best_h = h.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement on DFT basis
    # Initialize DFT coefficients from best_h
    if deadline_ok():
        try:
            # Compute DFT coefficients from best_h
            h_best = best_h
            mean_h = np.mean(h_best)
            # Project onto DFT basis
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                cos_k = np.cos(angle)
                sin_k = np.sin(angle)
                # Least squares projection
                A_k = np.column_stack([cos_k, sin_k])
                target = h_best - mean_h
                # Solve A_k @ [c, s] = target
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            # Full parameter vector: [coeffs..., base]
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res3 = minimize(
                _dft_obj,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res3.x is not None:
                h = _dft_h(res3.x[:-1], res3.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            # Re-initialize from current best_h
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                cos_k = np.cos(angle)
                sin_k = np.sin(angle)
                A_k = np.column_stack([cos_k, sin_k])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res4 = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000},
            )
            if res4.x is not None:
                h = _dft_h(res4.x[:-1], res4.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
