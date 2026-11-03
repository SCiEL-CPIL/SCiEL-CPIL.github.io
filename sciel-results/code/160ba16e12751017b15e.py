"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600

# EVOLVE-BLOCK-START

def _compute_ratio_fast(h):
    """Fast computation of normalized max autoconvolution ratio."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    padded = np.zeros(2 * n, dtype=np.float64)
    padded[:n] = h
    F = np.fft.fft(padded)
    conv = np.fft.ifft(F * np.conj(F)).real
    max_conv = np.max(conv[:2*n-1])
    total = np.sum(h)
    if total == 0:
        return 0.0
    dx = 1.0 / (2 * n)
    return max_conv / (total * total * dx)

def _envelope_modulated_heights(params):
    """
    Generate heights as product of a raised-cosine envelope and a harmonic modulation.
    params = [env_center, env_width, mod_freq, mod_phase, mod_amp, offset]
    - env_center: center of raised cosine envelope (0 to N-1)
    - env_width: width of envelope (fraction of N)
    - mod_freq: number of periods of modulation across the domain
    - mod_phase: phase of modulation (0 to 2*pi)
    - mod_amp: amplitude of modulation (0 to 1)
    - offset: DC offset to ensure non-negativity (0 to 1)
    """
    env_center = params[0]
    env_width = params[1]
    mod_freq = params[2]
    mod_phase = params[3]
    mod_amp = params[4]
    offset = params[5]
    
    t = np.arange(N, dtype=np.float64)
    t_norm = (t - env_center) / (env_width * N)
    
    # Raised cosine envelope: 1 inside [-1,1], 0 outside
    env = np.where(np.abs(t_norm) <= 1.0, 
                   0.5 * (1.0 + np.cos(np.pi * t_norm)), 
                   0.0)
    
    # Harmonic modulation
    modulation = 1.0 + mod_amp * np.cos(2.0 * np.pi * mod_freq * t / N + mod_phase)
    
    # Combine: envelope * (offset + modulation)
    h = env * (offset + modulation)
    
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    # Normalize scale so that average height is 1
    if np.sum(h) > 0:
        h = h / np.sum(h) * N
    
    return h

def _objective_envelope_mod(params):
    """Objective function for envelope-modulated parametric optimization."""
    h = _envelope_modulated_heights(params)
    return _compute_ratio_fast(h)

def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial best: uniform
    best_heights = np.ones(N, dtype=np.float64)
    best_score = _compute_ratio_fast(best_heights)
    
    t = np.arange(N, dtype=np.float64)
    
    # Try several parametric models
    # Raised cosine
    h_cos = 0.5 * (1.0 + np.cos(np.pi * t / (N - 1)))
    h_cos = np.maximum(h_cos, 0.0)
    if np.sum(h_cos) > 0:
        h_cos = h_cos / np.sum(h_cos) * N
    score_cos = _compute_ratio_fast(h_cos)
    if score_cos < best_score:
        best_score = score_cos
        best_heights = h_cos.copy()
    
    # Gaussian with various sigmas
    center = N / 2.0
    for sigma_frac in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        sigma = sigma_frac * N
        h_gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        h_gauss = np.maximum(h_gauss, 0.0)
        if np.sum(h_gauss) > 0:
            h_gauss = h_gauss / np.sum(h_gauss) * N
        score_gauss = _compute_ratio_fast(h_gauss)
        if score_gauss < best_score:
            best_score = score_gauss
            best_heights = h_gauss.copy()
    
    # Use Differential Evolution for global search in envelope-modulated space
    if time.time() < deadline - 30.0:
        try:
            # Bounds: [env_center, env_width, mod_freq, mod_phase, mod_amp, offset]
            bounds = [
                (0.2 * N, 0.8 * N),   # env_center
                (0.1, 0.5),           # env_width (fraction of N)
                (1.0, 20.0),          # mod_freq
                (0.0, 2.0 * np.pi),   # mod_phase
                (0.0, 1.0),           # mod_amp
                (0.0, 1.0)            # offset
            ]
            
            result_de = differential_evolution(
                _objective_envelope_mod,
                bounds=bounds,
                seed=42,
                maxiter=80,
                popsize=20,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            
            if result_de.fun < best_score:
                h_opt = _envelope_modulated_heights(result_de.x)
                h_opt = np.maximum(h_opt, 0.0)
                score = _compute_ratio_fast(h_opt)
                if score < best_score:
                    best_score = score
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B on the full height vector
    if time.time() < deadline - 10.0:
        try:
            h0 = best_heights.copy()
            
            def _smooth_objective(h):
                h_pos = np.maximum(h, 0.0)
                score = _compute_ratio_fast(h_pos)
                # Light smoothness penalty
                diff = np.diff(h_pos)
                penalty = 1e-4 * np.sum(diff**2)
                return score + penalty
            
            result_local = minimize(
                _smooth_objective,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            
            if result_local.fun < best_score + 1e-6:
                h_opt = np.maximum(result_local.x, 0.0)
                score = _compute_ratio_fast(h_opt)
                if score < best_score:
                    best_score = score
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    return best_heights.tolist()

# EVOLVE-BLOCK-END
