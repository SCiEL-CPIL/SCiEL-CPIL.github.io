"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 600
DELTA = 0.5 / N  # interval width

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    x = np.linspace(-0.25, 0.25, N, endpoint=False)

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if h.size != N:
            h = np.resize(h, N)
        h = np.maximum(h, 0.0)
        
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
            
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.rfft(h_padded)
        autoconv_circ = np.fft.irfft(H * H, n_fft)
        
        # Linear autoconvolution is in the first 2N-1 entries
        autoconv_vals = autoconv_circ[:2*N-1] * DELTA
        max_autoconv = np.max(autoconv_vals)
        
        l2_sq = np.sum(h**2) * DELTA
        if l2_sq < 1e-15:
            return float('inf')
        
        return max_autoconv / l2_sq

    def envelope_modulation_to_h(params):
        """
        Convert parameters to heights.
        params: [env_width, env_center, mod_freq, mod_phase, mod_amp, base]
        h(x) = base + env(x) * (1 + mod_amp * cos(mod_freq * x + mod_phase))
        Ensures non-negativity by clipping.
        """
        env_width = max(0.01, abs(params[0]))
        env_center = np.clip(params[1], -0.25, 0.25)
        mod_freq = abs(params[2])
        mod_phase = params[3]
        mod_amp = np.clip(params[4], 0.0, 1.0)  # Keep modulation bounded to help non-negativity
        base = max(0.0, params[5])
        
        env = np.exp(-0.5 * ((x - env_center) / env_width) ** 2)
        mod = 1.0 + mod_amp * np.cos(mod_freq * x + mod_phase)
        
        h = base + env * mod
        return np.maximum(h, 0.0)

    def objective(params):
        h = envelope_modulation_to_h(params)
        return autoconv_max_ratio(h)

    def coordinate_descent_refine(h, max_iters=200, step_decay=0.95):
        """Refine h using coordinate descent to minimize autoconv_max_ratio."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        step_size = 0.05
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try increasing and decreasing h[i]
                h_trial = h.copy()
                h_trial[i] = max(0, h[i] + step_size)
                trial_score = autoconv_max_ratio(h_trial)
                if trial_score < current_score:
                    h = h_trial
                    current_score = trial_score
                    improved = True
                    continue
                    
                h_trial[i] = max(0, h[i] - step_size)
                trial_score = autoconv_max_ratio(h_trial)
                if trial_score < current_score:
                    h = h_trial
                    current_score = trial_score
                    improved = True
            
            if not improved:
                step_size *= step_decay
                if step_size < 1e-4:
                    break
            
        return h, current_score

    best_score = float('inf')
    best_h = None

    # 1. Differential Evolution on the parametric envelope + modulation model
    # This explores a structured family of functions that are smooth and non-negative.
    bounds = [
        (0.01, 0.2),   # env_width
        (-0.2, 0.2),   # env_center
        (0.0, 50.0),   # mod_freq
        (0.0, 2*np.pi),# mod_phase
        (0.0, 1.0),    # mod_amp
        (0.0, 2.0),    # base
    ]
    
    try:
        res = optimize.differential_evolution(
            objective, bounds, seed=42, maxiter=50, tol=1e-5, popsize=15, polish=True
        )
        if res.fun < best_score:
            best_score = res.fun
            best_h = envelope_modulation_to_h(res.x).copy()
    except Exception:
        pass

    # 2. If DE didn't find anything or is still slow, fallback to diverse candidates
    if best_h is None or time.time() < deadline - 30:
        centers = np.arange(N) - N // 2
        candidates = []
        
        # Gaussian peaks
        for sigma in [1, 2, 5, 10, 20, 50, 100]:
            h = np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
        
        # Triangular peaks
        for width in [10, 50, 100, 200, 300]:
            h = np.maximum(0, 1 - np.abs(centers) / width)
            candidates.append(h)
            
        # Box functions
        for half_width in [10, 50, 100, 200, 300]:
            h = np.zeros(N)
            mask = np.abs(centers) <= half_width
            h[mask] = 1.0
            candidates.append(h)

        # Gaussian + base
        for sigma in [5, 20, 50]:
            for base in [0.1, 0.3, 0.5]:
                h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
                candidates.append(h)

        for h in candidates:
            if time.time() > deadline - 20:
                break
            h = np.asarray(h, dtype=np.float64)
            h = np.maximum(h, 0)
            if np.sum(h) < 1e-10:
                continue
            score = autoconv_max_ratio(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()

    # 3. Refine the best candidate with coordinate descent
    if best_h is not None and time.time() < deadline - 5:
        refined_h, refined_score = coordinate_descent_refine(best_h, max_iters=100, step_decay=0.9)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
