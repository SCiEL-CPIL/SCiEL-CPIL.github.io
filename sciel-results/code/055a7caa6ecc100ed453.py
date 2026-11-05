"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, minimize_scalar

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using Zadoff-Chu sequences modulated by chirps and windowed.
    Optimizes DC offset analytically for each candidate shape.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    
    def get_h_ac(params):
        """Returns the AC component (without DC) of the signal."""
        mu, alpha, win_param = params
        # Zadoff-Chu like sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp modulation
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        signal = zc * chirp
        
        # Windowing: Smooth Gaussian window
        # win_param controls width. Map to sigma.
        # Let's use a smooth window: exp(-0.5 * ((t - center) / sigma)^2)
        # win_param in [0, 1] maps to sigma from large (rect) to small (narrow).
        # Or just use win_param as sigma directly?
        # Let's stick to a smooth interpolation or a single family.
        # Gaussian is smooth.
        # Let win_param be in [0.1, 5.0] representing sigma in units of N/2?
        # Let's normalize t to [-1, 1].
        n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
        
        # Use a generalized window to allow both rect and gaussian
        # If win_param is close to 1, it's rectangular.
        # Let's use: window = 1 - win_param + win_param * exp(-0.5 * (n_norm / sigma)^2)
        # This is not smooth in derivative at the switch, but let's just use Gaussian
        # with a variable width, and allow it to be very wide (approx rect).
        
        # Let's map win_param (0 to 1) to sigma (0.2 to 1.5)
        sigma = 0.2 + win_param * 1.3
        window = np.exp(-0.5 * (n_norm / sigma)**2)
        
        h_ac = np.real(signal) * window
        return h_ac

    def optimize_dc(x):
        """Finds the optimal DC offset c to add to x to minimize the value."""
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U = np.convolve(x, np.ones(N))
        T = np.convolve(np.ones(N), np.ones(N))
        
        # Pre-compute max index? No, max changes with c.
        
        def obj(c):
            conv_c = C_x + 2*c*U + c**2*T
            m = np.max(conv_c)
            s = S_x + N*c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val
        
        # The function is unimodal-ish. Bounds for c:
        # If x is zero mean, optimal c is small.
        # Let's search in [-5, 5].
        res = minimize_scalar(obj, bounds=(-5, 5), method='bounded', options={'xatol': 1e-6})
        if res.success:
            return x + res.x
        else:
            return x # fallback

    def get_h_full(params):
        x = get_h_ac(params)
        h = optimize_dc(x)
        return h

    # Use Differential Evolution for global search
    # Bounds: mu, alpha, win_param
    bounds = [
        (0, 20),       # mu
        (-1, 1),       # alpha
        (0, 1)         # win_param (maps to sigma)
    ]
    
    np.random.seed(42)
    
    try:
        res_de = differential_evolution(
            lambda p: evaluate(get_h_full(p)),
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False
        )
        
        best_candidate_val = res_de.fun
        best_params = res_de.x
        best_candidate = get_h_full(best_params).copy()
        
        # Local refinement with Powell
        res_local = minimize(
            lambda p: evaluate(get_h_full(p)),
            best_params,
            method='Powell',
            options={'maxiter': 500, 'maxfev': 5000}
        )
        
        if res_local.fun < best_candidate_val:
            best_candidate_val = res_local.fun
            best_params = res_local.x
            best_candidate = get_h_full(best_params).copy()
            
    except Exception:
        pass

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
