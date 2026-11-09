"""DFT-based spectral root-finder with Newton refinement."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy:
      1. Sample f at a small number of equally-spaced points around x0.
      2. Use a DFT to estimate the dominant frequency and phase of the oscillation.
      3. Predict the nearest zero-crossing from the fitted sinusoid.
      4. Refine with Newton's method (finite-difference derivative).
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    x0 = float(x0)
    
    # --- Step 1: Initial sampling to detect oscillation characteristics ---
    # We'll sample at N points over a window around x0.
    # Start with a modest window; adjust if needed.
    
    best_x = x0
    best_fx = None
    try:
        best_fx = f(x0)
    except:
        best_fx = None
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return x0
    
    # Try multiple window sizes to find a good fit
    N = 16  # number of sample points
    window_sizes = [1.0, 0.5, 2.0, 0.25, 4.0, 0.1, 8.0, 0.05]
    
    predicted_roots = []
    
    for w in window_sizes:
        if time.time() > deadline - 10.0:
            break
        
        # Sample points centered around x0
        # We want to cover roughly one or two periods
        x_samples = x0 + np.linspace(-w, w, N)
        
        try:
            f_samples = np.array([f(x) for x in x_samples], dtype=float)
        except:
            continue
        
        if not np.all(np.isfinite(f_samples)):
            continue
        
        # Check if we already have a good root
        for i, fs in enumerate(f_samples):
            if abs(fs) <= 1e-8:
                return float(x_samples[i])
        
        # Compute DFT to find dominant frequency
        # Remove mean
        f_mean = np.mean(f_samples)
        f_centered = f_samples - f_mean
        
        # Compute FFT
        F = np.fft.rfft(f_centered)
        freqs = np.fft.rfftfreq(N, d=(2*w)/N)
        
        # Find dominant frequency (skip DC component)
        if len(F) < 2:
            continue
        
        magnitudes = np.abs(F)
        magnitudes[0] = 0  # skip DC
        
        if np.max(magnitudes) < 1e-10:
            continue
        
        k_max = np.argmax(magnitudes)
        omega = 2 * math.pi * freqs[k_max]  # angular frequency
        
        if omega <= 0 or not np.isfinite(omega):
            continue
        
        # Get phase from the complex coefficient
        phase = np.angle(F[k_max])
        
        # The fitted sinusoid: f(x) ≈ A*cos(omega*(x - x0) + phase) + f_mean
        # Actually, the FFT gives us coefficients for the sampled signal.
        # Let's reconstruct: f(x) ≈ Re[C * exp(i*omega*x)] + f_mean
        # where C = |F[k]|/N * exp(i*phase) approximately
        
        # For a cosine fit: f(x) ≈ A * cos(omega*x + phi) + f_mean
        # Zero crossings occur when omega*x + phi = pi/2 + n*pi
        # x = (pi/2 + n*pi - phi) / omega
        
        # Find the phase such that the model matches
        # The DFT coefficient at frequency k corresponds to:
        # f(t) ≈ (2/N) * |F[k]| * cos(2*pi*k*t/N + phase)
        # In terms of x: t = (x - x0 + w) / (2*w) * N  ... this is getting complicated.
        
        # Simpler approach: fit a sinusoid directly to the samples
        # f(x) ≈ a * cos(omega*x) + b * sin(omega*x) + c
        # Using linear least squares
        
        x_arr = x_samples
        A_mat = np.column_stack([np.cos(omega * x_arr), np.sin(omega * x_arr), np.ones(N)])
        
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, f_samples, rcond=None)
        except:
            continue
        
        a, b, c = coeffs
        
        # Find zeros of a*cos(omega*x) + b*sin(omega*x) + c = 0
        # R*cos(omega*x - delta) + c = 0
        # where R = sqrt(a^2 + b^2), delta = atan2(b, a)
        
        R = math.sqrt(a*a + b*b)
        if R < 1e-15:
            continue
        
        # R*cos(omega*x - delta) = -c
        # cos(omega*x - delta) = -c/R
        ratio = -c / R
        if ratio < -1.0 or ratio > 1.0:
            continue
        
        delta = math.atan2(b, a)
        base_angle = math.acos(ratio)
        
        # Solutions: omega*x - delta = ±base_angle + 2*pi*n
        # x = (delta ± base_angle + 2*pi*n) / omega
        
        candidates = []
        for sign in [1, -1]:
            for n in range(-3, 4):
                x_cand = (delta + sign * base_angle + 2 * math.pi * n) / omega
                if np.isfinite(x_cand):
                    candidates.append(x_cand)
        
        # Evaluate candidates and keep the best ones
        for xc in candidates:
            if time.time() > deadline - 5.0:
                break
            try:
                fxc = f(xc)
            except:
                continue
            if not np.isfinite(fxc):
                continue
            if abs(fxc) < (abs(best_fx) if best_fx is not None else float('inf')):
                best_fx = fxc
                best_x = float(xc)
            if abs(fxc) <= 1e-8:
                return float(xc)
            predicted_roots.append((abs(fxc), float(xc)))
    
    # Sort predicted roots by |f(x)|
    predicted_roots.sort(key=lambda t: t[0])
    
    # Try Newton refinement from the best candidates
    for _, xr in predicted_roots[:10]:
        if time.time() > deadline - 5.0:
            break
        x = xr
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            try:
                fx = f(x)
            except:
                break
            if not np.isfinite(fx):
                break
            if abs(fx) <= 1e-12:
                return x
            if abs(fx) < (abs(best_fx) if best_fx is not None else float('inf')):
                best_fx = fx
                best_x = x
            if abs(fx) <= 1e-8:
                return x
            h = 1e-7 * max(1.0, abs(x))
            try:
                d = (f(x + h) - fx) / h
            except:
                break
            if d == 0 or not np.isfinite(d):
                break
            x = x - fx / d
            if not np.isfinite(x):
                break
    
    # If nothing worked well, try a simple bisection-like search around x0
    if time.time() < deadline - 5.0:
        # Do a coarse scan
        for delta_x in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 5.0, -5.0]:
            if time.time() > deadline - 2.0:
                break
            x_try = x0 + delta_x
            try:
                fx_try = f(x_try)
            except:
                continue
            if not np.isfinite(fx_try):
                continue
            if abs(fx_try) < (abs(best_fx) if best_fx is not None else float('inf')):
                best_fx = fx_try
                best_x = float(x_try)
            if abs(fx_try) <= 1e-8:
                return float(x_try)
    
    return best_x
# EVOLVE-BLOCK-END
