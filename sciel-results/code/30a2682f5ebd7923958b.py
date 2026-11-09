"""DFT-based spectral root-finder with Newton refinement."""
import math
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
    x0 = float(x0)
    
    # --- Step 1: Initial sampling to detect oscillation characteristics ---
    # We'll sample at N points over a window around x0.
    # Start with a modest window; adjust if needed.
    
    N = 16  # number of samples for DFT
    # Initial window size: try a few scales
    base_window = max(1.0, abs(x0) * 0.1, 0.1)
    
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Try a few window sizes to find one that captures at least ~2 periods
    windows = [base_window, base_window * 2, base_window * 0.5, base_window * 4, base_window * 0.25]
    
    for window in windows:
        if window <= 0:
            continue
        # Sample N+1 points (N intervals) over [-window, window] centered at x0
        xs = x0 + np.linspace(-window, window, N + 1)
        # Remove duplicates if any
        xs = np.unique(xs)
        if len(xs) < 4:
            continue
        N_actual = len(xs)
        
        try:
            fs = np.array([f(float(x)) for x in xs])
        except Exception:
            continue
        
        if not np.all(np.isfinite(fs)):
            continue
        
        # Track best so far
        for xi, fi in zip(xs, fs):
            if abs(fi) < best_abs:
                best_abs = abs(fi)
                best_x = float(xi)
                best_fx = float(fi)
        
        if best_abs <= 1e-8:
            return best_x
        
        # --- Step 2: DFT to estimate dominant frequency and phase ---
        # Mean-center
        f_mean = np.mean(fs)
        fs_centered = fs - f_mean
        
        # Compute DFT
        F = np.fft.fft(fs_centered)
        freqs = np.fft.fftfreq(N_actual, d=(2 * window) / (N_actual - 1))
        
        # Find dominant frequency (excluding DC)
        magnitudes = np.abs(F)
        # Exclude index 0 (DC)
        magnitudes[0] = 0
        if len(magnitudes) < 2:
            continue
        idx = np.argmax(magnitudes)
        if magnitudes[idx] < 1e-10:
            continue
        dominant_freq = abs(freqs[idx])
        if dominant_freq < 1e-10:
            continue
        
        # Estimate period
        period = 1.0 / dominant_freq
        
        # --- Step 3: Predict nearest zero-crossing ---
        # The fitted sinusoid is: A * cos(2*pi*f*(x - x0) + phi) + mean
        # We want to find x where this is zero.
        # From the DFT, we can extract amplitude and phase.
        
        # Amplitude and phase from DFT
        A = 2.0 * magnitudes[idx] / N_actual
        phase = np.angle(F[idx])
        
        # The signal is approx: A * cos(2*pi*dominant_freq * (x - x0) + phase) + f_mean
        # Wait, the DFT gives us coefficients for the discrete samples. Let's be more careful.
        # For a real signal, x[n] = A*cos(2*pi*k*n/N + phi)
        # The DFT coefficient at k is: X[k] = (N/2) * A * exp(i*phi) for k > 0
        # So A = 2*|X[k]|/N, phi = angle(X[k])
        
        # The zero of A*cos(2*pi*f*(x-x0) + phi) + f_mean = 0
        # cos(2*pi*f*(x-x0) + phi) = -f_mean / A
        # 2*pi*f*(x-x0) + phi = arccos(-f_mean/A) + 2*pi*m  OR  -arccos(-f_mean/A) + 2*pi*m
        
        if abs(A) < 1e-10:
            continue
        ratio = -f_mean / A
        if ratio < -1.0:
            ratio = -1.0
        elif ratio > 1.0:
            ratio = 1.0
        
        theta = np.arccos(ratio)
        
        # Two families of solutions:
        # 2*pi*f*(x-x0) + phi = theta + 2*pi*m
        # 2*pi*f*(x-x0) + phi = -theta + 2*pi*m
        
        # Solve for x:
        # x = x0 + (theta - phi + 2*pi*m) / (2*pi*f)
        # x = x0 + (-theta - phi + 2*pi*m) / (2*pi*f)
        
        candidates = []
        for m in range(-3, 4):
            x1 = x0 + (theta - phase + 2 * math.pi * m) / (2 * math.pi * dominant_freq)
            x2 = x0 + (-theta - phase + 2 * math.pi * m) / (2 * math.pi * dominant_freq)
            candidates.append(float(x1))
            candidates.append(float(x2))
        
        # Evaluate candidates
        for xc in candidates:
            if not np.isfinite(xc):
                continue
            try:
                fxc = f(xc)
            except Exception:
                continue
            if not np.isfinite(fxc):
                continue
            if abs(fxc) < best_abs:
                best_abs = abs(fxc)
                best_x = xc
                best_fx = fxc
        
        if best_abs <= 1e-8:
            return best_x
        
        # If we found a good candidate, break to Newton refinement
        if best_abs < abs(best_fx) or True:
            break
    
    # --- Step 4: Newton refinement from best_x ---
    x = float(best_x)
    fx = best_fx
    for _ in range(40):
        if abs(fx) <= 1e-8:
            return x
        h = 1e-6 * max(1.0, abs(x))
        try:
            fxp = f(x + h)
            fxm = f(x - h)
        except Exception:
            break
        if not (np.isfinite(fxp) and np.isfinite(fxm)):
            break
        d = (fxp - fxm) / (2 * h)
        if d == 0 or not np.isfinite(d):
            break
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
        # Damping if step is too large
        step = abs(x_new - x)
        if step > max(1.0, abs(x)) * 0.5:
            x_new = x + math.copysign(min(step, max(1.0, abs(x)) * 0.1), x_new - x)
        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not np.isfinite(fx):
            break
    
    return x
# EVOLVE-BLOCK-END
