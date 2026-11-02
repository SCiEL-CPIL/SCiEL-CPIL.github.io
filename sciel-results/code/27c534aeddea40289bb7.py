"""Chebyshev spectral-envelope design for signed step functions.

The autoconvolution c(t) = (x*x)(t) has Fourier transform |X(f)|^2.  We want
min_t max|c(t)| / (sum x)^2.  By normalising sum x = 1 the problem becomes
minimising the L-infinity norm of the inverse Fourier transform of |X|^2.

We parameterise the spectral magnitude M(f) = |X(f)| on a discrete grid and
use an exchange (Remez-like) algorithm to flatten the periodic autoconvolution
C = IFFT(|X|^2).  The time-domain signal x is recovered by solving a small
phase-retrieval / least-squares problem that matches the target magnitude while
producing a real, signed sequence with unit sum.
"""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _spectral_to_time(mag, phase):
    """Convert spectral magnitude and phase to time-domain signal."""
    # mag has length N, phase has length N
    X = mag * np.exp(1j * phase)
    x = np.fft.ifft(X).real
    return x

def _time_to_spectral(x):
    """Get spectral magnitude and phase from time-domain signal."""
    X = np.fft.rfft(x, n=N)
    mag = np.abs(X)
    phase = np.angle(X)
    return mag, phase

def _chebyshev_exchange(mag, max_iters=500, tol=1e-10):
    """
    Remez-like exchange algorithm to flatten the periodic autoconvolution.
    
    We work with the spectral magnitude M(f) on the N-point DFT grid.
    The periodic autoconvolution is C = IFFT(M^2).
    We want to minimize max|C(t)| by adjusting M(f).
    
    The exchange algorithm maintains a set of 'extremal' frequency points where
    the constraint is active, and adjusts M at those points to equiripple the
    autoconvolution.
    """
    mag = mag.copy()
    mag = np.maximum(mag, 1e-10)  # Ensure positive
    
    # Initial phase: try to make x real and positive
    phase = np.zeros(N)
    
    # Compute initial autoconvolution
    M2 = mag ** 2
    C = np.fft.ifft(M2).real
    
    # Track best
    best_mag = mag.copy()
    best_C = C.copy()
    best_max = np.max(np.abs(C))
    
    # Extremal set: start with a few points
    n_ext = min(10, N)
    ext_indices = np.linspace(0, N - 1, n_ext, dtype=int)
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Compute current periodic autoconvolution
        M2 = mag ** 2
        C = np.fft.ifft(M2).real
        abs_C = np.abs(C)
        max_C = np.max(abs_C)
        
        # Update best
        if max_C < best_max:
            best_max = max_C
            best_mag = mag.copy()
            best_C = C.copy()
        
        # Find new extremal points in time domain (where |C| is near max)
        # We want to equiripple: adjust M so that |C| is as flat as possible
        # at the extremal time points
        
        # Get the top few time-domain peaks
        K = min(8, N)
        time_peaks = np.argsort(abs_C)[-K:]
        
        # For each time peak, the contribution to C[t] from frequency f is
        # M(f)^2 * exp(2j*pi*f*t/N).  To reduce the peak, we can adjust M(f).
        # 
        # Simple approach: at each iteration, find the frequency where the
        # gradient of max|C| with respect to M(f) is largest, and adjust M(f).
        
        # Compute gradient of C[t] with respect to M(f)
        # dC[t]/dM[f] = 2*M[f] * exp(2j*pi*f*t/N)
        
        # For the dominant time peak, find which frequency to adjust
        t_peak = time_peaks[0]
        
        # Gradient at the peak
        freqs = np.arange(N)
        grad = 2.0 * mag * np.exp(2j * np.pi * freqs * t_peak / N)
        
        # The sign of the real part tells us whether increasing or decreasing M helps
        # We want to reduce |C[t_peak]|, so we adjust M in the direction that
        # opposes C[t_peak]
        
        # Find the frequency with the largest opposing gradient
        # We want grad * conj(C[t_peak]) to be positive (to reduce magnitude)
        phase_C = np.angle(C[t_peak])
        effective_grad = np.real(grad * np.exp(-1j * phase_C))
        
        # Adjust M at the frequency with the largest effective gradient
        f_adj = np.argmax(np.abs(effective_grad))
        
        # Determine direction: if C[t_peak] is positive, decrease M where
        # grad is positive; if negative, increase M where grad is positive
        direction = -np.sign(np.real(C[t_peak] * np.conj(np.exp(1j * phase_C))))
        direction = np.sign(direction)
        if direction == 0:
            direction = 1.0
        
        # Step size
        step = 0.01 * mag[f_adj]
        mag[f_adj] += direction * step
        mag[f_adj] = max(mag[f_adj], 1e-10)
        
        # Also try adjusting a few other frequencies
        top_f = np.argsort(np.abs(effective_grad))[-3:]
        for f in top_f:
            if f == f_adj:
                continue
            grad_f = 2.0 * mag[f] * np.exp(2j * np.pi * f * t_peak / N)
            eff_g = np.real(grad_f * np.exp(-1j * phase_C))
            d = -np.sign(np.real(C[t_peak])) * np.sign(eff_g)
            mag[f] += d * 0.005 * mag[f]
            mag[f] = max(mag[f], 1e-10)
        
        # Check convergence
        if abs(max_C - best_max) < tol * best_max and it > 50:
            break
    
    return best_mag, best_max

def _phase_retrieval(mag, max_iters=200):
    """
    Recover a real time-domain signal x with |X(f)| ≈ mag(f) using
    alternating projections (Gerchberg-Saxton style).
    """
    # Initialize with random phase
    rng = np.random.default_rng(1)
    phase = rng.uniform(0, 2 * np.pi, size=N)
    
    best_x = None
    best_err = 1e10
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        # Forward: time -> frequency
        x = np.fft.ifft(mag * np.exp(1j * phase)).real
        
        # Enforce real-valued (already done by taking real part)
        # Enforce non-negative? No, we want signed.
        
        # Backward: frequency -> time
        X = np.fft.rfft(x, n=N)
        phase = np.angle(X)
        # Keep the magnitude fixed
        # (mag is already fixed, we only update phase)
        
        # Check error
        X_check = mag * np.exp(1j * phase)
        x_check = np.fft.ifft(X_check).real
        err = np.max(np.abs(np.abs(np.fft.rfft(x_check, n=N)) - mag[:N//2+1]))
        
        if err < best_err:
            best_err = err
            best_x = x_check.copy()
    
    if best_x is None:
        best_x = np.fft.ifft(mag * np.exp(1j * phase)).real
    
    return best_x

def _optimize_from_mag(mag):
    """Given a spectral magnitude, find the best time-domain signal."""
    # Phase retrieval
    x = _phase_retrieval(mag, max_iters=100)
    
    # Normalize to unit sum
    s = np.sum(x)
    if abs(s) < 1e-10:
        return x, 1e10
    x = x / s
    
    val = _objective(x)
    return x, val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Generate diverse spectral magnitude initializations
    t = np.linspace(0, 1, N)
    freqs = np.arange(N) / N  # Normalized frequency [0, 1)
    
    mag_initials = []
    
    # Flat spectrum (white noise)
    mag_initials.append(np.ones(N))
    
    # Low-pass filters
    for bw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        mag = np.exp(-freqs**2 / (2 * bw**2))
        mag_initials.append(mag)
    
    # Band-pass filters
    for center in [0.1, 0.2, 0.3, 0.5]:
        for bw in [0.05, 0.1, 0.2]:
            mag = np.exp(-(freqs - center)**2 / (2 * bw**2))
            mag_initials.append(mag)
    
    # Multi-band
    for c1 in [0.2, 0.3, 0.4]:
        for c2 in [0.6, 0.7, 0.8]:
            mag = np.exp(-(freqs - c1)**2 / 0.02) + np.exp(-(freqs - c2)**2 / 0.02)
            mag_initials.append(mag)
    
    # Cosine-modulated
    for mod_freq in [2, 4, 6, 8]:
        mag = 1.0 + 0.5 * np.cos(2 * np.pi * mod_freq * freqs)
        mag = np.maximum(mag, 0.1)
        mag_initials.append(mag)
    
    # Random smooth spectra
    rng = np.random.default_rng(42)
    for _ in range(15):
        n_harm = rng.integers(3, 8)
        harm_freqs = rng.uniform(1, 10, size=n_harm)
        harm_amps = rng.uniform(0.1, 0.5, size=n_harm)
        harm_phases = rng.uniform(0, 2 * np.pi, size=n_harm)
        mag = 1.0 + sum(a * np.cos(2 * np.pi * f * freqs + p) 
                        for a, f, p in zip(harm_amps, harm_freqs, harm_phases))
        mag = np.maximum(mag, 0.05)
        mag_initials.append(mag)
    
    # Phase 2: For each magnitude, run Chebyshev exchange then phase retrieval
    for mag0 in mag_initials:
        if _check_deadline():
            break
        
        # Normalize magnitude
        mag0 = mag0 / np.max(mag0)
        
        # Chebyshev exchange to flatten autoconvolution
        mag_opt, _ = _chebyshev_exchange(mag0, max_iters=100)
        
        # Phase retrieval to get time-domain signal
        x, val = _optimize_from_mag(mag_opt)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: Local refinement of the best signal
    # Use L-BFGS-B with a smooth surrogate
    
    def _smooth_obj(x, beta=50.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(x)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-15:
            return 1e10, np.zeros_like(x)
        
        shifted = beta * (abs_ac - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
        val = soft_max / (s * s)
        
        g = weights * np.sign(ac)
        
        L_fft = 1
        while L_fft < 2 * N - 1:
            L_fft *= 2
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_softmax_dx = term1 + term2
        grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
        
        return val, grad
    
    # Try multiple beta values
    for beta in [30.0, 50.0, 80.0, 120.0]:
        if _check_deadline():
            break
        
        def _obj_beta(x, b=beta):
            return _smooth_obj(x, beta=b)
        
        try:
            result = minimize(_obj_beta, best_x, method='L-BFGS-B', jac=True,
                            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-12})
            if result.fun < best_val:
                # Verify with exact objective
                val_exact = _objective(result.x)
                if val_exact < best_val:
                    best_val = val_exact
                    best_x = result.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Coordinate descent polish
    def _coordinate_descent(x, max_iters=2000, step_scale=0.05):
        best_val_cd = _objective(x)
        best_x_cd = x.copy()
        
        for it in range(max_iters):
            if _check_deadline():
                break
            
            improved = False
            for j in range(N):
                for delta in [-step_scale, -step_scale*0.5, step_scale*0.5, step_scale]:
                    trial = best_x_cd.copy()
                    trial[j] += delta
                    val = _objective(trial)
                    if val < best_val_cd:
                        best_val_cd = val
                        best_x_cd = trial.copy()
                        improved = True
                        break
                
                if _check_deadline():
                    break
            
            if not improved:
                break
        
        return best_x_cd, best_val_cd
    
    best_x, best_val = _coordinate_descent(best_x, max_iters=1500, step_scale=0.03)
    
    if _check_deadline():
        return best_x
    
    # Phase 5: Peak-targeted polish
    def _peak_targeted_polish(x, max_iters=1500):
        best_val_pt = _objective(x)
        best_x_pt = x.copy()
        
        for it in range(max_iters):
            if _check_deadline():
                break
            
            ac_current = _autoconv_fft(best_x_pt)
            abs_ac_current = np.abs(ac_current)
            
            K = min(15, 2*N-1)
            peak_indices = np.argsort(abs_ac_current)[-K:]
            
            improved = False
            for peak_idx in peak_indices:
                if _check_deadline():
                    break
                
                j_range = np.arange(max(0, peak_idx - N + 1), min(N, peak_idx + 1))
                if len(j_range) == 0:
                    continue
                
                partner_indices = peak_idx - j_range
                valid = (partner_indices >= 0) & (partner_indices < N)
                j_range = j_range[valid]
                
                if len(j_range) == 0:
                    continue
                
                for j in j_range[:min(30, len(j_range))]:
                    if _check_deadline():
                        break
                    
                    for delta in [-0.05, -0.02, 0.02, 0.05]:
                        trial = best_x_pt.copy()
                        trial[j] += delta
                        val = _objective(trial)
                        if val < best_val_pt:
                            best_val_pt = val
                            best_x_pt = trial.copy()
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_x_pt, best_val_pt
    
    best_x, best_val = _peak_targeted_polish(best_x, max_iters=1000)
    
    if _check_deadline():
        return best_x
    
    # Phase 6: Random perturbation refinement
    rng = np.random.default_rng(99)
    step_size = 0.03
    stall_count = 0
    
    for iteration in range(30000):
        if _check_deadline():
            break
        
        perturbation = rng.normal(0, step_size, size=N)
        x_trial = best_x + perturbation
        
        val_smooth, _ = _smooth_obj(x_trial, beta=60.0)
        
        if val_smooth < best_val * 0.999:
            val_exact = _objective(x_trial)
            if val_exact < best_val:
                best_val = val_exact
                best_x = x_trial.copy()
                stall_count = 0
                step_size = max(0.001, step_size * 0.9998)
            else:
                stall_count += 1
        else:
            stall_count += 1
        
        if stall_count % 5000 == 0:
            step_size = min(0.3, step_size * 1.05)
    
    # Phase 7: Final BFGS polish
    if not _check_deadline():
        try:
            result = minimize(_objective, best_x, method='BFGS', 
                            options={'maxiter': 100, 'gtol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
