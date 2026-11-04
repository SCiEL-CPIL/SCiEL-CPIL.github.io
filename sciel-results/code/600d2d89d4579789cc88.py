"""Erdos minimum overlap - multi-harmonic flat autocorrelation optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
TOTAL_TIME_LIMIT = 110.0  # safety margin before 120s kill


def _compute_overlap_max(h, num_shifts=None):
    """Compute max overlap of h with shifted (1-h).
    
    overlap(s) = integral h(x)*(1-h)(x+s) dx = 1 - autocorr_h(s)
    We return max_s overlap(s) = 1 - min_s autocorr_h(s)
    """
    if num_shifts is None:
        num_shifts = N
    # h is step function: h[i] is value on interval [2i/N, 2(i+1)/N)
    # autocorr_h(s) for shift s: integral h(x)h(x+s) dx
    # For step functions, compute via discrete correlation
    # Shift s in units of step width: s = k * 2/N for k=0..N-1 (periodic)
    # autocorr at shift k: (2/N) * sum_i h[i] * h[(i+k) % N]
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # autocorr for all shifts 0..n-1
    # autocorr[k] = (2/n) * sum_i h[i]*h[(i+k)%n]
    # Use FFT for efficiency
    H = np.fft.rfft(h, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)[:n] * (2.0 / n)
    # acf[k] = (2/n) * sum_i h[i]*h[(i+k)%n] for k=0..n-1
    # overlap at shift k = 1 - acf[k]
    max_overlap = 1.0 - np.min(acf)
    return max_overlap


def _soft_max(vals, temperature=0.01):
    """Soft approximation of max using log-sum-exp."""
    m = np.max(vals)
    return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))


def construct():
    start_time = time.time()
    
    n = N
    # Initial: multi-harmonic with several frequencies
    # h(x) = 0.5 + sum of harmonics
    # Start with a reasonable multi-harmonic profile
    x = np.arange(n) * (2.0 / n)  # positions in [0, 2)
    
    # Initial multi-harmonic: use frequencies that spread energy
    # Frequencies: 1, 2, 3, 5, 7, 11, 13, 17, 19 (primes to avoid resonance)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = 0.15 / np.sqrt(len(freqs))
    phases = np.random.default_rng(42).uniform(0, 2*np.pi, len(freqs))
    
    h = np.zeros(n)
    for i, (f, a, ph) in enumerate(zip(freqs, amplitudes, phases)):
        h += a * np.cos(f * np.pi * x + ph)
    
    # Project to [0, 1] with sum = n/2
    h = np.clip(h, 0, 1)
    # Adjust to satisfy sum constraint
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (n / 2.0) / current_sum
        h = np.clip(h, 0, 1)
        # Re-adjust after clipping
        for _ in range(50):
            s = np.sum(h)
            if abs(s - n/2.0) < 1e-10:
                break
            # Distribute the difference
            diff = n/2.0 - s
            # Adjust elements that can absorb the change
            if diff > 0:
                mask = h < 1.0
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
            else:
                mask = h > 0.0
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
    
    best_h = h.copy()
    best_score = _compute_overlap_max(best_h)
    
    elapsed = time.time() - start_time
    if elapsed > TOTAL_TIME_LIMIT * 0.3:
        return best_h
    
    # Optimization: minimize max overlap = 1 - min autocorr
    # Use projected gradient descent on h directly
    # Or better: work in a parameterization that preserves constraints
    
    # Alternative: use scipy minimize with penalty
    def objective(params):
        """Penalty-based objective: maximize min autocorr = minimize max overlap + penalty for constraint violation"""
        h_p = np.clip(params, 0, 1)
        # Soft penalty for sum constraint
        sum_penalty = 100.0 * (np.sum(h_p) - n/2.0) ** 2
        # Max overlap via soft-max
        H = np.fft.rfft(h_p, n=2*n)
        acf = np.fft.irfft(H * np.conj(H), n=2*n)[:n] * (2.0 / n)
        overlap_vals = 1.0 - acf
        # Use soft-max for differentiability
        sm = _soft_max(overlap_vals, temperature=0.02)
        return sm + sum_penalty
    
    def gradient(params):
        """Numerical gradient"""
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(n):
            params[i] += eps
            f1 = objective(params)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad
    
    # Use L-BFGS-B with bounds
    from scipy.optimize import minimize as sp_minimize
    
    best_h_local = h.copy()
    best_obj = objective(best_h_local)
    
    # Run multiple restarts with different initializations
    rng = np.random.default_rng(123)
    max_iterations = 300
    
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIME_LIMIT * 0.7:
            break
        
        # Different initial conditions
        if restart == 0:
            h_init = h.copy()
        elif restart == 1:
            # Add noise to current best
            h_init = best_h_local + rng.normal(0, 0.05, n)
            h_init = np.clip(h_init, 0, 1)
        elif restart == 2:
            # Different frequency set
            x = np.arange(n) * (2.0 / n)
            freqs2 = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
            amplitudes2 = 0.12 / np.sqrt(len(freqs2))
            phases2 = rng.uniform(0, 2*np.pi, len(freqs2))
            h_init = np.zeros(n)
            for f, a, ph in zip(freqs2, amplitudes2, phases2):
                h_init += a * np.cos(f * np.pi * x + ph)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        elif restart == 3:
            # Random smooth function
            t = np.linspace(0, 2, n)
            h_init = 0.5 + 0.3 * np.sin(3*t + 0.5) + 0.2 * np.cos(7*t + 1.2) + 0.1 * np.sin(13*t + 2.1)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        else:
            # Perturbed best
            h_init = best_h_local + rng.normal(0, 0.1, n)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        
        # Optimize
        try:
            result = sp_minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, 1)] * n,
                options={'maxiter': max_iterations, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            h_opt = np.clip(result.x, 0, 1)
            # Enforce sum constraint more tightly
            s = np.sum(h_opt)
            if abs(s - n/2.0) > 1e-6:
                # Project
                for _ in range(100):
                    diff = n/2.0 - np.sum(h_opt)
                    if abs(diff) < 1e-12:
                        break
                    if diff > 0:
                        mask = h_opt < 1.0
                    else:
                        mask = h_opt > 0.0
                    if not np.any(mask):
                        break
                    h_opt[mask] += diff / np.sum(mask)
                    h_opt = np.clip(h_opt, 0, 1)
            
            obj_val = objective(h_opt)
            actual_overlap = _compute_overlap_max(h_opt)
            
            if actual_overlap < best_score:
                best_score = actual_overlap
                best_h = h_opt.copy()
                best_h_local = h_opt.copy()
        except Exception:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIME_LIMIT * 0.8:
            break
    
    # Final polish: local refinement with tighter tolerance
    elapsed = time.time() - start_time
    if elapsed < TOTAL_TIME_LIMIT * 0.9 and best_h is not None:
        try:
            # Fine-tune with smaller step
            h_polish = best_h.copy()
            for _ in range(50):
                if time.time() - start_time > TOTAL_TIME_LIMIT * 0.95:
                    break
                # Try small perturbations
                improved = False
                for trial in range(20):
                    h_try = h_polish + rng.normal(0, 0.01, n)
                    h_try = np.clip(h_try, 0, 1)
                    s = np.sum(h_try)
                    if abs(s - n/2.0) > 1e-4:
                        continue
                    ov = _compute_overlap_max(h_try)
                    if ov < best_score - 1e-10:
                        best_score = ov
                        best_h = h_try.copy()
                        h_polish = h_try.copy()
                        improved = True
                        break
        except Exception:
            pass
    
    # Final validation
    h_final = best_h
    s = np.sum(h_final)
    if abs(s - n/2.0) > 1e-6:
        # Force exact sum
        diff = n/2.0 - s
        mask = (h_final < 1.0) if diff > 0 else (h_final > 0.0)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0, 1)
    
    return h_final


# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 100
TOTAL_TIME_LIMIT = 110.0

def _compute_overlap_max(h, num_shifts=None):
    if num_shifts is None:
        num_shifts = N
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.rfft(h, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)[:n] * (2.0 / n)
    max_overlap = 1.0 - np.min(acf)
    return max_overlap

def _soft_max(vals, temperature=0.01):
    m = np.max(vals)
    return m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))

def construct():
    start_time = time.time()
    n = N
    x = np.arange(n) * (2.0 / n)
    
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = 0.15 / np.sqrt(len(freqs))
    phases = np.random.default_rng(42).uniform(0, 2*np.pi, len(freqs))
    
    h = np.zeros(n)
    for i, (f, a, ph) in enumerate(zip(freqs, amplitudes, phases)):
        h += a * np.cos(f * np.pi * x + ph)
    
    h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (n / 2.0) / current_sum
        h = np.clip(h, 0, 1)
        for _ in range(50):
            s = np.sum(h)
            if abs(s - n/2.0) < 1e-10:
                break
            diff = n/2.0 - s
            if diff > 0:
                mask = h < 1.0
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
            else:
                mask = h > 0.0
                if np.sum(mask) == 0:
                    break
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
    
    best_h = h.copy()
    best_score = _compute_overlap_max(best_h)
    
    elapsed = time.time() - start_time
    if elapsed > TOTAL_TIME_LIMIT * 0.3:
        return best_h
    
    def objective(params):
        h_p = np.clip(params, 0, 1)
        sum_penalty = 100.0 * (np.sum(h_p) - n/2.0) ** 2
        H = np.fft.rfft(h_p, n=2*n)
        acf = np.fft.irfft(H * np.conj(H), n=2*n)[:n] * (2.0 / n)
        overlap_vals = 1.0 - acf
        sm = _soft_max(overlap_vals, temperature=0.02)
        return sm + sum_penalty
    
    best_h_local = h.copy()
    
    rng = np.random.default_rng(123)
    max_iterations = 300
    
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIME_LIMIT * 0.7:
            break
        
        if restart == 0:
            h_init = h.copy()
        elif restart == 1:
            h_init = best_h_local + rng.normal(0, 0.05, n)
            h_init = np.clip(h_init, 0, 1)
        elif restart == 2:
            freqs2 = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
            amplitudes2 = 0.12 / np.sqrt(len(freqs2))
            phases2 = rng.uniform(0, 2*np.pi, len(freqs2))
            h_init = np.zeros(n)
            for f, a, ph in zip(freqs2, amplitudes2, phases2):
                h_init += a * np.cos(f * np.pi * x + ph)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        elif restart == 3:
            t = np.linspace(0, 2, n)
            h_init = 0.5 + 0.3 * np.sin(3*t + 0.5) + 0.2 * np.cos(7*t + 1.2) + 0.1 * np.sin(13*t + 2.1)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        else:
            h_init = best_h_local + rng.normal(0, 0.1, n)
            h_init = np.clip(h_init, 0, 1)
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (n/2.0) / s
                h_init = np.clip(h_init, 0, 1)
        
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, 1)] * n,
                options={'maxiter': max_iterations, 'ftol': 1e-12, 'gtol': 1e-8},
            )
            h_opt = np.clip(result.x, 0, 1)
            s = np.sum(h_opt)
            if abs(s - n/2.0) > 1e-6:
                for _ in range(100):
                    diff = n/2.0 - np.sum(h_opt)
                    if abs(diff) < 1e-12:
                        break
                    if diff > 0:
                        mask = h_opt < 1.0
                    else:
                        mask = h_opt > 0.0
                    if not np.any(mask):
                        break
                    h_opt[mask] += diff / np.sum(mask)
                    h_opt = np.clip(h_opt, 0, 1)
            
            actual_overlap = _compute_overlap_max(h_opt)
            
            if actual_overlap < best_score:
                best_score = actual_overlap
                best_h = h_opt.copy()
                best_h_local = h_opt.copy()
        except Exception:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > TOTAL_TIME_LIMIT * 0.8:
            break
    
    elapsed = time.time() - start_time
    if elapsed < TOTAL_TIME_LIMIT * 0.9 and best_h is not None:
        try:
            h_polish = best_h.copy()
            for _ in range(50):
                if time.time() - start_time > TOTAL_TIME_LIMIT * 0.95:
                    break
                for trial in range(20):
                    h_try = h_polish + rng.normal(0, 0.01, n)
                    h_try = np.clip(h_try, 0, 1)
                    s = np.sum(h_try)
                    if abs(s - n/2.0) > 1e-4:
                        continue
                    ov = _compute_overlap_max(h_try)
                    if ov < best_score - 1e-10:
                        best_score = ov
                        best_h = h_try.copy()
                        h_polish = h_try.copy()
                        break
        except Exception:
            pass
    
    h_final = best_h
    s = np.sum(h_final)
    if abs(s - n/2.0) > 1e-6:
        diff = n/2.0 - s
        mask = (h_final < 1.0) if diff > 0 else (h_final > 0.0)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0, 1)
    
    return h_final
# EVOLVE-BLOCK-END
