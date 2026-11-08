"""Combined approach: FFT-based overlap + Fourier-mode parameterization + smooth optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # seconds, with 10s safety margin before 150s limit

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h over all circular shifts.
    
    h is an array of N values on [0, 2], so dx = 2/N.
    sum(h) * dx = 1, so sum(h) = N/2.
    
    overlap(k) = sum_j h[j] * (1 - h[(j+k)%N]) * dx
               = (sum_j h[j] - sum_j h[j]*h[(j+k)%N]) * dx
               = (N/2 - C[k]) * dx
    where C[k] = sum_j h[j]*h[(j+k)%N]
    """
    n = len(h)
    dx = 2.0 / n
    sum_h = np.sum(h)
    
    # Compute circular cross-correlation C[k] = sum_j h[j]*h[(j+k)%n]
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    
    overlaps = (sum_h - C) * dx
    return np.max(overlaps)

def _soft_max(values, temperature=0.01):
    """Soft maximum using log-sum-exp for smooth optimization."""
    shifted = values - np.max(values)
    log_sum = np.log(np.sum(np.exp(shifted / temperature))) * temperature
    return np.max(values) + log_sum

def _build_h_from_fourier(coeffs, enforce_bounds=True):
    """
    Build h from Fourier coefficients.
    coeffs = [a_1, b_1, a_2, b_2, ..., a_m, b_m]
    h(x) = 0.5 + sum_{k=1}^{m} (a_k * cos(pi*k*x) + b_k * sin(pi*k*x))
    where x = 2*j/N for j=0,...,N-1
    """
    x = np.linspace(0, 2, N, endpoint=False)
    h = np.full(N, 0.5)
    n_modes = len(coeffs) // 2
    for k in range(1, n_modes + 1):
        a_k = coeffs[2*(k-1)]
        b_k = coeffs[2*(k-1)+1]
        h += a_k * np.cos(np.pi * k * x) + b_k * np.sin(np.pi * k * x)
    
    if enforce_bounds:
        # Use sigmoid-like soft clipping to keep values in [0,1] smoothly
        # h_soft = 1/(1+exp(-alpha*(h - 0.5))) mapped to [0,1]
        alpha = 20.0
        h_soft = 1.0 / (1.0 + np.exp(-alpha * (h - 0.5)))
        # Scale to have mean 0.5
        mean_h = np.mean(h_soft)
        if mean_h > 1e-10:
            h_soft = h_soft * (0.5 / mean_h)
        h = np.clip(h_soft, 0.0, 1.0)
    else:
        h = np.clip(h, 0.0, 1.0)
    
    # Normalize to sum = N/2 with exact correction
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        scale = target_sum / current_sum
        h = h * scale
        h = np.clip(h, 0.0, 1.0)
        # Final exact correction: adjust one element to make sum exactly target
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            # Find element with most room to adjust
            if diff > 0:
                idx = np.argmin(1.0 - h)  # closest to 1, has most room to increase
            else:
                idx = np.argmax(h)  # closest to 0, has most room to decrease
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0.0, 1.0)
            # If clipping caused new error, do one more pass
            diff2 = target_sum - h.sum()
            if abs(diff2) > 1e-12:
                # Distribute remaining error to elements not at bounds
                mask = (h > 1e-8) & (h < 1.0 - 1e-8)
                if np.sum(mask) > 0:
                    h[mask] += diff2 / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_fourier_smooth(coeffs, temperature=0.01):
    """Smooth objective using soft-max for the overlap."""
    h = _build_h_from_fourier(coeffs, enforce_bounds=True)
    n = len(h)
    dx = 2.0 / n
    sum_h = np.sum(h)
    
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    C = np.zeros(n)
    C[0] = autocorr[0]
    if n > 1:
        C[1:] = autocorr[n-1:0:-1]
    
    overlaps = (sum_h - C) * dx
    return _soft_max(overlaps, temperature)

def _objective_fourier(coeffs):
    """Objective function: max overlap from Fourier coefficients."""
    h = _build_h_from_fourier(coeffs, enforce_bounds=True)
    return _compute_max_overlap(h)

def _try_candidate(h):
    """Try a candidate h, return (h, score) if valid and better, else (None, inf)."""
    h = np.clip(h, 0.0, 1.0)
    target_sum = N / 2.0
    current_sum = h.sum()
    if current_sum > 1e-10:
        scale = target_sum / current_sum
        h = h * scale
        h = np.clip(h, 0.0, 1.0)
        # Exact sum correction
        diff = target_sum - h.sum()
        if abs(diff) > 1e-12:
            mask = (h > 1e-8) & (h < 1.0 - 1e-8)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    score = _compute_max_overlap(h)
    return h, score

def construct():
    start_time = time.time()
    
    # Baseline: uniform 0.5
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)  # should be 0.25
    
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Try analytic candidates first
    candidates = []
    
    # h = 0.5 + 0.5*sin(pi*x)
    h1 = 0.5 + 0.5 * np.sin(np.pi * x)
    candidates.append(h1)
    
    # h = 0.5 + 0.5*cos(pi*x)
    h2 = 0.5 + 0.5 * np.cos(np.pi * x)
    candidates.append(h2)
    
    # Square wave: 1 on [0,1), 0 on [1,2)
    h3 = (x < 1.0).astype(float)
    candidates.append(h3)
    
    # h = 0.5 + 0.3*sin(2*pi*x)
    h4 = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    candidates.append(h4)
    
    # h = 0.5 + 0.4*cos(2*pi*x)
    h5 = 0.5 + 0.4 * np.cos(2 * np.pi * x)
    candidates.append(h5)
    
    # h = 0.5 + 0.3*sin(3*pi*x)
    h6 = 0.5 + 0.3 * np.sin(3 * np.pi * x)
    candidates.append(h6)
    
    # h = 0.5 + 0.25*sin(4*pi*x)
    h7 = 0.5 + 0.25 * np.sin(4 * np.pi * x)
    candidates.append(h7)
    
    # h = 0.5 + 0.2*cos(5*pi*x)
    h8 = 0.5 + 0.2 * np.cos(5 * np.pi * x)
    candidates.append(h8)
    
    for h_cand in candidates:
        h_new, score = _try_candidate(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_new
    
    # Multi-start optimization with Fourier coefficients
    n_modes = 8  # Use 8 modes for more flexibility
    n_coeffs = 2 * n_modes
    bounds = [(-0.4, 0.4)] * n_coeffs
    
    rng = np.random.default_rng(42)
    
    n_starts = 40
    for start_idx in range(n_starts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Random initial point
        if start_idx == 0:
            x0 = np.zeros(n_coeffs)
        elif start_idx == 1:
            # Start from sine wave
            x0 = np.zeros(n_coeffs)
            x0[1] = 0.5  # b_1 for sin(pi*x)
        elif start_idx == 2:
            # Start from cosine wave
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.5  # a_1 for cos(pi*x)
        elif start_idx == 3:
            # Start from mixed
            x0 = np.zeros(n_coeffs)
            x0[0] = 0.3
            x0[1] = 0.3
        elif start_idx == 4:
            # Start from higher frequency
            x0 = np.zeros(n_coeffs)
            x0[4] = 0.4  # a_2
        else:
            x0 = rng.uniform(-0.3, 0.3, n_coeffs)
        
        try:
            # Use smooth objective for better gradient behavior
            result = minimize(
                lambda c: _objective_fourier_smooth(c, temperature=0.01), x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_new = _build_h_from_fourier(result.x, enforce_bounds=True)
            score = _compute_max_overlap(h_new)
            
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Polish phase: use exact (non-smooth) objective with tight tolerances
    if time.time() - start_time < DEADLINE - 10:
        try:
            # Get Fourier coefficients from best_h
            n_modes_refine = min(6, n_modes)
            A = np.zeros((N, 2 * n_modes_refine))
            for k in range(1, n_modes_refine + 1):
                A[:, 2*(k-1)] = np.cos(np.pi * k * x)
                A[:, 2*(k-1)+1] = np.sin(np.pi * k * x)
            target = best_h - 0.5
            coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            # Refine with smooth objective first
            result_smooth = minimize(
                lambda c: _objective_fourier_smooth(c, temperature=0.005), coeffs,
                method='L-BFGS-B',
                bounds=[(-0.4, 0.4)] * (2 * n_modes_refine),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            h_smooth = _build_h_from_fourier(result_smooth.x, enforce_bounds=True)
            score_smooth = _compute_max_overlap(h_smooth)
            
            if score_smooth < best_score:
                best_score = score_smooth
                best_h = h_smooth.copy()
                coeffs = result_smooth.x
            else:
                coeffs = result_smooth.x
            
            # Then refine with exact objective
            result_exact = minimize(
                _objective_fourier, coeffs,
                method='L-BFGS-B',
                bounds=[(-0.4, 0.4)] * (2 * n_modes_refine),
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            
            h_exact = _build_h_from_fourier(result_exact.x, enforce_bounds=True)
            score_exact = _compute_max_overlap(h_exact)
            
            if score_exact < best_score:
                best_score = score_exact
                best_h = h_exact.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish if time permits
    if time.time() - start_time < DEADLINE - 5:
        try:
            n_modes_nm = min(4, n_modes)
            A = np.zeros((N, 2 * n_modes_nm))
            for k in range(1, n_modes_nm + 1):
                A[:, 2*(k-1)] = np.cos(np.pi * k * x)
                A[:, 2*(k-1)+1] = np.sin(np.pi * k * x)
            target = best_h - 0.5
            coeffs_nm, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
            
            result_nm = minimize(
                _objective_fourier, coeffs_nm,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            h_nm = _build_h_from_fourier(result_nm.x, enforce_bounds=True)
            score_nm = _compute_max_overlap(h_nm)
            
            if score_nm < best_score:
                best_score = score_nm
                best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final validation: ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    target_sum = N / 2.0
    diff = target_sum - best_h.sum()
    if abs(diff) > 1e-10:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final check
    if abs(np.sum(best_h) - N/2) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h
