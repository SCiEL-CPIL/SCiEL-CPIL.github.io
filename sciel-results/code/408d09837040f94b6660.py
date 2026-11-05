"""Sparse-spectrum IDFT approach for the third autocorrelation inequality."""
N = 400

import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0


def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _h_from_spectrum(spectrum):
    """
    Build h from a sparse complex spectrum.
    spectrum: complex array of length N (most entries zero).
    h = real part of IDFT(spectrum) scaled to have positive mean.
    """
    h = np.real(np.fft.ifft(spectrum)) * N  # ifft includes 1/N, multiply back
    # Ensure sum is positive and non-zero
    s = np.sum(h)
    if abs(s) < 1e-10:
        # Add a small DC component
        h += 0.01
        s = np.sum(h)
    if s < 0:
        h = -h
    return h


def _spectrum_objective(params, n_bins, bin_indices):
    """
    Objective: minimize task value given complex amplitudes at selected bins.
    params: real array of length 2*n_bins = [re1, im1, re2, im2, ...]
    """
    spectrum = np.zeros(N, dtype=complex)
    for i, k in enumerate(bin_indices):
        spectrum[k] = params[2*i] + 1j * params[2*i+1]
    
    h = _h_from_spectrum(spectrum)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _optimize_spectrum(bin_indices, init_params, deadline):
    """Optimize complex amplitudes for given bin selection."""
    n_bins = len(bin_indices)
    n_params = 2 * n_bins
    
    # Use L-BFGS-B with analytical structure awareness
    try:
        res = minimize(
            _spectrum_objective,
            init_params,
            args=(n_bins, bin_indices),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        best_params = res.x
        best_val = _spectrum_objective(best_params, n_bins, bin_indices)
        return best_params, best_val
    except Exception:
        return init_params, _spectrum_objective(init_params, n_bins, bin_indices)


def _local_refine_spectrum(bin_indices, params, deadline, max_iter=5000):
    """Local refinement in spectrum space."""
    best_val = _spectrum_objective(params, len(bin_indices), bin_indices)
    best_params = params.copy()
    rng = np.random.default_rng(77)
    
    step_size = 0.5
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, min(4, len(bin_indices) + 1))
        idx = rng.choice(2 * len(bin_indices), size=n_perturb, replace=False)
        p_new = best_params.copy()
        p_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _spectrum_objective(p_new, len(bin_indices), bin_indices)
        if np.isfinite(v_new) and v_new < best_val - 1e-12:
            best_val = v_new
            best_params = p_new
            step_size *= 0.97
        elif step % 500 == 0:
            step_size *= 0.998
    
    return best_params, best_val


def construct():
    """
    Construct step heights using sparse-spectrum IDFT representation.
    The autocorrelation in the time domain corresponds to the squared
    magnitude of the DFT in the frequency domain. By controlling a small
    number of spectral bins, we can shape the autocorrelation sidelobes
    more directly than in the time domain.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(2024)
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Key insight: For a signal with DFT H[k], the autoconvolution's DFT is |H[k]|^2.
    # The peak of autoconvolution at lag 0 is sum(h^2) = (1/N)*sum|H[k]|^2.
    # Other lags' autoconvolution values relate to interference between bins.
    # If we use only 2-3 bins, the autoconvolution becomes a sum of a few cosines,
    # and we can minimize the max sidelobe relative to the peak.
    
    # Strategy: Try different numbers of spectral bins (2 to 8) and different
    # bin selections, optimizing the complex amplitudes.
    
    # Always include DC (bin 0) to ensure non-zero sum, or handle via offset.
    # Actually, let's allow any bins but ensure the resulting h has non-zero sum.
    
    n = np.arange(N)
    
    # Phase 1: Systematic bin selection with optimization
    # Try small numbers of bins for clean spectral structure
    for n_bins in range(2, 9):
        if time.time() > deadline - 20:
            break
        
        # Generate multiple random bin selections
        n_trials = max(5, 20 - n_bins * 2)
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                break
            
            # Select bins: always include bin 0 for DC, plus random others
            # Use symmetric pairs (k and N-k) for real-valued h
            available = list(range(1, N // 2 + 1))
            rng.shuffle(available)
            
            # Pick n_bins//2 pairs + maybe one unpaired
            n_pairs = n_bins // 2
            selected_k = available[:n_pairs]
            
            bin_indices = [0]  # DC
            for k in selected_k:
                bin_indices.append(k)
                if k != N - k:
                    bin_indices.append(N - k)
            
            # Trim to exactly n_bins if needed (keep DC + first n_bins-1)
            bin_indices = bin_indices[:n_bins]
            
            # Initialize: DC gets a positive real value, others random
            init_params = np.zeros(2 * len(bin_indices))
            init_params[0] = 1.0  # DC real part
            for i in range(1, len(bin_indices)):
                init_params[2*i] = rng.uniform(-1.0, 1.0)
                init_params[2*i+1] = rng.uniform(-1.0, 1.0)
            
            # Optimize
            opt_params, opt_val = _optimize_spectrum(bin_indices, init_params, deadline - 10)
            
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                spectrum = np.zeros(N, dtype=complex)
                for i, k in enumerate(bin_indices):
                    spectrum[k] = opt_params[2*i] + 1j * opt_params[2*i+1]
                best_h = _h_from_spectrum(spectrum)
    
    # Phase 2: Structured spectral patterns
    # Pattern A: Two-bin (single frequency) - autoconvolution is a pure cosine
    if time.time() < deadline - 20:
        for f in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50]:
            if time.time() > deadline - 15:
                break
            bin_indices = [0, f, N - f]
            # h[n] = a0 + a1*cos(2*pi*f*n/N) + b1*sin(2*pi*f*n/N)
            # Optimize a0, a1, b1
            init_params = np.array([1.0, 0.5, 0.0])
            opt_params, opt_val = _optimize_spectrum(bin_indices, init_params, deadline - 10)
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                spectrum = np.zeros(N, dtype=complex)
                for i, k in enumerate(bin_indices):
                    spectrum[k] = opt_params[2*i] + 1j * opt_params[2*i+1]
                best_h = _h_from_spectrum(spectrum)
    
    # Pattern B: Three-bin (two frequencies) - autoconvolution has two cosine terms
    if time.time() < deadline - 20:
        for (f1, f2) in [(1, 2), (1, 3), (2, 3), (1, 5), (2, 5), (3, 5),
                         (1, 7), (2, 7), (3, 7), (5, 7), (1, 10), (3, 10)]:
            if time.time() > deadline - 15:
                break
            bin_indices = [0, f1, N - f1, f2, N - f2]
            init_params = np.array([1.0, 0.5, 0.0, 0.3, 0.0])
            opt_params, opt_val = _optimize_spectrum(bin_indices, init_params, deadline - 10)
            if np.isfinite(opt_val) and opt_val < best_val:
                best_val = opt_val
                spectrum = np.zeros(N, dtype=complex)
                for i, k in enumerate(bin_indices):
                    spectrum[k] = opt_params[2*i] + 1j * opt_params[2*i+1]
                best_h = _h_from_spectrum(spectrum)
    
    # Phase 3: Random dense-ish spectra with few non-zero bins
    if time.time() < deadline - 25:
        for n_active in [3, 4, 5, 6, 8, 10]:
            if time.time() > deadline - 20:
                break
            
            n_trials = max(3, 10 - n_active // 2)
            for trial in range(n_trials):
                if time.time() > deadline - 15:
                    break
                
                # Random bins (not necessarily symmetric - h will be complex, take real)
                bins = rng.choice(range(1, N), size=n_active, replace=False)
                bin_indices = sorted(set([0] + list(bins)))
                
                init_params = np.zeros(2 * len(bin_indices))
                init_params[0] = 1.0
                for i in range(1, len(bin_indices)):
                    init_params[2*i] = rng.uniform(-0.5, 0.5)
                    init_params[2*i+1] = rng.uniform(-0.5, 0.5)
                
                opt_params, opt_val = _optimize_spectrum(bin_indices, init_params, deadline - 10)
                if np.isfinite(opt_val) and opt_val < best_val:
                    best_val = opt_val
                    spectrum = np.zeros(N, dtype=complex)
                    for i, k in enumerate(bin_indices):
                        spectrum[k] = opt_params[2*i] + 1j * opt_params[2*i+1]
                    best_h = _h_from_spectrum(spectrum)
    
    # Phase 4: Local refinement of best solution in spectrum space
    if time.time() < deadline - 10 and np.isfinite(best_val) and best_val < 2.0:
        # Reconstruct spectrum from best_h
        H = np.fft.fft(best_h)
        # Find the most significant bins
        mag = np.abs(H)
        top_k = np.argsort(mag)[::-1][:12]
        # Include DC
        if 0 not in top_k:
            top_k = np.append(top_k, 0)
        bin_indices = sorted(top_k.tolist())
        
        # Initialize params from current spectrum
        init_params = np.zeros(2 * len(bin_indices))
        for i, k in enumerate(bin_indices):
            init_params[2*i] = np.real(H[k])
            init_params[2*i+1] = np.imag(H[k])
        
        # Refine in spectrum space
        opt_params, opt_val = _local_refine_spectrum(bin_indices, init_params, deadline - 5)
        if np.isfinite(opt_val) and opt_val < best_val:
            best_val = opt_val
            spectrum = np.zeros(N, dtype=complex)
            for i, k in enumerate(bin_indices):
                spectrum[k] = opt_params[2*i] + 1j * opt_params[2*i+1]
            best_h = _h_from_spectrum(spectrum)
    
    # Phase 5: Final time-domain local refinement
    if time.time() < deadline - 5:
        best_val_final = _evaluate(best_h)
        if np.isfinite(best_val_final) and best_val_final < best_val:
            best_val = best_val_final
        
        # Quick time-domain refinement
        rng2 = np.random.default_rng(99)
        step_size = 0.05
        for step in range(5000):
            if time.time() > deadline - 2:
                break
            n_perturb = rng2.integers(1, 6)
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_new = best_h.copy()
            h_new[idx] += rng2.normal(0, step_size, n_perturb)
            v_new = _evaluate(h_new)
            if np.isfinite(v_new) and v_new < best_val - 1e-12:
                best_val = v_new
                best_h = h_new
                step_size *= 0.98
    
    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
