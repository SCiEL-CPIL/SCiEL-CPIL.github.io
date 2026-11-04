import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Pre-compute FFT size for linear autoconvolution
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]

def _metric(x):
    """
    Compute the score: max(|autoconv|) / (sum(x))^2
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_lin(x)
    max_abs_conv = np.max(np.abs(conv))
    return max_abs_conv / (s * s)

def _phase_to_signal(phases, magnitudes):
    """
    Reconstruct signal from phases and fixed magnitudes.
    phases: array of length N//2 + 1 (for rfft)
    magnitudes: array of length N//2 + 1
    """
    # Construct complex spectrum
    # rfft output has DC (index 0) and Nyquist (index N//2) as real.
    # We treat phases for all, but force real values for DC and Nyquist if N is even.
    # To keep it simple and robust, we optimize phases for the complex part.
    
    spectrum = np.zeros(N // 2 + 1, dtype=complex)
    
    # DC component (index 0)
    # We can fix its phase to 0 (or optimize separately, but usually 0 is fine for symmetry)
    spectrum[0] = magnitudes[0] * np.exp(1j * phases[0])
    
    # Middle components
    for k in range(1, N // 2):
        spectrum[k] = magnitudes[k] * np.exp(1j * phases[k])
        
    # Nyquist component (index N//2)
    spectrum[N // 2] = magnitudes[N // 2] * np.exp(1j * phases[N // 2])
    
    # Inverse FFT
    x = np.fft.irfft(spectrum, _FFT_SIZE)
    # Ensure we only take the first N samples? 
    # Wait, the signal x is defined in time domain of length N.
    # The autoconvolution is linear.
    # If we define x via its DFT, the length of x is N.
    # irfft with n=_FFT_SIZE gives a circular convolution context if not careful.
    # But we are using rfft/irfft with padding for LINEAR convolution.
    # So x should be the first N samples of the inverse transform?
    # Actually, if we construct x from a spectrum of length N, 
    # np.fft.irfft(spectrum, n=N) gives x.
    # Let's stick to n=N for the signal definition.
    
    x = np.fft.irfft(spectrum, n=N)
    return x

def _construct_magnitudes(strategy='flat'):
    """
    Define fixed magnitudes for the frequency components.
    'flat': White noise power spectrum.
    'lowpass': Low-pass filter.
    """
    mags = np.ones(N // 2 + 1)
    if strategy == 'flat':
        # Normalize so that energy is roughly 1
        pass
    elif strategy == 'lowpass':
        # Hanning window in frequency domain
        w = np.hanning(N // 2 + 1)
        mags = mags * w
        # Ensure DC is not zero
        mags[0] = 1.0
        mags[-1] = 1.0 if N % 2 == 0 else 0.0 # Nyquist
        
    # Normalize total energy to 1
    # Energy in time domain = sum(|x|^2) = (1/N) * sum(|X|^2) for DFT
    # We want a specific scale, but the metric is scale invariant (numerator scales with s^2, denominator with s^2)
    # Wait, metric is max(|conv|) / s^2.
    # conv scales with s^2 if x scales with s.
    # So metric is invariant to scaling of x.
    # Thus, absolute magnitude level doesn't matter, only relative magnitudes and phases.
    
    return mags

def _objective_phases(phases, magnitudes, deadline):
    """
    Objective function for Nelder-Mead: minimize max(|autoconv|) / sum(x)^2
    """
    if time.time() > deadline:
        return 1e10
    x = _phase_to_signal(phases, magnitudes)
    return _metric(x)

def _optimize_phases(magnitudes, initial_phases, deadline, rng):
    """
    Run Nelder-Mead on the phases.
    """
    n_params = len(phases)
    
    # Initial simplex setup
    x0 = initial_phases.copy()
    
    # Use Nelder-Mead with tight tolerances
    # Nelder-Mead is derivative-free and handles non-smoothness better than L-BFGS for max functions.
    
    # We need to be careful with time.
    # Nelder-Mead can be slow for high dimensions (200 params).
    # Strategy: Use a coarse optimization first, then refine.
    
    # Alternative: Coordinate descent on phases?
    # Let's try a hybrid: 
    # 1. Random phases
    # 2. Local search: For each phase, rotate it to minimize the global peak.
    
    # Actually, Nelder-Mead on 200 variables might be too slow or get stuck.
    # Let's use a "Peak Suppression" heuristic which is faster.
    
    best_phases = initial_phases.copy()
    best_val = _metric(_phase_to_signal(best_phases, magnitudes))
    
    # Iterative peak suppression
    # Identify the lag k with max |conv[k]|.
    # The contribution to conv[k] from frequency component m is related to x.
    # Changing phase of component m rotates the vector contribution.
    # We can optimize the phase of each component independently to minimize the peak.
    
    # Let's do a few rounds of coordinate descent on phases.
    num_rounds = 50
    for r in range(num_rounds):
        if time.time() > deadline:
            break
        improved = False
        for k in range(N // 2 + 1):
            if time.time() > deadline:
                break
            
            # Optimize phase[k]
            # We want to find phase_p such that metric is minimized.
            # Use a simple 1D optimizer or ternary search on [0, 2pi).
            # Since the function is periodic and potentially multi-modal, 
            # we sample a few points or use a local search.
            
            # Let's try 8-point search around current phase
            best_local_val = best_val
            best_local_phase = best_phases[k]
            
            # Current value
            current_val = best_val # We don't have it exactly, but best_val is global.
            # Let's recompute current val for this specific k change? 
            # No, just search.
            
            # Sample 8 points in [-0.5, 0.5] rad around current
            deltas = np.linspace(-0.5, 0.5, 8)
            for d in deltas:
                new_phases = best_phases.copy()
                new_phases[k] = best_phases[k] + d
                x = _phase_to_signal(new_phases, magnitudes)
                val = _metric(x)
                if val < best_local_val:
                    best_local_val = val
                    best_local_phase = new_phases[k]
            
            if best_local_val < best_val:
                best_phases[k] = best_local_phase
                best_val = best_local_val
                improved = True
        
        if not improved:
            break
            
    return best_phases, best_val

def construct():
    deadline = time.time() + 145.0
    
    # 1. Define Magnitudes
    # Flat spectrum is a good start for "white noise" like properties.
    mags = _construct_magnitudes('flat')
    
    # 2. Initial Phases
    rng = np.random.default_rng(2024)
    
    best_overall_phases = None
    best_overall_val = float('inf')
    
    # Multiple starts to avoid local minima
    num_starts = 10
    time_per_start = (deadline - time.time()) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random phases
        phases = rng.uniform(0, 2 * np.pi, size=N // 2 + 1)
        
        # Ensure DC and Nyquist are real? 
        # In rfft, X[0] and X[N//2] are real. 
        # If we allow arbitrary phases, irfft will produce a complex signal? 
        # No, irfft assumes Hermitian symmetry. 
        # If we input arbitrary complex numbers to irfft, it treats them as the positive half 
        # and mirrors them. The result is real. 
        # However, the DC and Nyquist components *must* be real for the symmetry to hold perfectly 
        # if we want a real signal? 
        # Actually, np.fft.irfft just takes the first N//2+1 coeffs. 
        # It implicitly assumes the rest are conjugates. 
        # If X[0] is complex, the result x will be complex? 
        # No, irfft always returns a real array. 
        # It effectively does: x[n] = 1/N sum_k X[k] exp(2j pi kn/N) ... 
        # If X[0] is complex, the "mirror" X[N] would be conjugate? 
        # The standard irfft implementation assumes X[0] and X[N//2] are real. 
        # If they are not, the result might be unexpected or it might just take the real part?
        # Documentation: "The input is assumed to be in the form returned by rfft."
        # rfft returns real for DC and Nyquist. 
        # So we should force phases[0] and phases[N//2] to be 0 or pi (to keep them real).
        # Let's force them to 0 for simplicity.
        phases[0] = 0.0
        if N % 2 == 0:
            phases[N // 2] = 0.0
            
        # Optimize
        opt_phases, val = _optimize_phases(mags, phases, deadline - 10, rng)
        
        if val < best_overall_val:
            best_overall_val = val
            best_overall_phases = opt_phases.copy()
            
    # Final Polish: Coordinate descent with finer steps
    if best_overall_phases is not None and time.time() < deadline - 5:
        current_phases = best_overall_phases.copy()
        current_val = best_overall_val
        
        # Finer coordinate descent
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            improved = False
            for k in range(1, N // 2): # Skip DC and Nyquist
                if time.time() > deadline - 2:
                    break
                # Ternary search or fine grid
                # Let's do 5 points in [-0.1, 0.1]
                deltas = np.linspace(-0.1, 0.1, 5)
                best_local_val = current_val
                best_local_phase = current_phases[k]
                
                for d in deltas:
                    new_phases = current_phases.copy()
                    new_phases[k] = current_phases[k] + d
                    x = _phase_to_signal(new_phases, mags)
                    val = _metric(x)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_phase = new_phases[k]
                        
                if best_local_val < current_val:
                    current_phases[k] = best_local_phase
                    current_val = best_local_val
                    improved = True
            if not improved:
                break
        
        if current_val < best_overall_val:
            best_overall_val = current_val
            best_overall_phases = current_phases

    # Convert best phases back to time domain signal
    if best_overall_phases is not None:
        x_final = _phase_to_signal(best_overall_phases, mags)
    else:
        # Fallback
        x_final = np.ones(N)
        
    return x_final

