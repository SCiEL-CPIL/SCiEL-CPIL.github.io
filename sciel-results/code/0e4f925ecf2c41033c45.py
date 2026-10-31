import numpy as np
import time

N = 400

def _psl_of_power_spectrum(x):
    """
    Compute the Peak Sidelobe Level (PSL) of the power spectrum of x.
    PSL = max(sidelobes) / max(main_lobe)
    """
    X = np.fft.fft(x)
    P = np.abs(X)**2
    
    # The peak is at DC (index 0) for a symmetric, positive window
    peak_val = P[0]
    
    # Find the end of the main lobe.
    # The main lobe ends at the first local minimum after DC.
    # We look at the first half of the spectrum (0 to N/2) due to symmetry.
    half_P = P[:N//2]
    
    # Find local minima
    # A local minimum at i is if P[i] < P[i-1] and P[i] < P[i+1]
    # We start searching from index 1.
    min_idx = -1
    for i in range(1, len(half_P) - 1):
        if half_P[i] < half_P[i-1] and half_P[i] < half_P[i+1]:
            min_idx = i
            break
            
    if min_idx == -1:
        # If no local minimum found, the whole thing is the main lobe?
        # This shouldn't happen for a windowed function.
        # Fallback: exclude first 10% of bins
        min_idx = int(0.1 * len(half_P))
        
    # The sidelobes are from min_idx to the end of the half spectrum
    # Note: The main lobe includes the peak and the skirt down to the first null.
    # So sidelobes start after the first null.
    sidelobe_region = half_P[min_idx+1:]
    
    if len(sidelobe_region) == 0:
        return 0.0
        
    max_sidelobe = np.max(sidelobe_region)
    
    if peak_val == 0:
        return 0.0
        
    return max_sidelobe / peak_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_beta = 8.0
    best_psl = float('inf')
    best_window = np.kaiser(N, best_beta)
    
    # Search for optimal beta
    # Kaiser beta controls the sidelobe roll-off.
    # Higher beta -> lower sidelobes, wider main lobe.
    # We want to minimize PSL.
    
    # Try a coarse grid first
    betas = np.linspace(1.0, 20.0, 50)
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        w = np.kaiser(N, beta)
        psl = _psl_of_power_spectrum(w)
        
        if psl < best_psl:
            best_psl = psl
            best_beta = beta
            best_window = w
            
    # Refine around the best beta
    if time.time() < deadline:
        low = max(1.0, best_beta - 1.0)
        high = min(20.0, best_beta + 1.0)
        fine_betas = np.linspace(low, high, 50)
        
        for beta in fine_betas:
            if time.time() > deadline:
                break
            w = np.kaiser(N, beta)
            psl = _psl_of_power_spectrum(w)
            if psl < best_psl:
                best_psl = psl
                best_beta = beta
                best_window = w
                
    return best_window
