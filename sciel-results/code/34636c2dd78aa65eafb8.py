"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac[:size]))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    rng = random.Random(42)
    
    # Strategy: Spectral shaping via iterative phase adjustment.
    # Instead of random flips, we treat the DFT magnitudes as a target to minimize
    # the peak of the squared magnitude (which corresponds to max autoconv).
    # We use a soft thresholding approach on the DFT coefficients to reduce peaks.
    
    # Initialize with a random phase signal
    h = np.ones(N)
    for _ in range(5):
        h = np.array([1.0 if rng.random() > 0.5 else -1.0 for _ in range(N)])
        if np.sum(h) == 0:
            h[0] = 1.0
        
        H = np.fft.rfft(h, fsize)
        
        # Iterate to reduce spectral peaks
        for it in range(100):
            if time.time() > deadline - 20:
                break
                
            # Compute autoconvolution
            ac = np.fft.irfft(H * H, fsize)
            ac_valid = ac[:size]
            max_val = np.max(np.abs(ac_valid))
            s = np.sum(h)
            if abs(s) < 1e-9:
                continue
            current_score = max_val / (s * s)
            
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()
            
            # Find the peak location in time domain
            peak_idx = np.argmax(np.abs(ac_valid))
            
            # To reduce the peak at peak_idx, we can adjust the phases of H.
            # The value at peak_idx is sum_k H[k] H[-k] e^{i k omega t}
            # This is complex. A simpler heuristic:
            # If the peak is positive, we want to reduce the constructive interference.
            # If negative, reduce destructive? No, we want to reduce magnitude.
            
            # Let's try a different spectral approach:
            # Minimize sum |H[k]|^4 (which is related to the energy of the autoconvolution)
            # This is equivalent to minimizing the L4 norm of H.
            # We can do gradient descent on the phases of H.
            
            # Calculate gradient of sum |H|^4 with respect to phases.
            # Let H[k] = |H[k]| e^{i phi_k}.
            # d/dphi_k sum |H|^4 = 4 |H[k]|^2 Im(H[k] conj(H[-k])) ? No.
            
            # Let's stick to a simpler, robust method: 
            # "Time-domain peak shaving" in frequency domain.
            # If ac[peak_idx] is the max, we can try to flip bits that contribute most to it.
            # Contribution of bit i to ac[peak_idx] is sum_j h[j] h[j + peak_idx - i] ... this is getting complicated.
            
            # Alternative: Use the fact that we want to minimize the peak of |ac|.
            # Let's try to flatten the spectrum.
            # Target: |H[k]|^2 = constant.
            # This minimizes the peak of the autoconvolution in a statistical sense.
            
            # Update H to be closer to a flat spectrum while maintaining the constraint that h is binary?
            # This is hard.
            
            # Let's try a simple heuristic:
            # Flip the bit that causes the largest increase in the score if flipped?
            # No, that's local search.
            
            # Let's try "Chirp" initialization and refinement.
            # Chirps have good autocorrelation properties (Barker sequences are related).
            break # Disable this inner loop for now, switch to pure local search with better init
        
        # Refine with local search
        h_local = h.copy()
        H_local = np.fft.rfft(h_local, fsize)
        k_indices = np.arange(fsize // 2 + 1)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        
        current_val = _score(h_local)
        
        # Greedy Hill Climb with random restarts on the order
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Randomize the order of checks to avoid bias
            indices = list(range(N))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                old_val = h_local[i]
                new_val = -old_val
                h_local[i] = new_val
                delta = -2 * old_val
                H_local += delta * twiddles[i]
                
                ac = np.fft.irfft(H_local * H_local, fsize)
                new_ac_max = np.max(np.abs(ac[:size]))
                s = np.sum(h_local)
                if abs(s) < 1e-9:
                    new_score = 1e10
                else:
                    new_score = new_ac_max / (s * s)
                
                if new_score < current_val:
                    current_val = new_score
                    if current_val < best_val:
                        best_val = current_val
                        best_h = h_local.copy()
                    improved = True
                else:
                    h_local[i] = old_val
                    H_local -= delta * twiddles[i]
                    
    # Simulated Annealing for final polish
    if time.time() < deadline - 10:
        h = best_h.copy()
        H = np.fft.rfft(h, fsize)
        k_indices = np.arange(fsize // 2 + 1)
        twiddles = np.exp(-1j * 2 * np.pi * np.outer(np.arange(N), k_indices) / fsize)
        
        current_val = _score(h)
        T = 0.5
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            i = rng.randint(0, N - 1)
            old_val = h[i]
            new_val = -old_val
            h[i] = new_val
            delta = -2 * old_val
            H += delta * twiddles[i]
            
            ac = np.fft.irfft(H * H, fsize)
            new_ac_max = np.max(np.abs(ac[:size]))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = new_ac_max / (s * s)
            
            delta_score = new_score - current_val
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                current_val = new_score
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i] = old_val
                H -= delta * twiddles[i]
            T *= 0.9995
            
    return best_h

# EVOLVE-BLOCK-END
