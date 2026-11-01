"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(s):
    """Compute autocorrelation using FFT."""
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _peak_sidelobe(acf):
    """Peak absolute sidelobe (excluding main lobe)."""
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_seq = None
    best_val = np.inf
    
    np.random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 3:
            break
        
        s = np.random.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(s)
        current_val = _peak_sidelobe(acf)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Local search: try all single flips, pick best
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_flip_val = current_val
            best_flip_idx = -1
            
            for i in range(N):
                # Compute new acf after flipping i
                # Δr[k] = -2*s[i]*s[i+k]*[i+k<N] - 2*s[i-k]*s[i]*[i-k>=0]
                new_acf = acf.copy()
                si = s[i]
                # For k where i+k < N
                k_max = N - i
                if k_max > 0:
                    new_acf[:k_max] -= 2 * si * s[i:i+k_max]
                # For k where i-k >= 0, i.e., k <= i
                if i > 0:
                    new_acf[:i+1] -= 2 * s[:i+1] * si
                
                new_val = _peak_sidelobe(new_acf)
                if new_val < best_flip_val:
                    best_flip_val = new_val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                s[best_flip_idx] = -s[best_flip_idx]
                acf = _compute_acf(s)  # Recompute to avoid drift
                current_val = _peak_sidelobe(acf)
                improved = True
                if current_val < best_val:
                    best_val = current_val
                    best_seq = s.copy()
        
        if best_val <= 5:  # Good enough, stop early
            break
    
    if best_seq is None:
        return np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
