"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_off_peak_autocorr(seq):
    """Compute the maximum absolute off-peak autocorrelation."""
    n = len(seq)
    if n < 2:
        return 0.0
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    f = np.fft.fft(seq, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return np.max(np.abs(acf[1:]))

def construct():
    """Construct a signed step function to minimize the autoconvolution upper bound C3."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Phase 1: Greedy Construction
    seq = np.array([1.0])
    
    for i in range(1, N):
        if time.time() > deadline - 20:
            break
        
        seq_plus = np.append(seq, 1.0)
        score_plus = _max_off_peak_autocorr(seq_plus)
        
        seq_minus = np.append(seq, -1.0)
        score_minus = _max_off_peak_autocorr(seq_minus)
        
        if score_plus <= score_minus:
            seq = seq_plus
        else:
            seq = seq_minus
            
    if len(seq) < N:
        seq = np.concatenate([seq, np.ones(N - len(seq))])

    # Phase 2: Local Search (Hill Climbing)
    current_score = _max_off_peak_autocorr(seq)
    max_iterations = 200
    
    for _ in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        f = np.fft.fft(seq, n=n_fft)
        acf = np.fft.ifft(f * np.conj(f)).real[:N]
        
        abs_acf = np.abs(acf[1:])
        max_val = np.max(abs_acf)
        
        best_improvement = 0.0
        best_flip_idx = -1
        best_new_score = current_score
        
        for j in range(N):
            old_val = seq[j]
            seq[j] = -old_val
            
            new_score = _max_off_peak_autocorr(seq)
            
            seq[j] = old_val
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_flip_idx = j
                best_improvement = current_score - new_score
        
        if best_flip_idx != -1:
            seq[best_flip_idx] = -seq[best_flip_idx]
            current_score = best_new_score
        else:
            break
            
    return seq
# EVOLVE-BLOCK-END
