"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def _calc_score_fft(h):
    """
    Calculate the upper bound on C3 for a given step function h.
    Score = max_{k != 0} |C(k)| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    C(k) = sum_{i} h[i] h[i+k]
    """
    size = 2 * N
    H = np.fft.fft(h, size)
    power_spectrum = H * np.conj(H)
    corr = np.fft.ifft(power_spectrum).real
    
    # Linear autocorrelation extraction
    # c[0:N] -> lags 0 to N-1
    # c[N:2N] -> lags -(N-1) to -1 (specifically c[N+1] is C(-(N-1)) ... c[2N-1] is C(-1))
    # We want max abs for k != 0.
    # Positive lags: 1 to N-1 -> indices 1 to N-1
    # Negative lags: -(N-1) to -1 -> indices N+1 to 2N-1
    
    # Note: index N in the IFFT of zero-padded sequence corresponds to lag -N (or N depending on convention), 
    # which is 0 for linear correlation of length N sequence padded to 2N.
    # Actually, let's verify indices.
    # x = [x0, ..., x_{N-1}, 0, ..., 0] length 2N
    # circular autocorr r[m] = sum_{n=0}^{2N-1} x[n] x[(n+m) mod 2N]
    # For m in 0..N-1, r[m] = sum_{n=0}^{N-1-m} x[n] x[n+m] = C(m)
    # For m in N+1..2N-1, let m = 2N - k where k in 1..N-1.
    # r[2N-k] = sum_{n=0}^{2N-1} x[n] x[(n-k) mod 2N]
    # Since x is zero-padded, non-zero terms only when n in 0..N-1 and (n-k) mod 2N in 0..N-1.
    # If n >= k, (n-k) is in 0..N-1-k. Term is x[n]x[n-k].
    # If n < k, (n-k) mod 2N = 2N+n-k. This is > N-1 (since n<k<=N-1 => 2N+n-k > 2N-(N-1) = N+1). So x is 0.
    # So r[2N-k] = sum_{n=k}^{N-1} x[n] x[n-k]. Let j = n-k. sum_{j=0}^{N-1-k} x[j+k] x[j] = C(k).
    # So r[2N-k] = C(k) = C(-k) by symmetry.
    # Thus, C(k) for k=1..N-1 is at index k.
    # C(-k) for k=1..N-1 is at index 2N-k.
    
    # So we need max of abs(corr[1:N]) and abs(corr[N+1:2N]).
    # corr[N] is the "middle" which corresponds to lag N (or -N), which is 0.
    
    max_val = np.max(np.abs(np.concatenate((corr[1:N], corr[N+1:2*N]))))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return max_val / norm

def construct():
    global _start_time
    
    # Initial random sequence
    np.random.seed(42) # Fixed seed for reproducibility in this run? No, better to use random.
    # Actually, using random is fine.
    h = np.random.choice([-1.0, 1.0], size=N)
    current_score = _calc_score_fft(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 1e-6
    cooling = 0.9995
    
    while time.time() - _start_time < _DEADLINE:
        if T < T_min:
            break
            
        # Choose a move
        # Mix of single flips and double flips
        if np.random.random() < 0.7:
            idx = np.random.randint(N)
            h[idx] *= -1
            new_score = _calc_score_fft(h)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[idx] *= -1
        else:
            # Double flip
            idx1, idx2 = np.random.choice(N, 2, replace=False)
            h[idx1] *= -1
            h[idx2] *= -1
            new_score = _calc_score_fft(h)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[idx1] *= -1
                h[idx2] *= -1
        
        T *= cooling
        
        # Periodic local search refinement
        if T < 0.05 and time.time() - _start_time < _DEADLINE - 10:
            improved = True
            while improved and time.time() - _start_time < _DEADLINE - 5:
                improved = False
                for i in range(N):
                    if time.time() - _start_time > _DEADLINE - 5:
                        break
                    h[i] *= -1
                    val = _calc_score_fft(h)
                    if val < current_score:
                        current_score = val
                        if val < best_score:
                            best_score = val
                            best_h = h.copy()
                        improved = True
                    else:
                        h[i] *= -1
            # Reset temperature slightly to escape potential local minima if we just did a deep local search
            T = max(T, 0.01)
            cooling = 0.9999 # Slower cooling for the rest

    return best_h
# EVOLVE-BLOCK-END
