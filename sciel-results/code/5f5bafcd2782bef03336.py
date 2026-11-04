"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # New Approach: Simulated Annealing on Binary Signals
    # Objective: Minimize Integrated Sidelobe Level (ISL)
    # ISL = sum_{tau != 0} R(tau)^2
    
    # Initialize random binary signal
    f = np.random.choice([-1, 1], size=N)
    
    # Function to compute ISL
    def compute_isl(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        # R[0] is the peak (DC term), usually we exclude it
        # R is circular autocorrelation
        # Sidelobes are R[1:] and R[-1] (which is R[N-1])
        # In circular autocorrelation, R[k] corresponds to lag k.
        # The peak is at lag 0.
        # We sum squares of all lags except 0.
        return np.sum(R[1:]**2) + R[-1]**2 if N > 1 else 0
        
    # Wait, np.fft.ifft returns indices 0 to N-1.
    # R[0] is lag 0.
    # R[1] is lag 1.
    # ...
    # R[N-1] is lag -1 (or N-1).
    # So sidelobes are R[1:N] and R[0] is peak.
    # Actually, for real signals, R is symmetric.
    # ISL = sum(R[1:]**2) + sum(R[:N-1][::-1]**2)? No.
    # Circular autocorrelation R[k] = sum f[n] f[n+k].
    # The "sidelobes" are all k != 0.
    # So ISL = sum(R[k]**2 for k in 1..N-1).
    
    def compute_isl_fast(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        # Exclude R[0]
        return np.sum(R[1:]**2)

    current_isl = compute_isl_fast(f)
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    max_iter = 50000
    
    best_f = f.copy()
    best_isl = current_isl
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        # Pick random index to flip
        idx = np.random.randint(0, N)
        
        # Flip
        old_val = f[idx]
        f[idx] = -f[idx]
        
        new_isl = compute_isl_fast(f)
        delta = new_isl - current_isl
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept
            current_isl = new_isl
            if new_isl < best_isl:
                best_isl = new_isl
                best_f = f.copy()
        else:
            # Reject
            f[idx] = old_val
            
        T *= cooling_rate
        
    return best_f
# EVOLVE-BLOCK-END
