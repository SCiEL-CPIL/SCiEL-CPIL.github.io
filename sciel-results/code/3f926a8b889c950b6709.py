import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _init_sine(freq, phase):
    """Initialize sign vector from a low-frequency sine wave."""
    x = np.linspace(-0.25, 0.25, N)
    wave = np.sin(2 * np.pi * freq * x + phase)
    s = np.where(wave >= 0, 1.0, -1.0)
    return s

def _local_search(s, A, current_val, deadline, temperature=0.0):
    """Perform local search with exact delta updates for ACF.
    If temperature > 0, uses Simulated Annealing to accept worse moves."""
    improved = True
    passes = 0
    max_passes = 50
    t = temperature
    min_t = 1e-6
    cooling = 0.95
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline:
                return s, A, current_val
            
            old_si = s[i]
            
            # Compute delta for A when flipping s[i]
            delta = np.zeros(N)
            
            # Part 1: k where i+k < N, so k in [0, N-1-i]
            if N - 1 - i >= 0:
                delta[:N - i] += -2.0 * old_si * s[i:N]
            
            # Part 2: k where i-k >= 0, so k in [0, i]
            if i >= 0:
                delta[:i+1] += -2.0 * old_si * s[i::-1]
            
            # Apply delta to get A_new
            A_new = A + delta
            
            # Compute new max abs (excluding k=0)
            val_new = float(np.max(np.abs(A_new[1:])))
            delta_val = val_new - current_val
            
            if delta_val < -1e-12:
                # Accept the flip
                s[i] = -old_si
                A = A_new
                current_val = val_new
                improved = True
            elif temperature > 0 and t > min_t:
                # Accept worse moves with probability exp(-delta_val / t)
                if np.random.random() < np.exp(-delta_val / t):
                    s[i] = -old_si
                    A = A_new
                    current_val = val_new
                    improved = True
                    t *= cooling
                    if t < min_t:
                        t = min_t
        
        if temperature > 0:
            t *= cooling
            if t < min_t:
                t = min_t

    return s, A, current_val

def construct():
    start_time = time.time()
    deadline = time.time() + 110.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Try different sine wave initializations
    freqs = [1, 2, 3, 4, 5, 6, 8, 10]
    
    while time.time() < deadline:
        # Choose a frequency and phase
        freq = np.random.choice(freqs)
        phase = np.random.uniform(0, 2 * np.pi)
        
        s = _init_sine(freq, phase)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Greedy local search first
        s, A, current_val = _local_search(s, A, current_val, deadline, temperature=0.0)
        
        # Then annealing to escape local minima
        if time.time() < deadline - 5.0:
            s, A, current_val = _local_search(s, A, current_val, deadline, temperature=1.0)
            # Final greedy polish
            s, A, current_val = _local_search(s, A, current_val, deadline, temperature=0.0)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Also try random initialization occasionally
        if np.random.random() < 0.3:
            s = np.random.randint(0, 2, size=N) * 2 - 1
            A = _calc_acf(s)
            current_val = _max_abs_acf_from_array(A)
            
            s, A, current_val = _local_search(s, A, current_val, deadline, temperature=0.0)
            if time.time() < deadline - 5.0:
                s, A, current_val = _local_search(s, A, current_val, deadline, temperature=1.0)
                s, A, current_val = _local_search(s, A, current_val, deadline, temperature=0.0)
            
            if current_val < best_val:
                best_val = current_val
                best_seq = s.copy()

    return best_seq
