import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for overhead/evaluation
    
    # Objective function: Peak Sidelobe Level (PSL)
    # R(k) for k=1..N-1. Minimize max |R(k)|.
    def get_psl(s):
        # s is a numpy array of length N
        # Autocorrelation
        # np.correlate(s, s, mode='full') gives length 2N-1
        # Center is at N-1.
        # We want lags 1 to N-1.
        # Full correlation indices: 0 to 2N-2.
        # Lag 0 is at index N-1.
        # Lag 1 is at index N and N-2? No, for real signals, R(k) = R(-k).
        # np.correlate(s, s, 'full')[N-1 + k] is R(k) for k >= 0?
        # Let's verify:
        # s = [a, b]
        # full: [a^2, ab, ba, b^2] -> indices 0,1,2,3. N=2. N-1=1.
        # R(0) = a^2+b^2. Index 1 is ab. This is wrong.
        # Actually, np.correlate(s, s, 'full') computes sum s[i] * conj(s[i+k])?
        # Documentation: c_k = sum_n a_n * conj(v_{n+k})
        # If a=s, v=s.
        # For s=[1, 2], full = [1, 2, 4, 2, 1]? No.
        # Let's just use FFT for speed and correctness, or check np.correlate carefully.
        # Actually, for N=400, np.correlate is fast enough.
        # Let's use a simpler check:
        # R(k) = sum_{i=0}^{N-1-k} s[i] * s[i+k]
        # This is the standard autocorrelation.
        # np.correlate(s, s, mode='full') returns:
        # c[k] = sum_n s[n] * s[n + (k - (N-1))] ??
        # It's safer to compute via FFT to avoid confusion and potential O(N^2) slowdown if np.correlate isn't optimized for this specific case (it usually is).
        # FFT method:
        # F = fft(s, n=2*N)
        # R = ifft(|F|^2)
        # R[0:N] contains R(0)...R(N-1)
        # R[N:2N] contains R(-N+1)...R(-1)
        # We want max(|R[1:N]|, |R[N:2N-1]|) which is symmetric.
        # So max(|R[1:N]|) is sufficient.
        
        n_fft = 2 * N
        F = np.fft.fft(s, n=n_fft)
        power = np.abs(F)**2
        R = np.fft.ifft(power).real
        # R[0] is R(0)
        # R[1] is R(1)
        # ...
        # R[N-1] is R(N-1)
        # R[N] is R(-N+1) = R(N-1)
        # So we just look at R[1:N]
        return np.max(np.abs(R[1:N]))

    # Initial state: Random signs
    # Using a fixed seed for reproducibility in the construct function? 
    # The task implies "Choose a signed step function". 
    # If we use random, we need to ensure it's deterministic or just pick a good one found during search.
    # Since construct() is called by the evaluator, it should probably return the best found.
    # We will run the search inside construct().
    
    # To avoid different results on different calls (if called multiple times), 
    # we should use a fixed seed or cache the result. 
    # However, usually these tasks call construct() once. 
    # To be safe, we can use a fixed seed for the initial random state, 
    # but the search itself is deterministic given the seed.
    
    rng = np.random.default_rng(42)
    s = rng.choice([-1, 1], size=N)
    
    current_energy = get_psl(s)
    best_s = s.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    
    T = T_start
    steps = 0
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move: flip a random bit
        idx = rng.integers(0, N)
        old_val = s[idx]
        s[idx] = -old_val
        
        new_energy = get_psl(s)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_s = s.copy()
        else:
            # Reject move, revert
            s[idx] = old_val
            
        T *= alpha
        steps += 1
        
        # Optional: If stuck, do a random restart or larger jump?
        # For now, standard SA.
        
    return best_s
