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

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin, evaluator kills at 150s
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Parameters for Simulated Annealing
    # We will run multiple restarts
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random ±1
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Simulated Annealing
        # Initial temperature should be higher than typical delta E
        # Delta E for a block flip can be significant.
        # Let's estimate max possible delta.
        # If we flip half the signal, A changes completely.
        # A[k] is sum of 400 terms. Max change ~ 400.
        # So T_start ~ 100 or 200 is reasonable.
        T = 50.0
        T_min = 0.01
        cooling_rate = 0.995
        # Number of steps per restart
        # We want to use most of the time budget distributed across restarts
        # Or just run one long SA?
        # Restarting helps escape deep local minima.
        # Let's run SA until time is up or T is very low.
        
        # To balance restarts, let's limit steps per restart if we have many restarts planned,
        # or just let it run until time is up for the first few, then shorter?
        # Simpler: Run SA for a fixed number of steps, then check time.
        # If we do 100 restarts, and we have 120s, that's 1.2s per restart.
        # ~10,000 steps per restart.
        
        steps_per_restart = 10000
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Generate block flip
            # Block length L between 1 and 16
            L = np.random.randint(1, 17)
            i = np.random.randint(0, N - L + 1)
            
            # Flip block
            s[i:i+L] *= -1
            
            # Recalculate A and objective
            A = _calc_acf(s)
            new_val = _max_abs_acf_from_array(A)
            
            delta_E = new_val - current_val
            
            # Acceptance criterion
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_seq = s.copy()
            else:
                # Reject and flip back
                s[i:i+L] *= -1
            
            T *= cooling_rate
            if T < T_min:
                break
        
        # After SA, do a quick local polish with single-bit flips
        # This is cheap and can squeeze out more improvement
        improved = True
        polish_passes = 0
        max_polish_passes = 50
        
        while improved and polish_passes < max_polish_passes and time.time() < deadline:
            improved = False
            polish_passes += 1
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline:
                    break
                old_si = s[i]
                s[i] = -old_si
                A = _calc_acf(s)
                new_val = _max_abs_acf_from_array(A)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_seq = s.copy()
                else:
                    s[i] = old_si
                    
        # If we found a very good solution, we might stop early
        # What is a good solution?
        # Theoretical lower bound for max autocorrelation is ~ sqrt(N) ?
        # For N=400, sqrt(400)=20.
        # Barker sequences have max 1, but they don't exist for N>13.
        # Typical good sequences for N=400 might have max autocorr ~ 5-10.
        # If we get below 5, it's very good.
        if best_val < 5.0:
            break

    return best_seq
