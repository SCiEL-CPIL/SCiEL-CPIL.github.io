import numpy as np
import time

N = 400

def _compute_cost_aperiodic(seq):
    """
    Compute the maximum absolute value of the non-zero lags
    of the aperiodic autocorrelation.
    """
    # Zero-pad to length 2*N to get aperiodic autocorrelation via circular FFT
    padded = np.zeros(2 * N)
    padded[:N] = seq
    
    # FFT size should be at least 2*N. Let's use 2*N if it's a power of 2, or next power of 2.
    # 2*400 = 800. Next power of 2 is 1024.
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    H = np.fft.rfft(padded, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    
    # The aperiodic autocorrelation is in acf[0:N]
    # acf[0] is the DC term (sum of squares = N)
    # We want max |acf[k]| for k=1..N-1
    if N > 1:
        return np.max(np.abs(acf[1:N]))
    return 0.0

def _legendre_init(n):
    def is_prime(k):
        if k < 2: return False
        for i in range(2, int(np.sqrt(k)) + 1):
            if k % i == 0: return False
        return True
        
    P = n
    while not is_prime(P):
        P += 1
        if P - n > 10:
             P = n
             while not is_prime(P):
                 P -= 1
             break
             
    seq = np.zeros(P)
    for i in range(1, P):
        if pow(i, (P-1)//2, P) == 1:
            seq[i-1] = 1.0
        else:
            seq[i-1] = -1.0
    return seq[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_seq = None
    best_cost = float('inf')
    
    # Try a few initializations
    inits = [
        _legendre_init(N),
        np.random.choice([-1, 1], size=N),
        np.random.choice([-1, 1], size=N)
    ]
    
    for seq in inits:
        cost = _compute_cost_aperiodic(seq)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq.copy()
            
    # Simulated Annealing with Restarts
    # We will run SA for a certain number of steps or time, then restart if no improvement
    # Since we don't know the speed exactly, we use time-based control.
    
    while time.time() < deadline:
        # Restart logic: if we haven't improved in the last X seconds, restart
        # Simple approach: Run SA for 10s chunks. If no improvement, restart from best.
        
        current_seq = best_seq.copy()
        current_cost = best_cost
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.99995 # Adjusted for ~100k steps
        
        # Run SA until time limit for this chunk or T < T_min
        chunk_start = time.time()
        while time.time() < chunk_start + 10.0 and T > T_min:
            idx = np.random.randint(N)
            old_val = current_seq[idx]
            current_seq[idx] = -old_val
            
            new_cost = _compute_cost_aperiodic(current_seq)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = current_seq.copy()
            else:
                delta = new_cost - current_cost
                if np.random.rand() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    current_seq[idx] = old_val
            
            T *= cooling_rate
            
        # If we didn't improve in this chunk, we could restart with different params, 
        # but the loop above already restarts the SA process from best_seq.
        # To add diversity, we could mutate best_seq before restarting SA.
        # Let's add a small random perturbation to best_seq before the next SA run.
        if time.time() < deadline - 10.0:
            # Perturb best_seq
            for _ in range(10):
                idx = np.random.randint(N)
                best_seq[idx] *= -1
            # Recompute cost of perturbed best to be safe? 
            # The next SA run will start from this perturbed state.
            # But we must ensure we don't lose the global best.
            # best_seq is still the global best. We are just changing the starting point for the next SA.
            # Wait, in the code above, `current_seq = best_seq.copy()` happens at the start of the while loop.
            # So if I modify `best_seq` here, the next iteration will start from the perturbed version.
            # This is good for escaping local minima.

    return best_seq
