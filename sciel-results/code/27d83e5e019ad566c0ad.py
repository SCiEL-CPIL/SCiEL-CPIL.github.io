import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def get_peak(h):
        n_fft = 512 # 2^9, > 2*400
        F = np.fft.fft(h, n=n_fft)
        P = np.fft.ifft(np.abs(F)**2).real
        left = P[1:N]
        right = P[n_fft-N+1:]
        return np.max(np.abs(np.concatenate([left, right])))

    best_h = np.ones(N, dtype=np.float64)
    best_P = get_peak(best_h)
    
    # Simulated Annealing
    T_start = 10.0
    T_end = 0.1
    alpha = 0.999 # cooling rate
    T = T_start
    
    # Initialize with random
    h = np.random.choice([-1.0, 1.0], size=N)
    P = get_peak(h)
    
    if P < best_P:
        best_P = P
        best_h = h.copy()
        
    steps = 0
    while time.time() < deadline - 5:
        # Pick random index
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_P = get_peak(h)
        delta = new_P - P
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            P = new_P
            if P < best_P:
                best_P = P
                best_h = h.copy()
        else:
            h[i] = old_val
            
        T *= alpha
        if T < T_end:
            T = T_end
            # Optionally restart or just keep going
            # Restarting might help escape local minima
            # Let's do a soft restart: keep h, but increase T slightly? No, just continue.
            # Or full restart?
            # Full restart is better for SA.
            h = np.random.choice([-1.0, 1.0], size=N)
            P = get_peak(h)
            T = T_start * 0.5 # Start cooler for subsequent restarts?
            # Actually, let's just restart with same T_start
            T = T_start
            
    return best_h
