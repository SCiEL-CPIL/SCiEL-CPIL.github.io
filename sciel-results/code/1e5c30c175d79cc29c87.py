import numpy as np
import time

N = 400
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _autocorr_max_abs(s):
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.default_rng(12345)
    
    # Try multiple random restarts
    best_s = None
    best_val = np.inf
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        s = rng.choice([-1.0, 1.0], size=N)
        val = _autocorr_max_abs(s)
        
        if val < best_val:
            best_val = val
            best_s = s.copy()
        
        # Simulated annealing
        T = 3.0
        T_min = 0.01
        cooling = 0.9998
        cur_s = s.copy()
        cur_val = val
        
        steps_this_restart = 0
        while time.time() < deadline:
            block_size = int(rng.integers(2, 33))
            if block_size >= N:
                block_size = N - 1
            start = int(rng.integers(0, N - block_size + 1))
            
            old_block = cur_s[start:start+block_size].copy()
            cur_s[start:start+block_size] = -old_block
            
            new_val = _autocorr_max_abs(cur_s)
            delta = new_val - cur_val
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                cur_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = cur_s.copy()
            else:
                cur_s[start:start+block_size] = old_block
            
            steps_this_restart += 1
            T = max(T_min, T * cooling)
            
            # Restart cooling if stuck
            if steps_this_restart % 10000 == 0 and T < 0.1:
                T = 1.0
        
        # Greedy polish
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                old = cur_s[i]
                cur_s[i] = -old
                nv = _autocorr_max_abs(cur_s)
                if nv < cur_val:
                    cur_val = nv
                    if nv < best_val:
                        best_val = nv
                        best_s = cur_s.copy()
                    improved = True
                else:
                    cur_s[i] = old
    
    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
