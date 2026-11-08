"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    def compute_c3(h):
        """Compute C3: max |autoconv(t)| for signed step function via FFT."""
        M = 2 * N  # fine grid resolution
        dx = 0.5 / M
        
        m_idx = np.arange(M)
        i_idx = (m_idx * N // M).astype(np.int64)
        i_idx = np.clip(i_idx, 0, N - 1)
        f_fine = h.astype(np.float64)[i_idx]
        
        fft_f = np.fft.fft(f_fine, n=2 * M)
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
        
        c3 = np.max(np.abs(autoconv))
        return c3
    
    # Structured initialization
    candidates = []
    candidates.append(np.random.choice([-1, 1], size=N))
    
    for block_size in [2, 4, 8, 16, 32, 64]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1 if (i // block_size) % 2 == 0 else -1
        candidates.append(h)
    
    t = np.linspace(0, 4 * np.pi, N)
    candidates.append(np.sign(np.sin(t)))
    candidates.append(np.sign(np.sin(2*t)))
    
    best_h = None
    best_c3 = np.inf
    for h in candidates:
        c3 = compute_c3(h)
        if c3 < best_c3:
            best_c3 = c3
            best_h = h.copy()
    
    h = best_h.copy()
    
    # Simulated annealing with restarts
    rng = np.random.RandomState(42)
    
    for restart in range(10):
        T_init = 0.05
        T_min = 0.0005
        cooling_rate = 0.99998
        
        T = T_init
        local_best_c3 = best_c3
        local_best_h = h.copy()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline:
            num_flips = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = rng.choice(N, size=num_flips, replace=False)
            
            h_new = h.copy()
            h_new[indices] *= -1
            
            c3_new = compute_c3(h_new)
            delta = c3_new - local_best_c3
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h = h_new
                if c3_new < local_best_c3:
                    local_best_c3 = c3_new
                    local_best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            if no_improve_count > max_no_improve:
                break
        
        if local_best_c3 < best_c3:
            best_c3 = local_best_c3
            best_h = local_best_h.copy()
            h = local_best_h.copy()
        else:
            # Restart from random
            h = np.random.choice([-1, 1], size=N)
            local_best_c3 = compute_c3(h)
            local_best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
