import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initialize
    f = np.random.choice([-1, 1], N)
    
    def calc_cost(f):
        F = np.fft.fft(f, n=2*N)
        R = np.fft.ifft(F * np.conj(F)).real
        return np.max(np.abs(R[1:N]))

    current_cost = calc_cost(f)
    best_f = f.copy()
    best_cost = current_cost
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    # Pre-allocate arrays for speed?
    
    while time.time() < deadline:
        idx = np.random.randint(N)
        f[idx] *= -1
        new_cost = calc_cost(f)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_f = f.copy()
        else:
            # Prob of accepting worse solution
            delta = new_cost - current_cost
            if np.random.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                f[idx] *= -1 # Revert
                # current_cost remains same
                
        T *= cooling
        if T < T_min:
            T = T_min

    return best_f
