import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def compute_cost(seq):
        padded = np.zeros(2 * N)
        padded[:N] = seq
        F = np.fft.fft(padded)
        power = np.abs(F) ** 2
        r = np.fft.ifft(power).real
        # Minimize sum of squares of non-zero lags
        return np.sum(r[1:N] ** 2)
    
    best_seq = None
    best_cost = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize random
        seq = np.random.choice([-1, 1], size=N).astype(float)
        current_cost = compute_cost(seq)
        
        if current_cost < best_cost:
            best_seq = seq.copy()
            best_cost = current_cost
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.995
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline:
            idx = np.random.randint(N)
            new_seq = seq.copy()
            new_seq[idx] *= -1
            
            new_cost = compute_cost(new_seq)
            
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or (T > T_min and np.random.random() < np.exp(-delta_cost / T)):
                seq = new_seq
                current_cost = new_cost
                if current_cost < best_cost:
                    best_seq = seq.copy()
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            T *= alpha
            if T < T_min or no_improve_count > max_no_improve:
                break
                
    return best_seq
