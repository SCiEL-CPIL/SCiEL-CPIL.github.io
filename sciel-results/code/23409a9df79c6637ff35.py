import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    np.random.seed(42)
    
    best_h = None
    best_metric = float('inf')
    
    # Number of restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize random sequence
        h = np.random.choice([-1.0, 1.0], size=N)
        
        # Compute initial autocorrelation
        # R[k] = sum_j h_j h_{j+k}
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H)**2).real
        # Due to floating point errors, R might have tiny imaginary parts or asymmetries
        # R[0] should be N
        current_max = np.max(np.abs(R[1:]))
        
        if current_max < best_metric:
            best_metric = current_max
            best_h = h.copy()
            
        # Simulated Annealing
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        steps_per_restart = 20000
        
        # Check time for this restart
        # Approx time per step
        # We'll just run a fixed number of steps or until time limit
        
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / steps_per_restart)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            # Pick random index to flip
            i = np.random.randint(N)
            
            # Calculate change in R
            # delta_R[k] = -2 * h[i] * (h[(i+k)%N] + h[(i-k)%N])
            # h[(i+k)%N] is roll(h, -i)
            # h[(i-k)%N] is roll(h, i)
            
            h_fwd = np.roll(h, -i)
            h_bwd = np.roll(h, i)
            
            delta_R = -2.0 * h[i] * (h_fwd + h_bwd)
            
            R_new = R + delta_R
            
            # Compute new max
            # Note: R[0] is constant N, so we ignore it
            new_max = np.max(np.abs(R_new[1:]))
            
            delta_max = new_max - current_max
            
            # Acceptance probability
            if delta_max < 0:
                accept = True
            else:
                # Avoid log(0)
                if T > 1e-9:
                    accept = np.random.rand() < np.exp(-delta_max / T)
                else:
                    accept = False
            
            if accept:
                # Update state
                h[i] = -h[i]
                R = R_new
                current_max = new_max
                
                if current_max < best_metric:
                    best_metric = current_max
                    best_h = h.copy()
            
            T *= cooling_rate
            
            if T < T_end:
                T = T_end

    if best_h is None:
        return np.ones(N)
        
    return best_h
