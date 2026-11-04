import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for 120s limit, or 150s kill
    
    # Initialization
    # Try a few different seeds and pick the best one after a short warm-up
    # Or just one long run.
    
    x = np.random.choice([-1, 1], N)
    
    def get_score(x):
        # Compute linear autocorrelation
        # Zero pad to 2*N
        X = np.fft.fft(x, 2*N)
        C = np.fft.ifft(np.abs(X)**2)
        # C[0:N] contains c_0 ... c_{N-1}
        # We care about k=1 to N-1
        c = np.real(C[:N])
        # Exclude c_0
        return np.max(np.abs(c[1:]))
        
    current_score = get_score(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    # Pre-allocate arrays if needed, but FFT is fast enough
    # N=400, FFT size 800. Very fast.
    
    steps = 0
    while time.time() - start_time < timeout:
        # Pick random index
        i = np.random.randint(N)
        
        # Flip
        x[i] *= -1
        
        new_score = get_score(x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = new_score - current_score
            if delta > 0 and np.random.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                x[i] *= -1
        
        steps += 1
        # Cool down
        if steps % 1000 == 0:
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # If T is min, we are doing greedy. 
                # We could break if no improvement for long, but let's keep running until time.
                
    return best_x
