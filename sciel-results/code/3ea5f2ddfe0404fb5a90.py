import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_f = None
    best_max = float('inf')
    
    def max_sidelobe(f):
        f_fft = np.fft.rfft(f, n=2*N)
        A_fft = f_fft * f_fft
        A = np.fft.irfft(A_fft, n=2*N)
        # A[0] is main lobe. Sidelobes are 1..N-1
        return np.max(np.abs(A[1:N]))
    
    # Run several iterations of local search
    # Each iteration: Random init, then greedy flips
    
    while time.time() < deadline:
        # Initialize
        f = np.random.choice([-1, 1], size=N)
        current_max = max_sidelobe(f)
        
        # Local Search
        stuck_count = 0
        while time.time() < deadline:
            # Find worst lag
            f_fft = np.fft.rfft(f, n=2*N)
            A_fft = f_fft * f_fft
            A = np.fft.irfft(A_fft, n=2*N)
            sidelobes = A[1:N]
            max_val = np.max(np.abs(sidelobes))
            
            if max_val < best_max:
                best_max = max_val
                best_f = f.copy()
                
            if max_val < 1e-9:
                break
                
            # If we haven't improved in a while, restart?
            # For now, just do random flips
            
            # Try flipping a random index
            flip_idx = np.random.randint(N)
            f[flip_idx] *= -1
            
            new_max = max_sidelobe(f)
            
            if new_max < current_max:
                current_max = new_max
                stuck_count = 0
            else:
                f[flip_idx] *= -1
                stuck_count += 1
                # If stuck too many times, break inner loop and restart outer
                if stuck_count > 100:
                    break
        
        # Check deadline before starting next outer iteration
        if time.time() > deadline - 5.0:
            break
            
    if best_f is None:
        # Fallback to seed if something went wrong (unlikely)
        return np.ones(N)
        
    return best_f

# EVOLVE-BLOCK-START
# The code above is placed here
# EVOLVE-BLOCK-END
