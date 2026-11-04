# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 400
    
    def compute_objective(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        # Exclude lag 0
        sidelobes = R[1:]
        # Minimize L3 norm of sidelobes
        return np.sum(np.abs(sidelobes)**3)

    best_f = None
    best_obj = np.inf
    
    # Use multiple restarts to escape local minima
    num_restarts = 20
    
    for _ in range(num_restarts):
        # Initialize with random binary sequence
        f = np.random.choice([-1, 1], size=N)
        current_obj = compute_objective(f)
        
        # Local Search with pair flips
        # We will do a limited number of iterations per restart
        max_iter = 2000
        no_improve_count = 0
        max_no_improve = 100
        
        for i in range(max_iter):
            if no_improve_count > max_no_improve:
                break
                
            # Pick two distinct indices to flip
            idx1 = np.random.randint(0, N)
            idx2 = np.random.randint(0, N)
            while idx2 == idx1:
                idx2 = np.random.randint(0, N)
            
            # Flip
            f[idx1] *= -1
            f[idx2] *= -1
            
            new_obj = compute_objective(f)
            
            if new_obj < current_obj:
                current_obj = new_obj
                no_improve_count = 0
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_f = f.copy()
            else:
                # Revert
                f[idx1] *= -1
                f[idx2] *= -1
                no_improve_count += 1
                
    return best_f
# EVOLVE-BLOCK-END
