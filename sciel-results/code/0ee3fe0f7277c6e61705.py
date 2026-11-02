import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def compute_cost(x):
        F = np.fft.fft(x)
        return np.sum(np.abs(F)**4)
        
    best_x = None
    best_cost = np.inf
    
    # Try a few different initializations
    # 1. Random
    # 2. Alternating (1, -1, 1, -1...)
    # 3. Blocks
    
    initializations = [
        np.random.choice([-1, 1], N),
        np.tile([1, -1], N // 2),
        np.concatenate([np.ones(N//2), -np.ones(N//2)])
    ]
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
            
        x = init.copy()
        current_cost = compute_cost(x)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
            
        # Greedy Coordinate Descent
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Random order to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                if time.time() > deadline - 5:
                    break
                
                # Calculate cost if we flip x[i]
                # Instead of full FFT, we can use the property that flipping x[i]
                # changes F[k] by -2*x[i]*exp(-j*2*pi*k*i/N)
                # But full FFT is fast enough for N=400.
                
                x[i] *= -1
                new_cost = compute_cost(x)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_x = x.copy()
                else:
                    x[i] *= -1 # Revert
                
        # Add some Simulated Annealing steps to escape local minima
        T = 10.0
        for _ in range(1000): # Fixed number of SA steps
            if time.time() > deadline - 5:
                break
            i = np.random.randint(N)
            x[i] *= -1
            new_cost = compute_cost(x)
            delta = new_cost - current_cost
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_x = x.copy()
            else:
                x[i] *= -1
            T *= 0.99

    return best_x
# EVOLVE-BLOCK-END
