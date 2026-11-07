"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    # np.correlate is O(N^2) for small N, which is fine for N=100.
    # We can optimize by pre-computing 1-h if h is static, but h changes.
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Simulated Annealing to minimize the maximum correlation.
    Strategy:
    1. Start with a diverse initial state (e.g., alternating 0/1 or random).
    2. Use moves that preserve the sum constraint exactly (swaps or paired perturbations).
    3. Accept worse solutions with a probability based on temperature to escape local minima.
    4. Cool down over time.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Initial best is the constant 0.5 solution
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # We will run multiple independent SA runs if time permits, 
    # or one long SA run with restarts if stuck.
    
    def simulated_annealing(h_init, max_iters=20000):
        h = h_init.copy()
        current_val = objective(h)
        
        # Initial temperature
        # The objective value is around 0.5. Deltas are small.
        T = 0.1
        T_min = 1e-5
        cooling_rate = 0.9999
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Generate a move
            # Move Type 1: Swap two elements (preserves sum exactly)
            # Move Type 2: Perturb two elements in opposite directions (preserves sum)
            
            move_type = np.random.random()
            if move_type < 0.5:
                # Swap
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    continue
                
                old1, old2 = h[idx1], h[idx2]
                h[idx1] = old2
                h[idx2] = old1
            else:
                # Paired Perturbation
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                if idx1 == idx2:
                    continue
                
                # Perturbation size scales with temperature
                delta = np.random.uniform(-T, T)
                
                new1 = h[idx1] + delta
                new2 = h[idx2] - delta
                
                # Check bounds
                if new1 < 0.0 or new1 > 1.0 or new2 < 0.0 or new2 > 1.0:
                    # Try to clip? No, just revert if out of bounds to keep it simple and valid
                    continue
                
                old1, old2 = h[idx1], h[idx2]
                h[idx1] = new1
                h[idx2] = new2
            
            new_val = objective(h)
            
            # Acceptance probability
            if new_val < current_val:
                # Always accept improvement
                current_val = new_val
            else:
                # Accept worse with probability exp(-deltaE / T)
                delta_E = new_val - current_val
                if delta_E > 0:
                    # T is small, so this is rare
                    prob = math.exp(-delta_E / max(T, 1e-9))
                    if np.random.random() < prob:
                        current_val = new_val
                    else:
                        # Revert
                        if move_type < 0.5:
                            h[idx1], h[idx2] = old1, old2
                        else:
                            h[idx1], h[idx2] = old1, old2
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return h, current_val

    # Strategy: Multiple restarts with different initializations
    # Initialization 1: Constant 0.5 (already best)
    # Initialization 2: Alternating 0, 1, 0, 1... (Sum = 50, valid)
    # Initialization 3: Random uniform, projected to sum=50
    # Initialization 4: Step function (0s then 1s)
    
    inits = []
    
    # 1. Alternating
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    inits.append(h_alt)
    
    # 2. Step function
    h_step = np.zeros(N)
    h_step[N//2:] = 1.0
    inits.append(h_step)
    
    # 3. Random
    h_rand = np.random.uniform(0, 1, N)
    h_rand += (N/2.0 - np.sum(h_rand)) / N
    h_rand = np.clip(h_rand, 0, 1)
    # Re-project roughly
    for _ in range(10):
        h_rand = np.clip(h_rand, 0, 1)
        h_rand += (N/2.0 - np.sum(h_rand)) / N
    inits.append(h_rand)
    
    # 4. Gradient
    h_grad = np.linspace(0, 1, N)
    h_grad += (N/2.0 - np.sum(h_grad)) / N
    h_grad = np.clip(h_grad, 0, 1)
    inits.append(h_grad)
    
    # Run SA on each initialization
    # We will iterate through them, giving each a chunk of time
    num_inits = len(inits)
    time_per_init = (time_limit - (time.time() - start_time)) / num_inits
    
    for i, h_init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        # Calculate max iters for this chunk based on time
        # Estimate time per iteration. 
        # objective() takes ~10-20 us. 
        # Let's just run until time slice is up or max_iters reached.
        # We can't easily know max_iters without running. 
        # Let's use a fixed number of iters per init, scaled by remaining time.
        # If we have 30s left and 4 inits, we have 7.5s each.
        # ~10k iters/sec? Let's assume 5k iters/sec to be safe.
        max_iters_chunk = int(time_per_init * 5000)
        if max_iters_chunk < 1000:
            max_iters_chunk = 1000
            
        h_new, val_new = simulated_annealing(h_init, max_iters_chunk)
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        # If we found the theoretical lower bound?
        # The lower bound is not strictly 0.5. 
        # If h is 0/1 valued, sum=50. 
        # Correlation of 0/1 sequence with its complement.
        # If h is 0/1, 1-h is the bitwise NOT.
        # corr(k) = sum h_i (1-h_{i+k}).
        # If h is alternating 0101..., 1-h is 1010...
        # Lag 0: sum 0*1 + 1*0 = 0.
        # Lag 1: sum h_i (1-h_{i+1}).
        # h: 0 1 0 1 ...
        # 1-h: 1 0 1 0 ...
        # i=0: h0=0, 1-h1=0 -> 0
        # i=1: h1=1, 1-h2=1 -> 1
        # i=2: h2=0, 1-h3=0 -> 0
        # i=3: h3=1, 1-h4=1 -> 1
        # Sum is roughly N/2 = 50.
        # Value = 50/100 * 2 = 1.0.
        # Constant 0.5 gives 0.5. So 0.5 is better than 1.0.
        # So we want to stay close to 0.5 but break the symmetry to reduce max correlation?
        # Wait, for constant 0.5, corr(k) = sum 0.5*0.5 = 25. Max is 25. Value 0.5.
        # If we perturb, does max corr go down?
        # If we make h slightly non-uniform, some lags might have lower product, but others might have higher?
        # Actually, by Cauchy-Schwarz or similar, the "most uniform" distribution minimizes the variance, but here we look at cross-correlation.
        # Let's let SA find out.
        
    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
