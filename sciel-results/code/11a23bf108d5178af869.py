import numpy as np
import time
import random

N = 600
# Safety margin: The evaluator kills at 150s. We aim to finish well before.
TIME_LIMIT = 145

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    Returns: max(conv) / (sum(f))^2
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
        
    # Use FFT for speed, though for N=600 direct is also fine.
    # For the score function, direct is simple and safe.
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def simulated_annealing(f_init, time_limit=10.0, T_start=1.0, T_end=0.01, cooling_rate=0.9999):
    """
    Perform Simulated Annealing on the step heights.
    Strategy:
    - Binary-ish search: Flip intervals on/off or adjust magnitude.
    - Actually, the prompt suggests "binary on/off intervals". 
    - However, continuous heights allow for better optimization. 
    - Let's stick to a hybrid: Start with continuous, but use SA to explore.
    - Move: Pick random index i, change f[i] to a new value.
    - New value generation: 
        1. Flip to 0 if non-zero, or to a random positive value if zero.
        2. Or, multiply by a random factor (geometric random walk).
    - The prompt specifically says: "Model the step function as a sequence of binary on/off intervals... stochastically flips individual intervals".
    - Let's implement a discrete SA that operates on a set of "active" bins.
    - But wait, if we restrict to binary (0 or 1), we lose the ability to scale. 
    - The score is scale invariant. So we can fix the sum or fix the heights to 1.
    - If heights are 1, score = max(conv) / (number_of_ones)^2.
    - Let's try a binary SA first as suggested, but allow for "weights" if it gets stuck?
    - No, let's follow the directive: "binary on/off intervals".
    
    Implementation:
    - State: boolean array of length N.
    - Objective: max(conv) / (sum(f))^2. Since f is 0/1, sum(f) is count of ones.
    - Move: Flip a random bit.
    - Acceptance: Metropolis criterion.
    """
    start_time = time.time()
    
    # Initialize from f_init: threshold at median or mean?
    # Or just random binary with some density?
    # Let's start with a random binary pattern with ~50% density.
    # Or use f_init to guide initial state: f[i] > threshold -> 1
    thresh = np.median(f_init[f_init > 0]) if np.any(f_init > 0) else 0.5
    f_bin = (f_init > thresh).astype(np.float64)
    
    # Ensure we have some ones
    if np.sum(f_bin) < 10:
        f_bin = np.random.rand(N) < 0.5
        f_bin = f_bin.astype(np.float64)
        
    f = f_bin.copy()
    s = np.sum(f)
    conv = np.convolve(f, f, mode='full')
    current_score = np.max(conv) / (s * s)
    
    best_f = f.copy()
    best_score = current_score
    
    T = T_start
    
    # Precompute indices
    indices = np.arange(N)
    
    while time.time() - start_time < time_limit:
        # Batch of moves
        moves_per_batch = 1000
        for _ in range(moves_per_batch):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random index
            i = random.randint(0, N - 1)
            old_val = f[i]
            
            # Flip
            new_val = 1.0 - old_val
            
            # Calculate delta score efficiently?
            # For binary flips, we can update conv incrementally.
            # But for simplicity in this structural change, let's just recalc if N is small enough?
            # N=600, convolve is ~microseconds. 1000 moves * 100ms? No.
            # Let's use incremental update for speed.
            
            # Update conv
            # conv_new = conv + (new_val - old_val) * (2 * f_shifted) + correction
            # If old=0, new=1: delta=1. conv += 2*f_shifted + 1 at 2i
            # If old=1, new=0: delta=-1. conv -= 2*f_shifted + 1 at 2i (wait, -2*f_shifted - 1? No)
            
            # Let's just do a fast check. If we do 1000 moves per second, we need ~1000 * 150s = 150k moves.
            # Recalculating conv each time: 150k * (600^2 / 2) ~ 27 billion ops. Too slow in Python loop.
            # Must use incremental update.
            
            delta = new_val - old_val
            if delta == 0:
                continue
                
            # Indices affected: k in [i, i+N-1] intersect [0, 2N-2]
            k_min = i
            k_max = i + N - 1
            
            # f_vals = f[k-i]
            f_vals = f[k_min - i : k_max - i + 1]
            
            # Save old conv values for revert if rejected
            conv_old_slice = conv[k_min:k_max+1].copy()
            
            # Apply change
            conv[k_min:k_max+1] += 2 * delta * f_vals
            
            # Diagonal correction
            k_diag = 2 * i
            if k_diag >= k_min and k_diag <= k_max:
                # Added 2*delta*old_val. Want new^2 - old^2.
                # diff = (new^2 - old^2) - 2*delta*old = delta^2
                conv[k_diag] += delta * delta
            
            s_new = s + delta
            if s_new <= 0:
                # Revert
                conv[k_min:k_max+1] = conv_old_slice
                if k_diag >= k_min and k_diag <= k_max:
                     # Revert diagonal correction too? 
                     # conv[k_diag] was updated by 2*delta*old + delta^2.
                     # We just restored the slice, so it's fine.
                     pass
                continue
            
            new_score = np.max(conv) / (s_new * s_new)
            
            # Metropolis acceptance
            if new_score < current_score or (T > 0 and random.random() < np.exp((current_score - new_score) / T)):
                # Accept
                f[i] = new_val
                s = s_new
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                # Reject: Revert conv
                conv[k_min:k_max+1] = conv_old_slice
                # Note: f[i] is not changed, so f_vals in next steps will be correct.
                
        # Cooling
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_f, best_score

def construct():
    start_time = time.time()
    
    # Initial best: Constant function (all 1s)
    f_const = np.ones(N)
    # Binary SA starts with a binary pattern. 
    # Let's start with a few different binary patterns.
    
    best_f = f_const.copy()
    best_score = compute_score(f_const)
    
    # List of initializations for SA
    # 1. Random 50%
    # 2. Random 30%
    # 3. Block of ones in center
    # 4. Alternating
    # 5. Gaussian shaped (thresholded)
    
    x = np.linspace(-0.25, 0.25, N)
    
    inits = []
    
    # 1. Random 50%
    f_rand1 = (np.random.rand(N) < 0.5).astype(np.float64)
    inits.append(f_rand1)
    
    # 2. Random 30%
    f_rand2 = (np.random.rand(N) < 0.3).astype(np.float64)
    inits.append(f_rand2)
    
    # 3. Block center
    f_block = np.zeros(N)
    f_block[N//4 : 3*N//4] = 1.0
    inits.append(f_block)
    
    # 4. Alternating
    f_alt = (np.arange(N) % 2 == 0).astype(np.float64)
    inits.append(f_alt)
    
    # 5. Gaussian thresholded
    f_gauss = np.exp(-((x - 0.0) / 0.1) ** 2)
    thresh = np.median(f_gauss)
    f_gauss_bin = (f_gauss > thresh).astype(np.float64)
    inits.append(f_gauss_bin)
    
    # Time allocation
    # Total ~140s. 5 inits. ~25s each?
    # SA is fast. Let's give each 20s.
    
    time_per_init = 20.0
    
    for idx, f_init in enumerate(inits):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Run SA
        f_res, score_res = simulated_annealing(f_init, time_limit=time_per_init)
        
        if score_res < best_score:
            best_score = score_res
            best_f = f_res
            
        # Optional: Local refinement? 
        # The directive says "leverage discrete nature... escape local minima".
        # SA does this. 
        # But maybe we can do a "refine" step where we allow continuous values?
        # The prompt says "Model the step function as a sequence of binary on/off intervals".
        # It implies the search space is binary. 
        # However, the final output can be any nonnegative step function.
        # If we find a good binary pattern, can we scale it? 
        # Scaling doesn't change score. 
        # Can we add small weights? 
        # If we allow small weights, we might improve. 
        # But let's stick to the binary SA for now to see if it beats the seed.
        
    # Final check
    # Ensure non-negative
    best_f = np.maximum(best_f, 0.0)
    
    # One last thing: If the best binary pattern is good, maybe a simple 
    # coordinate descent on the continuous values starting from this binary 
    # pattern could squeeze out more?
    # The current seed program does coordinate descent. 
    # This new strategy is SA. 
    # Let's add a short continuous refinement phase using the best binary pattern.
    
    if time.time() - start_time < TIME_LIMIT - 5:
        # Convert best_f to a starting point for a very short continuous CD
        # Just to clean up edges?
        # Actually, let's just return the binary result. 
        # If it's worse, so be it. It's a structural change.
        
        pass
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
