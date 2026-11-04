"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_peak(f):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(f)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    # Next power of 2 for efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, n=size)
    acf = np.fft.irfft(np.abs(F) ** 2, n=size)
    # acf[0] is the zero-lag term (sum of squares)
    # We want max |acf[k]| for k in 1..n-1
    # Note: acf is symmetric, but irfft returns full real vector
    # We only need to check positive lags up to n-1
    return float(np.max(np.abs(acf[1:n])))

def _local_search(f, time_limit):
    """
    Greedy local search:
    1. Flip signs of active elements.
    2. Move active elements to inactive positions.
    """
    n = len(f)
    best_f = f.copy()
    best_score = _autocorr_peak(best_f)
    start_time = time.time()
    
    # Identify active indices
    active_mask = best_f != 0
    active_indices = np.where(active_mask)[0]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # --- Phase 1: Sign Flips ---
        # Try flipping signs of all active indices in random order
        if len(active_indices) > 0:
            order = np.random.permutation(len(active_indices))
            for idx_pos in order:
                if (time.time() - start_time) > time_limit:
                    break
                i = active_indices[idx_pos]
                # Flip sign
                best_f[i] = -best_f[i]
                new_score = _autocorr_peak(best_f)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    best_f[i] = -best_f[i]
            
            # Update active indices if we moved things? No, signs don't change active set.
            # But if we do moves, we need to update active_indices.
            
        # --- Phase 2: Index Moves ---
        # Try moving an active element to an inactive position
        if len(active_indices) > 0:
            inactive_indices = np.where(~active_mask)[0]
            if len(inactive_indices) > 0:
                # Try a few random moves
                num_moves_to_try = min(50, len(active_indices) * 2)
                for _ in range(num_moves_to_try):
                    if (time.time() - start_time) > time_limit:
                        break
                    
                    # Pick random active index to move
                    i_src = active_indices[np.random.randint(len(active_indices))]
                    val = best_f[i_src]
                    
                    # Pick random inactive index to move to
                    i_dst = inactive_indices[np.random.randint(len(inactive_indices))]
                    
                    # Try moving with sign +1 and -1
                    best_move_score = best_score
                    best_sign = 0
                    
                    # Save original state
                    orig_src = best_f[i_src]
                    orig_dst = best_f[i_dst]
                    
                    # Try +1
                    best_f[i_src] = 0
                    best_f[i_dst] = 1
                    s_plus = _autocorr_peak(best_f)
                    if s_plus < best_move_score:
                        best_move_score = s_plus
                        best_sign = 1
                    
                    # Try -1
                    best_f[i_dst] = -1
                    s_minus = _autocorr_peak(best_f)
                    if s_minus < best_move_score:
                        best_move_score = s_minus
                        best_sign = -1
                    
                    # Apply best move if it improved
                    if best_sign != 0:
                        best_f[i_src] = 0
                        best_f[i_dst] = best_sign
                        best_score = best_move_score
                        improved = True
                        
                        # Update active_mask and indices
                        active_mask[i_src] = False
                        active_mask[i_dst] = True
                        # Reconstruct active_indices list for next iteration? 
                        # It's O(N) but N=400 is small.
                        active_indices = np.where(active_mask)[0]
                    else:
                        # Revert
                        best_f[i_src] = orig_src
                        best_f[i_dst] = orig_dst
        
    return best_f, best_score

def _simulated_annealing(f, time_limit):
    """
    Simulated Annealing to escape local minima.
    Moves: Flip sign, Move index, Zero out, Activate.
    """
    n = len(f)
    current_f = f.copy()
    current_score = _autocorr_peak(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 2.0
    cooling = 0.99
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial_f = current_f.copy()
        
        r = np.random.random()
        if r < 0.5:
            # Flip sign (or activate if 0)
            if trial_f[i] == 0:
                trial_f[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial_f[i] = -trial_f[i]
        elif r < 0.8:
            # Zero out
            trial_f[i] = 0.0
        else:
            # Move: pick another index j, swap or move
            j = np.random.randint(n)
            # Simple move: set i to 0, set j to sign
            if trial_f[i] != 0:
                val = trial_f[i]
                trial_f[i] = 0
                # Determine sign for j based on original val or random
                trial_f[j] = val if np.random.random() < 0.5 else -val
            else:
                # If i is 0, just activate
                trial_f[i] = 1.0 if np.random.random() < 0.5 else -1.0
        
        new_score = _autocorr_peak(trial_f)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_f = trial_f
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = current_f.copy()
        
        T *= cooling
        
    return best_f, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_f = np.ones(N)  # Fallback
    best_score = _autocorr_peak(best_f)
    
    # Try different sparsity levels
    # Sparse sequences often have lower peak autocorrelation
    sparsity_levels = [30, 50, 70, 90, 110, 130]
    
    # Multiple restarts per sparsity level
    num_restarts = 5
    
    for K in sparsity_levels:
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization
            indices = np.random.choice(N, size=K, replace=False)
            f = np.zeros(N)
            f[indices] = np.random.choice([-1, 1], size=K)
            
            # Phase 1: Local Search (Greedy)
            time_remaining = deadline - time.time()
            # Allocate 20% of remaining time to local search per restart?
            # Or fixed small time for local search to allow many restarts
            ls_time = min(2.0, time_remaining * 0.2)
            if ls_time > 0.1:
                f_opt, score_opt = _local_search(f, ls_time)
            else:
                f_opt, score_opt = f, _autocorr_peak(f)
            
            # Phase 2: Simulated Annealing
            if time.time() < deadline - 5:
                sa_time = min(5.0, (deadline - time.time()) * 0.5)
                if sa_time > 0.5:
                    f_opt, score_opt = _simulated_annealing(f_opt, sa_time)
            
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt.copy()
                
    # Final polish on the best solution found
    if time.time() < deadline - 5:
        final_time = deadline - time.time() - 2
        if final_time > 1.0:
            best_f, best_score = _local_search(best_f, final_time * 0.5)
            if time.time() < deadline - 2:
                best_f, best_score = _simulated_annealing(best_f, final_time * 0.5)
                
    return best_f
# EVOLVE-BLOCK-END
