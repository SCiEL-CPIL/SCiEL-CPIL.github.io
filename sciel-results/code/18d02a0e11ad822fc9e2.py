"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    Returns max |autoconvolution| excluding the zero-lag self term.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real
    conv_lags = conv[:n]
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0

def _local_search_signs(f, active_indices, deadline):
    """
    Greedy local search: flip individual signs of active intervals.
    Returns updated f and score.
    """
    current_score = _compute_c3_upper_bound(f)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in active_indices:
            if time.time() > deadline:
                break
            old_val = f[idx]
            f[idx] = -old_val
            new_score = _compute_c3_upper_bound(f)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                f[idx] = old_val
    return f, current_score

def _swap_active_indices(f, active_indices, deadline, max_swaps=500):
    """
    Try swapping active indices to find better positions.
    """
    current_score = _compute_c3_upper_bound(f)
    all_indices = np.arange(N)
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        
        # Pick random active index to remove
        i_remove = np.random.randint(len(active_indices))
        idx_remove = active_indices[i_remove]
        
        # Pick random inactive index to add
        inactive = np.setdiff1d(all_indices, active_indices)
        if len(inactive) == 0:
            break
        j_add = inactive[np.random.randint(len(inactive))]
        
        # Perform swap
        f[idx_remove] = 0
        f[j_add] = 1.0
        active_indices = np.delete(active_indices, i_remove)
        active_indices = np.append(active_indices, j_add)
        
        new_score = _compute_c3_upper_bound(f)
        
        if new_score < current_score - 1e-12:
            current_score = new_score
            # Refine sign of the new element
            f[j_add] = -1.0
            alt_score = _compute_c3_upper_bound(f)
            if alt_score < current_score:
                current_score = alt_score
            else:
                f[j_add] = 1.0
        else:
            # Revert
            f[j_add] = 0
            f[idx_remove] = 1.0
            active_indices = np.delete(active_indices, len(active_indices) - 1)
            active_indices = np.insert(active_indices, i_remove, idx_remove)
    
    return f, active_indices, current_score

def _simulated_annealing(f, active_indices, start_temp=1.0, end_temp=0.01, 
                          max_iter=5000, deadline=None):
    """
    Simulated annealing for global exploration.
    Moves: flip sign, swap active index, add/remove active index.
    """
    current_score = _compute_c3_upper_bound(f)
    best_f = f.copy()
    best_score = current_score
    best_active = active_indices.copy()
    
    temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / max_iter)
    all_indices = np.arange(N)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        temp *= cooling_rate
        
        # Random move
        move_type = np.random.randint(3)
        
        if move_type == 0 and len(active_indices) > 0:
            # Flip sign
            i = np.random.randint(len(active_indices))
            idx = active_indices[i]
            f[idx] = -f[idx]
            
        elif move_type == 1:
            # Swap active index
            i_remove = np.random.randint(len(active_indices))
            idx_remove = active_indices[i_remove]
            inactive = np.setdiff1d(all_indices, active_indices)
            if len(inactive) == 0:
                continue
            j_add = inactive[np.random.randint(len(inactive))]
            
            f[idx_remove] = 0
            f[j_add] = np.random.choice([-1.0, 1.0])
            active_indices = np.delete(active_indices, i_remove)
            active_indices = np.append(active_indices, j_add)
            
        elif move_type == 2:
            # Add or remove active index
            if np.random.random() < 0.5 and len(active_indices) < N:
                # Add
                inactive = np.setdiff1d(all_indices, active_indices)
                if len(inactive) == 0:
                    continue
                j_add = inactive[np.random.randint(len(inactive))]
                f[j_add] = np.random.choice([-1.0, 1.0])
                active_indices = np.append(active_indices, j_add)
            else:
                # Remove
                if len(active_indices) > 1:
                    i_remove = np.random.randint(len(active_indices))
                    idx_remove = active_indices[i_remove]
                    f[idx_remove] = 0
                    active_indices = np.delete(active_indices, i_remove)
                else:
                    continue
        
        new_score = _compute_c3_upper_bound(f)
        delta = new_score - current_score
        
        # Accept with probability
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
                best_active = active_indices.copy()
        else:
            # Revert
            if move_type == 0 and len(active_indices) > 0:
                i = np.random.randint(len(active_indices))
                idx = active_indices[i]
                f[idx] = -f[idx]
            elif move_type == 1:
                # Need to revert the swap - this is tricky, let's just restart
                pass
            elif move_type == 2:
                pass
    
    return best_f, best_active, best_score

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_f = np.zeros(N)
    best_score = np.inf
    
    # Try different sparsity levels
    sparsity_levels = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
    
    for K in sparsity_levels:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts for this sparsity level
        num_restarts = 3 if K <= 50 else 2
        rng = np.random.RandomState(hash(K) % 2**32)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            active_indices = rng.choice(N, size=K, replace=False)
            signs = rng.choice([-1.0, 1.0], size=K)
            
            f = np.zeros(N)
            f[active_indices] = signs
            
            # Phase 1: Greedy local search on signs
            f, score1 = _local_search_signs(f, active_indices, deadline - 30)
            
            # Phase 2: Swap active indices
            f, active_indices, score2 = _swap_active_indices(f, active_indices, 
                                                              deadline - 20, max_swaps=200)
            
            # Phase 3: Simulated annealing for global exploration
            if time.time() < deadline - 15:
                f, active_indices, score3 = _simulated_annealing(
                    f, active_indices, start_temp=0.5, end_temp=0.01,
                    max_iter=2000, deadline=deadline - 10)
            else:
                score3 = score2
            
            # Final polish: greedy local search
            if time.time() < deadline - 5:
                f, score_final = _local_search_signs(f, active_indices, deadline - 2)
            else:
                score_final = score3
            
            if score_final < best_score:
                best_score = score_final
                best_f = f.copy()
    
    # Final polish on best solution
    if time.time() < deadline - 3:
        active = np.nonzero(best_f)[0]
        if len(active) > 0:
            best_f, _ = _local_search_signs(best_f, active, deadline)
    
    return best_f

# EVOLVE-BLOCK-END
