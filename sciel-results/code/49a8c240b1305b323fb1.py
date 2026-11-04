"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s
    
    best_f = None
    best_val = np.inf
    
    np.random.seed(42)
    
    # Structural change: sparse combinatorial design with two-phase refinement
    # Phase 1: Greedy local exploitation (fast sign flips)
    # Phase 2: Simulated annealing for global exploration (support changes + sign flips)
    
    # Try different sparsity levels - sparse patterns can have much lower
    # autoconvolution peaks because most lags have few or zero overlapping non-zero terms
    sparsity_levels = [20, 30, 40, 50, 60, 80, 100, 120, 150]
    
    for K in sparsity_levels:
        if time.time() > deadline - 30:
            break
        
        # Multiple random starts for each sparsity level
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            # Random initialization: pick K random indices with random signs
            np.random.seed(42 + trial * 1000 + K)
            active_indices = np.random.choice(N, size=K, replace=False)
            signs = np.random.choice([-1, 1], size=K)
            
            f = np.zeros(N)
            f[active_indices] = signs
            
            current_score = _autoconvolution_max_abs(f)
            
            if current_score < best_val:
                best_val = current_score
                best_f = f.copy()
            
            # Phase 1: Fast greedy local exploitation - flip individual signs
            improved = True
            max_sweeps = 20
            sweep = 0
            while improved and sweep < max_sweeps and time.time() < deadline - 20:
                improved = False
                sweep += 1
                for idx in active_indices:
                    if time.time() > deadline - 20:
                        break
                    old_val = f[idx]
                    f[idx] = -old_val
                    new_score = _autoconvolution_max_abs(f)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                        if new_score < best_val:
                            best_val = new_score
                            best_f = f.copy()
                    else:
                        f[idx] = old_val
            
            # Phase 2: Simulated annealing for global exploration
            # This allows escaping local minima by accepting worse solutions with some probability
            if time.time() < deadline - 20:
                # Annealing parameters
                T_start = max(1.0, current_score * 0.1)
                T_end = 1e-6
                alpha = 0.95  # cooling rate
                n_iter = 5000
                
                T = T_start
                for it in range(n_iter):
                    if time.time() > deadline - 20:
                        break
                    
                    # Random move: either flip a sign or swap support
                    move_type = np.random.random()
                    
                    if move_type < 0.7:
                        # Sign flip
                        idx = active_indices[np.random.randint(len(active_indices))]
                        old_val = f[idx]
                        f[idx] = -old_val
                        new_score = _autoconvolution_max_abs(f)
                        
                        delta = new_score - current_score
                        if delta < 0 or np.random.random() < np.exp(-delta / T):
                            current_score = new_score
                            if new_score < best_val:
                                best_val = new_score
                                best_f = f.copy()
                        else:
                            f[idx] = old_val
                    else:
                        # Support swap: remove one active, add one inactive
                        i_remove = np.random.randint(len(active_indices))
                        inactive_mask = np.ones(N, dtype=bool)
                        inactive_mask[active_indices] = False
                        inactive_indices = np.where(inactive_mask)[0]
                        
                        if len(inactive_indices) == 0:
                            T *= alpha
                            continue
                        
                        j_add = inactive_indices[np.random.randint(len(inactive_indices))]
                        
                        idx_remove = active_indices[i_remove]
                        old_remove_val = f[idx_remove]
                        
                        f[idx_remove] = 0
                        f[j_add] = 1
                        
                        new_active_indices = list(active_indices)
                        new_active_indices.pop(i_remove)
                        new_active_indices.append(j_add)
                        
                        new_score = _autoconvolution_max_abs(f)
                        
                        delta = new_score - current_score
                        if delta < 0 or np.random.random() < np.exp(-delta / T):
                            current_score = new_score
                            active_indices = np.array(new_active_indices)
                            if new_score < best_val:
                                best_val = new_score
                                best_f = f.copy()
                        else:
                            f[j_add] = 0
                            f[idx_remove] = old_remove_val
                    
                    T *= alpha
    
    # Final polish: one more greedy pass on the best solution found
    if best_f is not None and time.time() < deadline - 5:
        active = np.nonzero(best_f)[0]
        improved = True
        max_sweeps = 10
        sweep = 0
        while improved and sweep < max_sweeps and time.time() < deadline - 5:
            improved = False
            sweep += 1
            for idx in active:
                if time.time() > deadline - 5:
                    break
                old_val = best_f[idx]
                best_f[idx] = -old_val
                s = _autoconvolution_max_abs(best_f)
                if s < best_val - 1e-12:
                    best_val = s
                    improved = True
                else:
                    best_f[idx] = old_val
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
