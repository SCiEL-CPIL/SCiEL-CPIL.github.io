import numpy as np
import time

N = 400

def _calc_metrics(h):
    """
    Calculate the autoconvolution, integral, and the objective score.
    Returns: max_abs_autoconv, integral, score
    """
    # Autoconvolution
    # np.convolve(h, h, mode='full') gives length 2N-1
    # Lag 0 is at index N-1
    C = np.convolve(h, h, mode='full')
    # We consider lags k != 0.
    # C indices: 0 to 2N-2.
    # Lag 0 is C[N-1].
    # Sidelobes are C[0:N-1] and C[N:2N-1].
    max_abs_sid = np.max(np.abs(np.concatenate([C[:N-1], C[N:]])))
    
    integral = np.sum(h)
    int_sq = integral * integral
    
    if int_sq == 0:
        return max_abs_sid, integral, float('inf')
        
    score = max_abs_sid / int_sq
    return max_abs_sid, integral, score

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    # Initial sequence: All 1s
    h = np.ones(N, dtype=np.int8)
    
    # Get initial metrics
    max_abs_sid, integral, current_score = _calc_metrics(h)
    best_score = current_score
    best_h = h.copy()
    
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    rng = np.random.default_rng(42)
    
    # Pre-calculate integral update logic? 
    # Flipping h[i] from 1 to -1 changes integral by -2
    # Flipping h[i] from -1 to 1 changes integral by +2
    # Integral^2 changes significantly.
    
    while time.time() < deadline and T > T_min:
        # Pick a random index to flip
        i = rng.integers(0, N)
        
        # Flip
        old_val = h[i]
        h[i] = -old_val
        
        # Calculate new score
        m, _, new_score = _calc_metrics(h)
        
        # Calculate delta
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Accept worse solution with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                h[i] = old_val
        
        # Cool down
        T *= cooling_rate
        
        # Safety check inside loop for long-running iterations
        if time.time() > deadline:
            break
            
    # Final polish: Hill climbing from the best state found by SA
    # This helps refine the result if SA ended in a local optimum
    h = best_h.copy()
    _, _, current_score = _calc_metrics(h)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_flip_score = current_score
        
        for i in range(N):
            if time.time() > deadline:
                break
            h[i] *= -1
            _, _, s = _calc_metrics(h)
            if s < best_flip_score:
                best_flip_score = s
                best_flip_idx = i
                improved = True
            else:
                h[i] *= -1
                
        if improved:
            h[best_flip_idx] *= -1
            current_score = best_flip_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
            else:
                # If we found a better local min than the global best tracked?
                # Actually, we track best_h separately.
                # If current_score is better than best_score, update.
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
        else:
            break

    return best_h
