import numpy as np
import time

N = 50

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.max(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    # Compute autoconvolution
    # Using FFT for speed if N is large, but N=50 is small, np.convolve is fine
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if l1 * peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _local_search(h, deadline, rng):
    """Coordinate ascent local search."""
    current_h = h.copy()
    current_score = _score(current_h)
    
    # Normalize to L1=1 to stabilize scaling
    l1 = np.sum(current_h)
    if l1 > 0:
        current_h = current_h / l1
        current_score = _score(current_h)
        
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            old_val = current_h[i]
            # Try perturbing the value
            # Candidates: 0, small, large, relative changes
            candidates = [0.0, 1e-3, 1e-2, 0.1, 0.5]
            if old_val > 0:
                candidates.extend([old_val * 0.5, old_val * 1.5, old_val + 0.01, old_val - 0.01])
            
            best_val = old_val
            best_s = current_score
            
            for val in candidates:
                if val < 0: continue
                current_h[i] = val
                # Re-normalize? 
                # If we don't normalize, the score scales. 
                # The evaluator likely normalizes. 
                # Let's assume the evaluator normalizes to L1=1.
                # So we should maximize score under L1=1 constraint.
                # Easiest: just evaluate raw score, but since score is scale invariant 
                # (if evaluator normalizes) or scales as 1/c (if not), 
                # we should probably keep the sum fixed or let the evaluator handle it.
                # Given the ambiguity, I will normalize internally to L1=1 
                # to search the simplex.
                
                l1_new = np.sum(current_h)
                if l1_new == 0:
                    continue
                h_norm = current_h / l1_new
                s = _score(h_norm)
                if s > best_s + 1e-9:
                    best_s = s
                    best_val = val
            
            if best_val != old_val:
                current_h[i] = best_val
                l1_new = np.sum(current_h)
                if l1_new > 0:
                    current_h = current_h / l1_new
                current_score = _score(current_h)
                improved = True
    
    return current_h, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = _score(best_h)
    
    rng = np.random.RandomState(42)
    
    # Try several random sparse supports
    for k in range(1, 15):
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random support
            idx = rng.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            
            # Local search
            h_opt, s_opt = _local_search(h, deadline, rng)
            
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
    # Final polish on the best found
    if time.time() < deadline - 5:
        best_h, best_score = _local_search(best_h, deadline, rng)
        
    return list(best_h)
