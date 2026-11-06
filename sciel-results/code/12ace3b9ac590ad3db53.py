import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 24
    k = np.arange(n)
    
    # Precompute grid powers for coarse search
    m_coarse = 2048
    thetas_coarse = np.linspace(0, 2*np.pi, m_coarse, endpoint=False)
    powers_coarse = np.exp(1j * np.outer(k, thetas_coarse))
    
    # Precompute grid powers for fine search (verification)
    m_fine = 16384
    thetas_fine = np.linspace(0, 2*np.pi, m_fine, endpoint=False)
    powers_fine = np.exp(1j * np.outer(k, thetas_fine))

    def fast_eval_coarse(c):
        vals = np.dot(c, powers_coarse)
        return np.max(np.abs(vals))

    def fast_eval_fine(c):
        vals = np.dot(c, powers_fine)
        return np.max(np.abs(vals))

    best_c = None
    best_score_coarse = float('inf')
    best_score_fine = float('inf')
    
    # Keep track of top candidates to verify with fine grid
    top_candidates = []
    
    target_score = 1.2584 # Slightly above the known optimum 1.2583758
    
    def local_search(c):
        current_score = fast_eval_coarse(c)
        improved = True
        # Limit iterations
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_idx = -1
            best_new_score = current_score
            
            for i in range(n):
                c[i] *= -1
                new_score = fast_eval_coarse(c)
                c[i] *= -1 # revert
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_idx = i
                    
            if best_idx != -1:
                c[best_idx] *= -1
                current_score = best_new_score
                improved = True
                
        return c, current_score

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        # Initialize
        c = np.random.choice([-1, 1], size=n)
        
        # Local Search
        c, score = local_search(c)
        
        if score < best_score_coarse:
            best_score_coarse = score
            best_c = c.copy()
            
        # Add to top candidates
        top_candidates.append((score, c.copy()))
        
        # Keep only top 50 candidates to save memory/time
        if len(top_candidates) > 50:
            top_candidates.sort(key=lambda x: x[0])
            top_candidates = top_candidates[:50]
            
        restarts += 1
        
    # Final verification of top candidates with fine grid
    # Sort top candidates by coarse score
    top_candidates.sort(key=lambda x: x[0])
    
    # Evaluate the best few with fine grid
    for score, c in top_candidates[:10]: # Check top 10
        if time.time() > deadline - 2.0:
            break
        fine_score = fast_eval_fine(c)
        if fine_score < best_score_fine:
            best_score_fine = fine_score
            best_c = c.copy()
            
        # Early termination if we hit the known optimum
        if best_score_fine < target_score:
            break
            
    # If no fine score was computed (shouldn't happen), use coarse best
    if best_score_fine == float('inf'):
        best_c = best_c if best_c is not None else np.random.choice([-1, 1], size=n)
    else:
        best_c = best_c.copy()

    return best_c.tolist()
