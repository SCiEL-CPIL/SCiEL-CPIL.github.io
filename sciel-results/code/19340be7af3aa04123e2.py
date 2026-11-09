import numpy as np
import time
import heapq

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Powers for polynomial evaluation
    powers = np.arange(N)
    
    # Coarse grid for initial filtering
    M_coarse = 4096
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * powers[None, :])
    
    # Fine grid for refinement
    M_fine = 200000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * powers[None, :])
    
    # Total candidates: 2^23 (fix c0=1)
    total_candidates = 1 << 23
    chunk_size = 20000
    
    # Heap to keep top K candidates
    K_top = 100
    top_candidates = []  # List of (score, coeffs)
    
    shifts = np.arange(23)
    
    def evaluate_refined(c):
        """Evaluate sup norm with golden section refinement."""
        # Evaluate on fine grid
        vals = Z_fine @ c
        abs_vals = np.abs(vals)
        
        # Find top 10 peaks
        part_idx = np.argpartition(abs_vals, -10)[-10:]
        peak_indices = part_idx[np.argsort(abs_vals[part_idx])[::-1]]
        
        max_val = 0.0
        step = 2 * np.pi / M_fine
        phi = (np.sqrt(5) - 1) / 2
        
        for idx in peak_indices:
            theta_center = thetas_fine[idx]
            a = theta_center - 1.5 * step
            b = theta_center + 1.5 * step
            
            c1 = b - phi * (b - a)
            c2 = a + phi * (b - a)
            f1 = np.abs(np.dot(c, np.exp(1j * c1 * powers)))
            f2 = np.abs(np.dot(c, np.exp(1j * c2 * powers)))
            
            for _ in range(40):
                if f1 < f2:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - phi * (b - a)
                    f1 = np.abs(np.dot(c, np.exp(1j * c1 * powers)))
                else:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + phi * (b - a)
                    f2 = np.abs(np.dot(c, np.exp(1j * c2 * powers)))
                    
            theta_max = (a + b) / 2
            val_max = np.abs(np.dot(c, np.exp(1j * theta_max * powers)))
            
            if val_max > max_val:
                max_val = val_max
                
        return max_val

    for start in range(0, total_candidates, chunk_size):
        if time.time() > deadline - 30:
            break
            
        end = min(start + chunk_size, total_candidates)
        count = end - start
        
        indices = np.arange(start, end)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        c_rest = 1 - 2 * bits
        c0 = np.ones((count, 1))
        coeffs = np.hstack((c0, c_rest))
        
        # Evaluate on coarse grid
        vals = coeffs @ Z_coarse.T
        max_norms = np.max(np.abs(vals), axis=1)
        
        # Select top K from this chunk
        if count > K_top:
            part_idx = np.argpartition(max_norms, K_top)[:K_top]
            top_scores_chunk = max_norms[part_idx]
            top_coeffs_chunk = coeffs[part_idx]
        else:
            top_scores_chunk = max_norms
            top_coeffs_chunk = coeffs
            
        for i in range(len(top_scores_chunk)):
            score = top_scores_chunk[i]
            coeff = top_coeffs_chunk[i]
            
            if len(top_candidates) < K_top:
                heapq.heappush(top_candidates, (score, coeff))
            else:
                if score < top_candidates[0][0]:
                    heapq.heapreplace(top_candidates, (score, coeff))
                    
    # Refine the top K candidates
    best_c = None
    best_score = np.inf
    
    # Sort top candidates by coarse score to refine the best ones first
    top_candidates.sort(key=lambda x: x[0])
    
    for score, c in top_candidates:
        if time.time() > deadline - 5:
            break
            
        fine_score = evaluate_refined(c)
        
        if fine_score < best_score:
            best_score = fine_score
            best_c = c.copy()
            
    if best_c is None:
        # Fallback: all ones
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
