import numpy as np
import time

def construct():
        N = 24
        start_time = time.time()
        deadline = start_time + 110
        
        # Grid for coarse filtering
        M_coarse = 1024
        thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
        Z_coarse = np.exp(1j * thetas_coarse[:, None] * np.arange(N)[None, :])
        
        # Grid for fine evaluation
        M_fine = 20000
        thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
        Z_fine = np.exp(1j * thetas_fine[:, None] * np.arange(N)[None, :])
        
        # Search space: fix c0=1.
        # Free bits: c1...c23 (23 bits)
        total_combinations = 1 << 23
        chunk_size = 500000 # Larger chunks to reduce overhead
        
        # List to hold (coarse_score, coeffs)
        # We only need to keep the top K globally.
        # We can use a heap or just store all if memory allows?
        # 8.4 million candidates. Storing all scores (floats) is 8.4M * 8 bytes = 67 MB.
        # Storing all coeffs (int8) is 8.4M * 24 bytes = 200 MB.
        # Total ~270 MB. This is acceptable.
        # However, sorting 8.4 million items might take time.
        # Better: keep a running top K.
        K = 1000
        top_scores = np.full(K, np.inf)
        top_coeffs = np.zeros((K, N), dtype=int)
        
        # We will process chunks and update the top K.
        
        for start in range(0, total_combinations, chunk_size):
            if time.time() > deadline - 20: # Leave 20s for refinement
                break
                
            end = min(start + chunk_size, total_combinations)
            count = end - start
            
            # Generate indices
            indices = np.arange(start, end)
            
            # Generate bits for c1...c23
            shifts = np.arange(23)
            bits = (indices[:, None] >> shifts[None, :]) & 1
            
            # Convert to +1/-1
            c_rest = 1 - 2 * bits
            
            # Prepend c0=1
            c0 = np.ones((count, 1), dtype=int)
            coeffs = np.hstack((c0, c_rest))
            
            # Evaluate on coarse grid
            # coeffs: (count, N), Z_coarse.T: (N, M_coarse)
            vals = coeffs @ Z_coarse.T
            max_norms = np.max(np.abs(vals), axis=1)
            
            # Find top K in this chunk
            if count > K:
                # Get indices of smallest K in this chunk
                part_idx = np.argpartition(max_norms, K)[:K]
                chunk_scores = max_norms[part_idx]
                chunk_coeffs = coeffs[part_idx]
            else:
                chunk_scores = max_norms
                chunk_coeffs = coeffs
            
            # Merge with global top K
            # We have global top K (scores, coeffs) and new chunk top K (scores, coeffs)
            # Combine them, sort, take top K
            all_scores = np.concatenate([top_scores, chunk_scores])
            all_coeffs = np.vstack([top_coeffs, chunk_coeffs])
            
            # Sort by score
            sort_idx = np.argsort(all_scores)
            top_scores = all_scores[sort_idx[:K]]
            top_coeffs = all_coeffs[sort_idx[:K]]
            
        # Refine the top K candidates
        best_c = None
        best_score = np.inf
        
        for i in range(len(top_scores)):
            if time.time() > deadline - 5:
                break
            c = top_coeffs[i]
            # Evaluate with fine grid
            vals_fine = Z_fine @ c
            fine_score = np.max(np.abs(vals_fine))
            
            if fine_score < best_score:
                best_score = fine_score
                best_c = c.copy()
                
        # Fallback
        if best_c is None:
            best_c = np.ones(N, dtype=int)
        else:
            best_c = best_c.astype(int)
            
        return [int(x) for x in best_c]
