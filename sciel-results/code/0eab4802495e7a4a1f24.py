import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    N = 24
    n_samples = 2048
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    k = np.arange(N)
    Z = np.exp(1j * np.outer(k, thetas)) # Shape (N, n_samples)
    
    batch_size = 4096
    total_candidates = 1 << 23
    
    # Store top candidates
    # List of (score, coeffs)
    top_candidates = []
    
    for start in range(0, total_candidates, batch_size):
        if time.time() - start_time > timeout - 10: # Leave time for refinement
            break
            
        end = min(start + batch_size, total_candidates)
        num_in_batch = end - start
        
        masks = np.arange(start, end, dtype=np.int64)
        masks_col = masks[:, None]
        shifts_row = np.arange(1, N)[None, :]
        bits = (masks_col >> shifts_row) & 1
        C = 1 - 2 * bits
        C = np.hstack((np.ones((num_in_batch, 1), dtype=np.int8), C))
        
        values = C @ Z
        max_vals = np.max(np.abs(values), axis=1)
        
        # Get indices of top 5 in this batch
        # Sort indices by max_vals
        # np.argpartition is faster than full sort
        if num_in_batch > 5:
            top_5_idx = np.argpartition(max_vals, 5)[:5]
        else:
            top_5_idx = np.argsort(max_vals)[:num_in_batch]
            
        for idx in top_5_idx:
            score = max_vals[idx]
            coeffs = C[idx]
            top_candidates.append((score, coeffs))
            
        # Keep only top 10 overall
        if len(top_candidates) > 10:
            top_candidates.sort(key=lambda x: x[0])
            top_candidates = top_candidates[:10]

    # Refine top candidates
    best_coeffs = None
    best_score = float('inf')
    
    def eval_p(theta, coeffs):
        # coeffs is array of shape (N,)
        # theta is scalar
        powers = np.exp(1j * theta * np.arange(N))
        return np.abs(np.sum(coeffs * powers))
        
    for score, coeffs in top_candidates:
        if time.time() - start_time > timeout - 5:
            break
            
        coeffs_f = coeffs.astype(np.float64)
        
        # Find grid peak for this candidate
        vals = Z.T @ coeffs_f
        peak_idx = np.argmax(np.abs(vals))
        theta_peak = thetas[peak_idx]
        
        dtheta = 2 * np.pi / n_samples
        a = theta_peak - dtheta
        b = theta_peak + dtheta
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_p(c, coeffs_f)
        fd = eval_p(d, coeffs_f)
        
        for _ in range(50):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_p(c, coeffs_f)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_p(d, coeffs_f)
                
            if time.time() - start_time > timeout - 2:
                break
                
        final_val = max(fc, fd, eval_p((a+b)/2, coeffs_f))
        
        if final_val < best_score:
            best_score = final_val
            best_coeffs = coeffs_f

    if best_coeffs is None:
        # Fallback to Golay-Rudin-Shapiro if search failed or timed out badly
        p, q = [1], [1]
        while len(p) < 24:
            p, q = p + q, p + [-c for c in q]
        return p[:24]
        
    return [int(c) for c in best_coeffs]
