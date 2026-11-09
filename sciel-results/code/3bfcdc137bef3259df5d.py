import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 24
    
    # Precompute cosine matrix for a grid
    M = 2048
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    pair_indices = [(k, l) for k in range(N) for l in range(k+1, N)]
    num_pairs = len(pair_indices)
    
    cos_vals = np.zeros((M, num_pairs))
    for idx, (k, l) in enumerate(pair_indices):
        cos_vals[:, idx] = np.cos((k - l) * thetas)
        
    def eval_sup_norm(c):
        # c is numpy array of shape (N,)
        pp = np.array([c[k] * c[l] for k, l in pair_indices])
        
        val_sq = 24 + 2 * (cos_vals @ pp)
        
        max_val = np.max(val_sq)
        
        top_indices = np.argsort(val_sq)[-3:]
        
        best_peak_sq = max_val
        
        def eval_p(theta):
            z_powers = np.exp(1j * theta * np.arange(N))
            val = np.dot(c, z_powers)
            return abs(val)
            
        for idx in top_indices:
            delta = 2 * (2 * np.pi / M)
            a = thetas[idx] - delta
            b = thetas[idx] + delta
            
            gr = (np.sqrt(5) - 1) / 2
            c_point = b - gr * (b - a)
            d_point = a + gr * (b - a)
            fc = eval_p(c_point)
            fd = eval_p(d_point)
            
            for _ in range(40):
                if fc < fd:
                    a = c_point
                    c_point = d_point
                    fc = fd
                    d_point = a + gr * (b - a)
                    fd = eval_p(d_point)
                else:
                    b = d_point
                    d_point = c_point
                    fd = fc
                    c_point = b - gr * (b - a)
                    fc = eval_p(c_point)
                    
            peak_val = max(fc, fd)
            if peak_val**2 > best_peak_sq:
                best_peak_sq = peak_val**2
                
        return np.sqrt(best_peak_sq)

    # Initial candidate: Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N])
    best_score = eval_sup_norm(best_coeffs)
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    T = 0.5
    cooling = 0.9995
    
    while time.time() < deadline:
        i = np.random.randint(N)
        new_coeffs = current_coeffs.copy()
        new_coeffs[i] = -new_coeffs[i]
        
        new_score = eval_sup_norm(new_coeffs)
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            current_coeffs = new_coeffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
                
        T *= cooling
        
    return [int(x) for x in best_coeffs]
