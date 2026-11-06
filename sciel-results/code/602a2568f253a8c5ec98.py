import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 100.0
    
    N = 24
    M = 2048
    B = 2000
    
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    k = np.arange(N)
    E = np.exp(1j * np.outer(thetas, k))
    
    best_norm = np.inf
    best_coeffs = None
    
    total_cases = 1 << 23
    shifts = np.arange(23)
    
    for start in range(0, total_cases, B):
        if time.time() - start_time > deadline:
            break
            
        end = min(start + B, total_cases)
        B_cur = end - start
        
        idx = np.arange(start, end)
        bits = (idx[:, None] >> shifts) & 1
        C_vars = 1 - 2 * bits
        C = np.hstack((np.ones((B_cur, 1)), C_vars))
        
        V = C @ E.T
        norms = np.max(np.abs(V), axis=1)
        
        local_min_idx = np.argmin(norms)
        local_min_val = norms[local_min_idx]
        
        if local_min_val < best_norm:
            best_norm = local_min_val
            best_coeffs = C[local_min_idx].astype(int)
            
    if best_coeffs is None:
        p, q = [1], [1]
        while len(p) < 24:
            p, q = p + q, p + [-c for c in q]
        best_coeffs = np.array(p[:24])
        
    return list(best_coeffs)
