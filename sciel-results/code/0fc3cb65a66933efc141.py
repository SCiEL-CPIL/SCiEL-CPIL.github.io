import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 24
    M = 4096 # Grid size
    
    # Precompute grid and exponentials
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    k = np.arange(N)
    # z_k has shape (N, M)
    z_k = np.exp(1j * np.outer(k, theta))
    
    def refine_peak(coeffs):
        # coeffs is list of +1/-1
        c = np.array(coeffs, dtype=float)
        # Find approximate max on grid
        P = np.sum(c[:, np.newaxis] * z_k, axis=0)
        mag = np.abs(P)
        idx_max = np.argmax(mag)
        theta_max = theta[idx_max]
        
        # Golden section search
        delta = 2 * np.pi / M
        a = theta_max - delta
        b = theta_max + delta
        
        def f(t):
            return np.abs(np.sum(c * np.exp(1j * k * t)))
            
        gr = (np.sqrt(5) - 1) / 2
        c_pt = a + (1 - gr) * (b - a)
        d_pt = a + gr * (b - a)
        
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(50):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = a + (1 - gr) * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
                
        return f((a + b) / 2)

    def local_search(init_coeffs):
        current_coeffs = init_coeffs[:]
        c = np.array(current_coeffs, dtype=float)
        
        # Compute initial P
        P = np.sum(c[:, np.newaxis] * z_k, axis=0)
        max_current = np.abs(P).max()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                # Flip bit i
                old_val = c[i]
                # P_new = P_old - 2 * old_val * z_k[i]
                P_trial = P - 2 * old_val * z_k[i]
                max_trial = np.abs(P_trial).max()
                
                if max_trial < max_current:
                    # Accept
                    c[i] = -old_val
                    P = P_trial
                    max_current = max_trial
                    improved = True
                # else: revert (do nothing, c[i] unchanged)
                    
        # Refine the peak of the local minimum found
        final_peak = refine_peak(current_coeffs)
        return current_coeffs, final_peak

    best_score = np.inf
    best_coeffs = None
    
    # Initial candidates
    # GRS
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = p[:N]
    
    candidates = [grs_coeffs]
    for _ in range(20):
        candidates.append([1 if np.random.rand() > 0.5 else -1 for _ in range(N)])
        
    for init in candidates:
        if time.time() >= deadline:
            break
        coeffs, peak = local_search(init)
        if peak < best_score:
            best_score = peak
            best_coeffs = coeffs[:]
            
    # More random restarts
    while time.time() < deadline - 5:
        init = [1 if np.random.rand() > 0.5 else -1 for _ in range(N)]
        coeffs, peak = local_search(init)
        if peak < best_score:
            best_score = peak
            best_coeffs = coeffs[:]
            
    return best_coeffs
