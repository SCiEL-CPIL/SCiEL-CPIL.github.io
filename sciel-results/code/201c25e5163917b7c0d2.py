import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 24
    best_coeffs = None
    best_score = float('inf')
    
    # Precompute powers of z for search grid
    M_search = 2000
    angles_s = np.linspace(0, 2 * np.pi, M_search, endpoint=False)
    z_s = np.exp(1j * angles_s)
    Z_s = np.power(z_s[None, :], np.arange(N)[:, None])
    
    def get_supnorm_search(c):
        P = Z_s @ c
        return np.max(np.abs(P)), P
    
    # Initial guess: Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial = np.array(p[:N], dtype=float)
    
    # Generate starts
    starts = [initial]
    # Add random starts
    for _ in range(50):
        starts.append(np.random.choice([-1, 1], size=N))
    
    for start in starts:
        if time.time() > deadline - 5: # Leave 5s for final check
            break
            
        c = start.copy()
        current_score, P = get_supnorm_search(c)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                # Flip i
                P_new = P - 2 * c[i] * Z_s[i]
                new_score = np.max(np.abs(P_new))
                if new_score < current_score - 1e-12:
                    c[i] *= -1
                    P = P_new
                    current_score = new_score
                    improved = True
                    break
        
        # Refine best
        if current_score < best_score:
            # Use finer grid for final score check
            M_fine = 50000
            angles_f = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
            z_f = np.exp(1j * angles_f)
            Z_f = np.power(z_f[None, :], np.arange(N)[:, None])
            fine_score = np.max(np.abs(Z_f @ c))
            if fine_score < best_score:
                best_score = fine_score
                best_coeffs = c
    
    if best_coeffs is None:
        return initial.tolist()
        
    return best_coeffs.tolist()
