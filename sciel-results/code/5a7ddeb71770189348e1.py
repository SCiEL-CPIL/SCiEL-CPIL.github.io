import numpy as np
import time

def construct():
    n_coeffs = 24
    # Use high resolution for accurate comparison
    n_samples = 4096
    
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    Z = np.exp(1j * np.outer(np.arange(n_coeffs), thetas))
    
    start_time = time.time()
    timeout = 145.0  # 5s margin
    
    best_coeffs = None
    best_norm = np.inf
    
    def local_search(initial_coeffs):
        coeffs = initial_coeffs.copy()
        # Initial V
        V = np.sum(coeffs[:, np.newaxis] * Z, axis=0)
        current_norm = np.max(np.abs(V))
        
        # Local search loop
        # We use a simple hill climbing. If stuck, we break out and let the caller restart.
        # To avoid infinite loops in flat regions, we add a small tolerance.
        tol = 1e-12
        
        while True:
            improved = False
            for k in range(n_coeffs):
                delta = 2 * coeffs[k] * Z[k]
                V_new = V - delta
                new_norm = np.max(np.abs(V_new))
                
                if new_norm < current_norm - tol:
                    coeffs[k] = -coeffs[k]
                    V = V_new
                    current_norm = new_norm
                    improved = True
                    break # Restart inner loop
                    
            if not improved:
                break
                
        return coeffs, current_norm

    # Multiple random restarts
    # We can also try starting from the seed program's output
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    seed_coeffs = np.array(p[:24])
    
    # Try seed first
    res_coeffs, res_norm = local_search(seed_coeffs)
    if res_norm < best_norm:
        best_norm = res_norm
        best_coeffs = res_coeffs.copy()
        
    while time.time() - start_time < timeout:
        # Random restart
        coeffs = np.random.choice([-1, 1], size=n_coeffs)
        
        # Optional: Perturb the best solution found so far to explore its neighborhood
        if np.random.random() < 0.2 and best_coeffs is not None:
            coeffs = best_coeffs.copy()
            # Flip 3 random bits
            flip_indices = np.random.choice(n_coeffs, 3, replace=False)
            coeffs[flip_indices] = -coeffs[flip_indices]
            
        res_coeffs, res_norm = local_search(coeffs)
        
        if res_norm < best_norm:
            best_norm = res_norm
            best_coeffs = res_coeffs.copy()
            
        # Check time
        if time.time() - start_time > timeout - 2:
            break

    if best_coeffs is None:
        return p[:24]
        
    return best_coeffs.tolist()
