import numpy as np
import time
import random

def construct():
    # Deadline guard: 145 seconds (5s margin from 150s limit)
    deadline = time.time() + 145.0
    
    n_coeffs = 24
    sqrt_n = np.sqrt(n_coeffs)
    
    # Initial best: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < n_coeffs:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:n_coeffs], dtype=float)
    
    # Precompute grid for fast evaluation
    # 2000 points is a good balance for speed and accuracy in local search
    n_grid = 2000
    thetas_search = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    z_search = np.exp(1j * thetas_search)
    powers_search = z_search[:, None] ** np.arange(n_coeffs)[None, :]
    
    def fast_eval(coeffs):
        P_vals = powers_search @ coeffs
        return np.max(np.abs(P_vals))
    
    # Initial evaluation
    current_mag = fast_eval(best_coeffs)
    best_mag = current_mag
    best_coeffs = best_coeffs.copy()
    
    random.seed(42)
    
    # Multi-start local search
    # We perform several restarts. Each restart does local search.
    # We keep the best result found so far.
    
    # To speed up, we can limit the number of steps per restart
    # and rely on the number of restarts.
    
    max_restarts = 1000
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # 1. Initialize candidate
        if restart == 0:
            c = best_coeffs.copy()
        else:
            # Random initialization
            # Using vectorized random generation
            c = np.where(np.random.rand(n_coeffs) > 0.5, 1.0, -1.0)
            
        c_mag = fast_eval(c)
        
        # 2. Local Search (Hill Climbing)
        # Try single flips
        improved = True
        local_steps = 0
        max_local_steps = 500
        while improved and local_steps < max_local_steps and time.time() < deadline:
            improved = False
            local_steps += 1
            
            # Shuffle indices to avoid bias
            indices = np.arange(n_coeffs)
            np.random.shuffle(indices)
            
            for i in indices:
                old_val = c[i]
                c[i] = -old_val
                new_mag = fast_eval(c)
                if new_mag < c_mag:
                    c_mag = new_mag
                    improved = True
                else:
                    c[i] = old_val
        
        # 3. Try pair flips to escape local minima
        # Do a few rounds of random pair flips
        for _ in range(50):
            if time.time() >= deadline: break
            i = np.random.randint(n_coeffs)
            j = np.random.randint(n_coeffs)
            if i == j: continue
            
            c[i] *= -1
            c[j] *= -1
            new_mag = fast_eval(c)
            if new_mag < c_mag:
                c_mag = new_mag
            else:
                c[i] *= -1
                c[j] *= -1
        
        # 4. Update best
        if c_mag < best_mag:
            best_mag = c_mag
            best_coeffs = c.copy()
            
        restart += 1
        
    # Final refinement of the best candidate
    # Use a much denser grid and golden section search on the peak
    def refine_sup_norm(coeffs):
        # Find peak on dense grid
        n_dense = 10000
        thetas = np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
        z = np.exp(1j * thetas)
        powers = z[:, None] ** np.arange(n_coeffs)[None, :]
        P_vals = powers @ coeffs
        mags = np.abs(P_vals)
        max_idx = np.argmax(mags)
        max_val = mags[max_idx]
        
        # Refine around max_idx
        theta_l = thetas[max_idx-1]
        theta_r = thetas[(max_idx+1) % n_dense]
        
        gr = (np.sqrt(5) - 1) / 2
        
        def obj(theta):
            z_val = np.exp(1j * theta)
            k = np.arange(n_coeffs)
            P_val = np.sum(coeffs * (z_val ** k))
            return np.abs(P_val)
            
        x1 = theta_r - gr * (theta_r - theta_l)
        x2 = theta_l + gr * (theta_r - theta_l)
        f1 = obj(x1)
        f2 = obj(x2)
        
        for _ in range(50):
            if f1 < f2:
                theta_r = x2
                x2 = x1
                f2 = f1
                x1 = theta_r - gr * (theta_r - theta_l)
                f1 = obj(x1)
            else:
                theta_l = x1
                x1 = x2
                f1 = f2
                x2 = theta_l + gr * (theta_r - theta_l)
                f2 = obj(x2)
                
        return max(max_val, f1, f2)

    # We don't strictly need to return the refined score, just the coefficients.
    # But it's good to ensure we are returning the best one.
    # The evaluator will re-calculate the score.
    
    return best_coeffs.tolist()
