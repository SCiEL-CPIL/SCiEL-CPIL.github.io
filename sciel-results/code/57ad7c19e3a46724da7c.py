import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
    ]
    
    M = 2**15
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def evaluate(coeffs):
        F = D @ coeffs
        return np.max(np.abs(F)) / np.sqrt(24)
        
    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_max_val = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
                    best_max_val = max_val
                    best_flip = n
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = D @ current_c
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max
    
    def local_search_2flip(current_c, current_f, current_max):
        """Try 2-flips to escape local minima."""
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
        return current_c, current_f, current_max
    
    def continuous_relaxation(x0, max_iter=5000, lr=0.1):
        """
        Continuous relaxation: optimize over real-valued x in [-1,1]
        using gradient descent on soft-max surrogate of sup-norm.
        """
        x = x0.copy()
        # Soft-max parameter (larger = closer to true max)
        beta = 50.0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Compute F = D @ x
            F = D @ x
            abs_F = np.abs(F)
            
            # Soft-max: use log-sum-exp for stability
            # soft_max = log(sum(exp(beta * abs_F))) / beta
            # gradient wrt x_k = (1/beta) * sum_m (beta * exp(beta*abs_F_m) / sum(exp(beta*abs_F))) * d(abs_F_m)/dx_k
            # d(abs_F_m)/dx_k = Re(conj(F_m) * D[m,k]) / abs_F_m (when abs_F_m > 0)
            
            # Numerical stability for log-sum-exp
            max_abs = np.max(abs_F)
            exp_vals = np.exp(beta * (abs_F - max_abs))
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp  # softmax weights
            
            # Gradient of soft-max wrt x
            # soft_max = max_abs + log(sum_exp) / beta
            # d(soft_max)/dx_k = (1/beta) * sum_m weights_m * d(abs_F_m)/dx_k
            grad = np.zeros(24, dtype=complex)
            for m in range(M):
                if abs_F[m] > 1e-10:
                    d_abs_F_dx = np.conj(F[m]) * D[m, :] / abs_F[m]
                    grad += weights[m] * d_abs_F_dx
            
            grad_real = np.real(grad)
            
            # Add penalty to keep x in [-1,1]: penalty = lambda * sum((x^2 - 1)^2) for |x| > 1
            # Or simpler: project after each step
            x -= lr * grad_real
            
            # Project to [-1, 1]
            x = np.clip(x, -1.0, 1.0)
            
            # Check convergence
            if it % 100 == 0:
                max_val = np.max(abs_F)
                if max_val / np.sqrt(24) < 1.26:
                    break
        
        return x
    
    def project_to_binary(x):
        """Project real-valued x to nearest binary {+1, -1}"""
        return np.sign(x)
        # Handle zeros: use 1.0
        # Actually np.sign(0) = 0, so we need to handle this
        # Better: x > 0 ? 1 : -1
        return np.where(x >= 0, 1.0, -1.0)
    
    best_score = float('inf')
    best_c = None
    
    # Phase 1: Evaluate all candidates and do local search on each
    for c in candidates:
        if time.time() > deadline:
            break
        c_arr = np.array(c, dtype=float)
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        # Do 1-flip local search
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        # Try 2-flip escapes, then re-do 1-flip
        for _ in range(5):
            if time.time() > deadline:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        score = max_val / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
    
    # Phase 2: Continuous relaxation from best_c and other candidates
    rng = np.random.RandomState(42)
    
    # Try continuous relaxation from multiple starting points
    start_points = [best_c.copy()]
    # Add some random perturbations of best_c
    for _ in range(10):
        if time.time() > deadline:
            break
        perturbed = best_c.copy()
        n_flips = rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
        indices = rng.choice(24, n_flips, replace=False)
        perturbed[indices] *= -1.0
        start_points.append(perturbed)
    
    for start_x in start_points:
        if time.time() > deadline:
            break
        
        # Run continuous relaxation
        x_relaxed = continuous_relaxation(start_x, max_iter=3000, lr=0.05)
        
        # Project to binary
        c_proj = project_to_binary(x_relaxed)
        f_proj = D @ c_proj
        max_val = np.max(np.abs(f_proj))
        
        # Do local search on the projected solution
        c_proj, f_proj, max_val = local_search_1flip(c_proj, f_proj, max_val)
        for _ in range(3):
            if time.time() > deadline:
                break
            c_proj, f_proj, max_val = local_search_2flip(c_proj, f_proj, max_val)
            c_proj, f_proj, max_val = local_search_1flip(c_proj, f_proj, max_val)
        
        score = max_val / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_c = c_proj.copy()
    
    # Phase 3: Iterated local search with random restarts
    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f))
    
    while time.time() < deadline:
        # Try 1-flip
        best_flip_1 = -1
        best_max_1 = current_max
        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new))
            if max_val < best_max_1 - 1e-12:
                best_max_1 = max_val
                best_flip_1 = n
        
        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Try 2-flip
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
            continue
        
        # Stuck: random restart by flipping 2-3 random bits
        n_flips = rng.choice([2, 3], p=[0.6, 0.4])
        indices = rng.choice(24, n_flips, replace=False)
        current_c[indices] *= -1.0
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        # Do a quick 1-flip local search after restart
        current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
        if current_max / np.sqrt(24) < best_score:
            best_score = current_max / np.sqrt(24)
            best_c = current_c.copy()
    
    return [int(x) for x in best_c]
