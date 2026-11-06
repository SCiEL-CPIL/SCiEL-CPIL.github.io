import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Precompute all arithmetic progressions
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
    
    Np = len(progs)
    
    # Build incidence matrix M (n x Np)
    # M[i, p] = 1 if index i is in progression p
    M = np.zeros((n, Np), dtype=np.float64)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
    
    # We want to minimize max(|M^T x|) where x in {-1, 1}^n
    # Strategy: Continuous relaxation. Minimize sum((M^T x)^2) via gradient descent.
    # Gradient of 0.5 * ||M^T x||^2 w.r.t x is M (M^T x).
    # We project x back to {-1, 1} or use a soft-thresholding / sigmoid approach.
    # A more robust approach for this specific problem: 
    # Minimize f(x) = sum(exp(beta * |s_p|) - 1) or similar smooth max approximation?
    # Actually, minimizing the sum of squares of the sums is a standard proxy for minimizing the max sum.
    # Let's use a hybrid: Gradient descent on x (continuous), then threshold to signs.
    # To avoid getting stuck in trivial solutions (like all zeros if we allowed continuous), 
    # we enforce the norm or use a penalty.
    
    # Better: Use a "soft" sign function or just project to hypercube at each step.
    # Let's try: 
    # x_{k+1} = x_k - lr * grad
    # x_{k+1} = sign(x_{k+1})  (Projection to {-1, 1})
    # This is essentially a variant of the "power iteration" or "sign SGD".
    
    # Even better for this specific structure: 
    # The objective is non-convex. 
    # Let's use a multi-start gradient descent on the continuous variables, 
    # followed by a local search on the discrete signs.
    
    def compute_grad(x):
        sums = M.T @ x
        return M @ sums
    
    def compute_sums(x):
        return M.T @ x
    
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    rng = np.random.RandomState(42)
    
    best_signs = None
    best_score = np.inf
    
    # Phase 1: Continuous Gradient Descent with Projection
    # We will run multiple restarts with different initializations.
    n_restarts = 10
    steps_per_restart = 2000
    lr = 0.05  # Learning rate
    
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize x
        # Mix of random and structured
        if r == 0:
            x = rng.choice([-1.0, 1.0], size=n)
        elif r == 1:
            # Alternating
            x = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        elif r == 2:
            # Block based
            x = np.zeros(n)
            x[:n//4] = 1
            x[n//4:n//2] = -1
            x[n//2:3*n//4] = 1
            x[3*n//4:] = -1
        else:
            x = rng.randn(n)
            x = np.sign(x)
            if np.all(x == 0):
                x[0] = 1
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 20:
                break
                
            # Compute gradient
            sums = compute_sums(x)
            grad = M @ sums
            
            # Update
            x = x - lr * grad
            
            # Project to hypercube (sign function)
            # To allow exploration, we can use a soft sign or just hard sign.
            # Hard sign converges fast but can get stuck.
            # Let's use a threshold with a small noise or just hard sign.
            x = np.sign(x)
            # Ensure no zeros (if x becomes 0, pick random sign)
            zeros = np.where(x == 0)[0]
            if len(zeros) > 0:
                x[zeros] = rng.choice([-1.0, 1.0], size=len(zeros))
                
            # Check discrete score
            if step % 10 == 0:
                discrete_signs = x.astype(int)
                sums = compute_sums(discrete_signs)
                score = max_abs_sum(sums)
                if score < best_score:
                    best_score = score
                    best_signs = discrete_signs.copy()
        
        # Final check for this restart
        discrete_signs = x.astype(int)
        sums = compute_sums(discrete_signs)
        score = max_abs_sum(sums)
        if score < best_score:
            best_score = score
            best_signs = discrete_signs.copy()

    # Phase 2: Local Search on the best found signs
    # This is the same as the seed program's greedy polish, which is effective.
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_sums = compute_sums(current_signs)
        current_score = best_score
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(n):
                sign_i = current_signs[i]
                # Flipping sign i changes sums by -2 * sign_i * M[i, :]
                delta = -2.0 * sign_i * M[i, :]
                new_sums = current_sums + delta
                new_score = max_abs_sum(new_sums)
                
                if new_score < current_score:
                    current_signs[i] = -sign_i
                    current_sums = new_sums
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = current_signs.copy()
        
        # Phase 3: Simulated Annealing for fine-tuning if time permits
        if time.time() < deadline - 10:
            current_signs = best_signs.copy()
            current_sums = compute_sums(current_signs)
            current_score = best_score
            
            T_start = 1.0
            T_end = 0.01
            n_iters = 5000
            
            for it in range(n_iters):
                if time.time() > deadline - 5:
                    break
                    
                T = T_start * (1.0 - it / float(n_iters)) + T_end
                
                i = rng.randint(n)
                sign_i = current_signs[i]
                delta = -2.0 * sign_i * M[i, :]
                new_sums = current_sums + delta
                new_score = max_abs_sum(new_sums)
                
                delta_score = new_score - current_score
                
                if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
                    current_signs[i] = -sign_i
                    current_sums = new_sums
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_signs = current_signs.copy()

    # Ensure we return a valid list of ints
    if best_signs is None:
        return [1] * n
        
    return [int(x) for x in best_signs]
