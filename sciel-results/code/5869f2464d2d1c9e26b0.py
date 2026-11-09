import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate a dense grid of test points for covering radius evaluation
    # Using Fibonacci spiral for uniform distribution
    M_test = 2000
    i_test = np.arange(M_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / M_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    Y = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # 2. Define Optimization Functions
    
    def get_initial_configurations():
        configs = []
        # Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        configs.append(X0)
        
        # Rotated Fibonacci spirals
        for _ in range(4):
            # Random rotation
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            # Rotation matrix
            R1 = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
            R2 = np.array([
                [np.cos(phi), 0, np.sin(phi)],
                [0, 1, 0],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            R = R2 @ R1
            X_rot = X0 @ R.T
            configs.append(X_rot)
            
        # Random configurations
        for _ in range(5):
            X_rand = np.random.randn(N, 3)
            norms = np.linalg.norm(X_rand, axis=1, keepdims=True)
            configs.append(X_rand / norms)
            
        return configs

    def normalize(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    def compute_rfd_gradient(X, tau=10.0):
        # Minimize -softmin(d_ij)
        # d_ij = arccos(x_i . x_j)
        dots = X @ X.T
        np.fill_diagonal(dots, 0.0)
        dots = np.clip(dots, -1.0, 1.0)
        
        d = np.arccos(dots)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d = d[mask]
        
        # Soft min: -1/tau * log(sum(exp(-tau * d)))
        # Gradient w.r.t X
        # We need sum_{j != i} exp(-tau * d_ij) * grad(d_ij)
        
        # Vectorized calculation is tricky, let's do a loop for gradient or use broadcasting
        # Broadcasting: dots (N, N)
        exp_terms = np.exp(-tau * d)
        # We need to map back to (N, N)
        exp_mat = np.zeros((N, N))
        exp_mat[mask] = exp_terms
        
        sum_exp = exp_mat.sum(axis=1, keepdims=True)
        
        # grad(d_ij) w.r.t x_i = -(x_j - (x_i.x_j)x_i) / sqrt(1 - (x_i.x_j)^2)
        # Let u = x_i . x_j
        # grad = -(x_j - u*x_i) / sqrt(1-u^2)
        
        # Calculate term for each i
        grad_X = np.zeros_like(X)
        for i in range(N):
            u = dots[i, :]
            u[i] = 0.0 # avoid self
            # sqrt(1-u^2)
            denom = np.sqrt(np.maximum(0, 1.0 - u*u))
            # Avoid division by zero
            denom[denom < 1e-6] = 1e-6
            
            # x_j - u*x_i
            # X (N,3), u (N,)
            # We want sum_j exp(-tau d_ij) * (-(x_j - u_ij x_i) / denom_ij)
            # Note: the negative sign is in the gradient of d, but we are minimizing -softmin, so 
            # d/dX (-softmin) = - d/dX(softmin) = - (1/sum) sum exp * grad(d)
            # Wait, E = - 1/tau log(sum exp(-tau d)).
            # dE/dX = -1/tau * (1/sum) * sum (-tau exp(-tau d) grad(d))
            # dE/dX = (1/sum) * sum exp(-tau d) grad(d)
            
            # grad(d_ij) w.r.t x_i = -(x_j - u x_i) / sqrt(1-u^2)
            # So we sum exp * (-(x_j - u x_i) / denom)
            
            # Vectorized:
            # x_i is X[i]
            # x_j are X
            # u is dots[i, :]
            # term_j = -(X[j] - u[j]*X[i]) / denom[j]
            
            # Let's compute this for all j at once
            # X (N,3)
            # u (N,)
            # X_i (3,)
            
            # x_j - u*x_i = X - np.outer(u, X[i])
            diff = X - np.outer(u, X[i])
            # Divide by denom
            diff /= denom[:, np.newaxis]
            # Multiply by exp_term
            weighted_diff = exp_mat[i, :][:, np.newaxis] * diff
            # Sum over j
            grad_X[i] = np.sum(weighted_diff, axis=0) / sum_exp[i, 0]
            
        return grad_X

    def compute_covering_gradient(X, Y, tau=20.0):
        # Minimize softmax(min_dist)
        # d_ji = arccos(y_j . x_i)
        # m_j = min_i d_ji
        # F = 1/tau log(sum_j exp(tau m_j))
        
        dots = Y @ X.T # (M, N)
        dots = np.clip(dots, -1.0, 1.0)
        d = np.arccos(dots) # (M, N)
        
        m = np.min(d, axis=1) # (M,)
        min_idx = np.argmin(d, axis=1) # (M,)
        
        exp_m = np.exp(tau * m)
        sum_exp = np.sum(exp_m)
        weights = exp_m / (tau * sum_exp) # (M,)
        
        # dF/dx_i = - sum_{j: min_idx[j]==i} weights[j] * y_j / sqrt(1 - (y_j . x_i)^2)
        
        grad_X = np.zeros_like(X)
        for i in range(N):
            mask = (min_idx == i)
            if not np.any(mask):
                continue
            y_sub = Y[mask]
            w_sub = weights[mask]
            
            # Calculate y_j . x_i for these j
            u = y_sub @ X[i]
            denom = np.sqrt(np.maximum(0, 1.0 - u*u))
            denom[denom < 1e-6] = 1e-6
            
            # grad term: - w_j * y_j / denom_j
            # Sum over j
            # y_sub (K, 3), w_sub (K,), denom (K,)
            coeff = -w_sub / denom
            grad_X[i] = np.sum(coeff[:, np.newaxis] * y_sub, axis=0)
            
        return grad_X

    def optimize_config(X_init, max_iter=500, deadline=None):
        X = X_init.copy()
        X = normalize(X)
        
        # Phase 1: RFD
        tau_rfd = 5.0
        lr = 0.05
        for k in range(max_iter // 2):
            if deadline and time.time() > deadline - 10:
                break
            grad = compute_rfd_gradient(X, tau_rfd)
            X = X - lr * grad
            X = normalize(X)
            
        # Phase 2: Covering
        tau_cov = 10.0
        lr = 0.01
        for k in range(max_iter // 2):
            if deadline and time.time() > deadline - 5:
                break
            grad = compute_covering_gradient(X, Y, tau_cov)
            X = X - lr * grad
            X = normalize(X)
            
        return X

    def evaluate_covering_radius(X, Y_eval=None):
        if Y_eval is None:
            Y_eval = Y
        dots = Y_eval @ X.T
        dots = np.clip(dots, -1.0, 1.0)
        d = np.arccos(dots)
        m = np.min(d, axis=1)
        return np.max(m) * 180.0 / np.pi

    # 3. Main Loop
    best_score = np.inf
    best_X = None
    
    configs = get_initial_configurations()
    
    for idx, X_init in enumerate(configs):
        if time.time() > deadline - 15:
            break
            
        X_opt = optimize_config(X_init, max_iter=200, deadline=deadline)
        
        # Evaluate
        score = evaluate_covering_radius(X_opt)
        
        if score < best_score:
            best_score = score
            best_X = X_opt
            
        # If we have a very good score, we can stop early?
        # Best known is 11.3165. If we get close, we might want to polish further.
        
    # Final Polish on the best candidate if time permits
    if best_X is not None and time.time() < deadline - 20:
        # Run a few more iterations with tighter tau
        X = best_X.copy()
        # High precision polish
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            grad = compute_covering_gradient(X, Y, tau=50.0)
            X = X - 0.005 * grad
            X = normalize(X)
        score = evaluate_covering_radius(X)
        if score < best_score:
            best_score = score
            best_X = X

    return best_X
