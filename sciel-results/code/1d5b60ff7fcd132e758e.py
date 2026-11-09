import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin from 150s limit
    
    rank_schoolbook = n * m * p  # 54
    
    # Schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    # Precompute target tensor indices
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i) else 0
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
    
    num_u = n * m  # 6
    num_v = m * p  # 27
    num_w = p * n  # 18
    
    # Precompute target lists for each u, v, w
    targets_u = [[] for _ in range(num_u)]
    targets_v = [[] for _ in range(num_v)]
    targets_w = [[] for _ in range(num_w)]
    
    for u, v, w in target_indices:
        targets_u[u].append((v, w))
        targets_v[v].append((u, w))
        targets_w[w].append((u, v))
    
    def compute_full_error(U, V, W):
        """Compute max absolute error over all target indices."""
        max_err = 0.0
        for u, v, w in target_indices:
            val = np.sum(U[u] * V[v] * W[w])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        return max_err
    
    def is_valid_decomposition(U, V, W, tol=1e-9):
        """Check if decomposition is exact within tolerance."""
        for u, v, w in target_indices:
            val = np.sum(U[u] * V[v] * W[w])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def try_round_and_verify(U, V, W, rank):
        """Try rounding to nearest half-integer and verify."""
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if is_valid_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
            return U_rounded, V_rounded, W_rounded, True
        
        # Also try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        
        if is_valid_decomposition(U_int, V_int, W_int, tol=1e-9):
            return U_int, V_int, W_int, True
        
        return None, None, None, False
    
    def als_step(U, V, W, max_iter=50, tol=1e-10):
        """Run ALS iterations to minimize error."""
        r = U.shape[1]
        
        for it in range(max_iter):
            # Update U
            new_U = np.zeros_like(U)
            A_flat = np.zeros((num_v * num_w, r))
            for col_r in range(r):
                outer = np.outer(V[:, col_r], W[:, col_r])
                A_flat[:, col_r] = outer.flatten()
            
            for u in range(num_u):
                b_u = np.zeros(num_v * num_w)
                for v, w in targets_u[u]:
                    b_u[v * num_w + w] = 1.0
                x, residuals, rank, sv = np.linalg.lstsq(A_flat, b_u, rcond=None)
                new_U[u] = x
            
            # Update V
            new_V = np.zeros_like(V)
            A_flat_v = np.zeros((num_u * num_w, r))
            for col_r in range(r):
                outer = np.outer(new_U[:, col_r], W[:, col_r])
                A_flat_v[:, col_r] = outer.flatten()
            
            for v in range(num_v):
                b_v = np.zeros(num_u * num_w)
                for u, w in targets_v[v]:
                    b_v[u * num_w + w] = 1.0
                x, residuals, rank, sv = np.linalg.lstsq(A_flat_v, b_v, rcond=None)
                new_V[v] = x
            
            # Update W
            new_W = np.zeros_like(W)
            A_flat_w = np.zeros((num_u * num_v, r))
            for col_r in range(r):
                outer = np.outer(new_U[:, col_r], new_V[:, col_r])
                A_flat_w[:, col_r] = outer.flatten()
            
            for w in range(num_w):
                b_w = np.zeros(num_u * num_v)
                for u, v in targets_w[w]:
                    b_w[u * num_v + v] = 1.0
                x, residuals, rank, sv = np.linalg.lstsq(A_flat_w, b_w, rcond=None)
                new_W[w] = x
            
            # Check convergence
            err = compute_full_error(new_U, new_V, new_W)
            U, V, W = new_U, new_V, new_W
            
            if err < tol:
                break
        
        return U, V, W, err
    
    def gradient_descent_polish(U, V, W, rank, max_iter=100, lr=0.01):
        """Polish solution using gradient descent on squared error."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for u, v, w in target_indices:
                val = np.sum(U[u] * V[v] * W[w])
                err = val - 1.0
                # d(err^2)/dU[u,r] = 2*err * V[v,r] * W[w,r]
                grad_U[u] += 2 * err * V[v] * W[w]
                grad_V[v] += 2 * err * U[u] * W[w]
                grad_W[w] += 2 * err * U[u] * V[v]
            
            # Update with gradient descent
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            err = compute_full_error(U, V, W)
            if err < 1e-12:
                break
        
        return U, V, W, err
    
    # Try ranks from 53 down to 40
    for r in range(53, 39, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        
        # Multiple initialization strategies
        init_strategies = []
        
        # Strategy 1: Random initialization
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            U = np.random.randn(num_u, r) * 0.5
            V = np.random.randn(num_v, r) * 0.5
            W = np.random.randn(num_w, r) * 0.5
            init_strategies.append((U, V, W, 'random'))
        
        # Strategy 2: Perturbed schoolbook (take first r columns)
        if r <= rank_schoolbook:
            U_init = U_school[:, :r].copy()
            V_init = V_school[:, :r].copy()
            W_init = W_school[:, :r].copy()
            # Add small perturbation
            U_init += np.random.randn(num_u, r) * 0.01
            V_init += np.random.randn(num_v, r) * 0.01
            W_init += np.random.randn(num_w, r) * 0.01
            init_strategies.append((U_init, V_init, W_init, 'perturbed_schoolbook'))
        
        # Strategy 3: Structured initialization based on symmetry
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            U = np.zeros((num_u, r))
            V = np.zeros((num_v, r))
            W = np.zeros((num_w, r))
            # Assign random ±1 values
            for col in range(r):
                U[:, col] = np.random.choice([-1, 1], size=num_u)
                V[:, col] = np.random.choice([-1, 1], size=num_v)
                W[:, col] = np.random.choice([-1, 1], size=num_w)
            init_strategies.append((U, V, W, 'structured'))
        
        for U_init, V_init, W_init, strat_name in init_strategies:
            if time.time() > deadline - 8:
                break
            
            # Run ALS
            U_als, V_als, W_als, err_als = als_step(U_init.copy(), V_init.copy(), W_init.copy(), 
                                                     max_iter=30, tol=1e-10)
            
            if err_als < 1e-8:
                # Try rounding
                U_r, V_r, W_r, success = try_round_and_verify(U_als, V_als, W_als, r)
                if success:
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = r
                    found = True
                    break
            
            # If ALS didn't converge well, try gradient descent polish
            if err_als < 0.5:  # Only polish if ALS made progress
                U_gd, V_gd, W_gd, err_gd = gradient_descent_polish(U_als, V_als, W_als, r,
                                                                    max_iter=50, lr=0.005)
                if err_gd < 1e-8:
                    U_r, V_r, W_r, success = try_round_and_verify(U_gd, V_gd, W_gd, r)
                    if success:
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        best_rank = r
                        found = True
                        break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
