import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def als_step(U, V, W, target_rank, T):
        """One ALS iteration: update U, V, W in sequence."""
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    A_T = A.T
                    U_new[i * m + j, :] = np.linalg.lstsq(A_T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    B_T = B.T
                    V_new[j * p + k, :] = np.linalg.lstsq(B_T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    C_T = C.T
                    W_new[k * n + i, :] = np.linalg.lstsq(C_T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                except:
                    pass
        
        return U_new, V_new, W_new
    
    def gradient_descent(U, V, W, target_rank, max_iters=200, lr=0.01, seed=None):
        """Gradient descent refinement with adaptive learning rate."""
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            U_r = U.reshape(n, m, target_rank)
            V_r = V.reshape(m, p, target_rank)
            W_r = W.reshape(p, n, target_rank)
            
            result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = T - result
            
            # Gradients
            grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
            grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
            grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
            
            # Adaptive learning rate based on gradient norm
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm < 1e-12:
                break
            
            # Scale learning rate inversely with gradient norm
            effective_lr = lr * 10.0 / (1.0 + grad_norm)
            
            # Update
            U -= effective_lr * grad_U.reshape(n * m, target_rank)
            V -= effective_lr * grad_V.reshape(m * p, target_rank)
            W -= effective_lr * grad_W.reshape(p * n, target_rank)
            
            err = compute_error(U, V, W, target_rank)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
                
        err = compute_error(best_U, best_V, best_W, target_rank)
        return best_U, best_V, best_W, err
    
    def try_rank(target_rank):
        """Try to find a valid rank-target_rank decomposition."""
        max_restarts = 120
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return False
            
            seed = restart * 1000 + target_rank * 7
            rng = np.random.RandomState(seed)
            
            # Multiple initialization strategies
            init_strategy = restart % 3
            
            if init_strategy == 0:
                # Random initialization
                U = rng.randn(n * m, target_rank) * 0.5
                V = rng.randn(m * p, target_rank) * 0.5
                W = rng.randn(p * n, target_rank) * 0.5
            elif init_strategy == 1:
                # Sparse initialization with some structure
                U = rng.randn(n * m, target_rank) * 0.3
                V = rng.randn(m * p, target_rank) * 0.3
                W = rng.randn(p * n, target_rank) * 0.3
                # Add some deterministic structure
                for r in range(min(3, target_rank)):
                    U[0, r] += 1.0
                    V[0, r] += 1.0
                    W[0, r] += 1.0
            else:
                # Initialize based on schoolbook, then perturb
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                U += rng.randn(n * m, target_rank) * 0.2
                V += rng.randn(m * p, target_rank) * 0.2
                W += rng.randn(p * n, target_rank) * 0.2
            
            # Normalize columns to avoid scale issues
            for r in range(target_rank):
                norms = [np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r])]
                if any(norm < 1e-10 for norm in norms):
                    continue
                total_norm = np.prod(norms) ** (1/3)
                if total_norm > 1e-10:
                    U[:, r] /= (norms[0] / total_norm)
                    V[:, r] /= (norms[1] / total_norm)
                    W[:, r] /= (norms[2] / total_norm)
            
            # Phase 1: ALS iterations
            for als_iter in range(80):
                if time.time() - start_time > deadline:
                    return False
                U, V, W = als_step(U, V, W, target_rank, T)
                err = compute_error(U, V, W, target_rank)
                if err < 1e-10:
                    break
            
            # Phase 2: Gradient descent polish with multiple learning rates
            for lr in [0.01, 0.005, 0.001]:
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=150, lr=lr)
                if err is not None and err < 1e-8:
                    break
            
            if err is not None and err < 1e-6:
                # Phase 3: Fine polish with smaller learning rate
                U, V, W, err = gradient_descent(U, V, W, target_rank, max_iters=300, lr=0.0005)
                
                if err is not None and err < 1e-8:
                    # Round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded
                    err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                    if err_rounded < 1e-6:
                        nonlocal best_U, best_V, best_W, best_rank
                        best_U = U_rounded
                        best_V = V_rounded
                        best_W = W_rounded
                        best_rank = target_rank
                        return True
        return False
    
    # Try ranks from 14 to 17 (18 is schoolbook)
    for target_rank in range(14, R_school):
        if try_rank(target_rank):
            break
    
    # Final verification
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err
    
    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
