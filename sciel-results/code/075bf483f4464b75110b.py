import time
import numpy as np
import random

DIMENSIONS = [2, 3, 9]

def _validate(U, V, W, n=2, m=3, p=9, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                # For matrix multiplication, the tensor entry is 1 if the term is valid.
                # All terms are valid in dense MM.
                if abs(s - 1.0) > tol:
                    return False
    return True

def _optimize(U, V, W, n, m, p, max_iters=50, tol=1e-9):
    """
    Local optimization using gradient descent / Gauss-Newton approximation.
    Minimizes sum (sum_c U_ijc V_jkc W_kic - 1)^2
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape for easier computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    def compute_error():
        # Tensor product sum
        # E[i,j,k] = sum_c U[i,m,j,c] V[j,m,p,k,c] W[k,p,n,i,c]
        # Using einsum: 'ijc,jkc,kic->ijk'
        E = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        target = np.ones((n, m, p))
        diff = E - target
        return np.sum(diff**2)
    
    def compute_grads():
        E = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = E - np.ones((n, m, p))
        
        # dE/dU[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        # dE/dV[j,k,c] = sum_i diff[i,j,k] * U[i,j,c] * W[k,i,c]
        # dE/dW[k,i,c] = sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
        
        gU = np.einsum('ijk,jkc,kic->ijc', diff, V, W) * 2
        gV = np.einsum('ijk,ijc,kic->jkc', diff, U, W) * 2
        gW = np.einsum('ijk,ijc,jkc->kic', diff, U, V) * 2
        
        return gU, gV, gW

    lr = 0.01
    for it in range(max_iters):
        err = compute_error()
        if err < tol:
            break
        
        gU, gV, gW = compute_grads()
        
        # Simple gradient step
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Optional: Adaptive learning rate
        # if err > prev_err: lr *= 0.5
        # else: lr *= 1.1
        
    return U, V, W, compute_error()

def construct():
    t_start = time.time()
    DEADLINE = t_start + 115.0
    
    n, m, p = 2, 3, 9
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    np.random.seed(12345)
    
    # Try ranks from lower bound to 53
    # The directive suggests investigating rank 18.
    # We try a range of ranks.
    for r in range(18, 54):
        if time.time() > DEADLINE:
            break
            
        # Fewer attempts for higher ranks, more for lower?
        # Lower rank is harder, so maybe more attempts?
        # But optimization is expensive.
        max_attempts = 20 if r < 30 else 5
        
        for attempt in range(max_attempts):
            if time.time() > DEADLINE:
                break
                
            # Random Gaussian initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Optimize
            U_opt, V_opt, W_opt, err = _optimize(U, V, W, n, m, p, max_iters=100, tol=1e-8)
            
            # Check validity
            if err < 1e-8:
                # Convert to list of lists of floats (or ints if close)
                U_list = U_opt.tolist()
                V_list = V_opt.tolist()
                W_list = W_opt.tolist()
                
                # Try to round to integers or half-integers if close
                # The problem allows integer/half-integer entries.
                # If the solution is exact, it might be rational.
                # For now, return floats. The validator uses tolerance.
                
                if _validate(U_list, V_list, W_list, n, m, p, tol=1e-7):
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
                    
    return best_result
