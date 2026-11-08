import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _build_target_tensor(n=3, m=3, p=6):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    return T

def _verify_fast(U, V, W, n=3, m=3, p=6, atol=1e-9):
    """Fast verification using einsum."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
    T = _build_target_tensor(n, m, p)
    return np.allclose(T_decomp, T, atol=atol)

def _verify_entries(U, V, W, n=3, m=3, p=6, atol=1e-9):
    """Verify by checking each entry directly (faster for sparse checks)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                if abs(val - 1.0) > atol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_gradient_descent(n, m, p, rank, T, max_iters=5000, lr=0.01, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute T_decomp
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        err = np.linalg.norm(T_decomp - T)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            return best_U, best_V, best_W
        
        # Compute gradients
        # dL/dU[b,r] = sum_{a,c} (T_decomp[a,b,c] - T[a,b,c]) * V[c,r] * W[a,r]
        # dL/dV[c,r] = sum_{a,b} (T_decomp[a,b,c] - T[a,b,c]) * U[b,r] * W[a,r]
        # dL/dW[a,r] = sum_{b,c} (T_decomp[a,b,c] - T[a,b,c]) * U[b,r] * V[c,r]
        
        residual = T_decomp - T
        
        dU = np.einsum('abc,cr,ar->br', residual, V, W)
        dV = np.einsum('abc,br,ar->cr', residual, U, W)
        dW = np.einsum('abc,br,cr->ar', residual, U, V)
        
        # Update with adaptive learning rate
        scale = min(lr, 0.1 / (1 + err))
        U -= scale * dU
        V -= scale * dV
        W -= scale * dW
        
        # Periodic random restart to escape local minima
        if it % 1000 == 999 and err > 1.0:
            U += rng.randn(n * m, rank) * 0.1
            V += rng.randn(m * p, rank) * 0.1
            W += rng.randn(p * n, rank) * 0.1
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def _try_column_merge(U, V, W, n, m, p, T, max_rounds=200):
    """Try to reduce rank by merging or removing columns."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for round_num in range(max_rounds):
        if rank <= max(n*m, m*p, p*n):
            break
        removed = False
        for c in range(rank):
            # Try removing column c
            U_new = np.delete(U, c, axis=1)
            V_new = np.delete(V, c, axis=1)
            W_new = np.delete(W, c, axis=1)
            if _verify_entries(U_new, V_new, W_new, n, m, p, atol=1e-7):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed = True
                break
        
        if not removed:
            # Try merging: replace two columns with their sum
            for c1 in range(rank - 1):
                for c2 in range(c1 + 1, rank):
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    # Merge c2 into c1
                    U_new[:, c1] += U_new[:, c2]
                    V_new[:, c1] += V_new[:, c2]
                    W_new[:, c1] += W_new[:, c2]
                    U_new = np.delete(U_new, c2, axis=1)
                    V_new = np.delete(V_new, c2, axis=1)
                    W_new = np.delete(W_new, c2, axis=1)
                    if _verify_entries(U_new, V_new, W_new, n, m, p, atol=1e-7):
                        U, V, W = U_new, V_new, W_new
                        rank -= 1
                        removed = True
                        break
                if removed:
                    break
        if not removed:
            break
    
    return U, V, W, rank

def _try_structured_patterns(n, m, p, T, deadline):
    """Try known structured patterns for low-rank decomposition."""
    results = []
    
    # Pattern: Block decomposition
    # For 3x3 x 3x6, think of B as [B1, B2] where B1, B2 are 3x3
    # C = A*B = [A*B1, A*B2]
    # If we have a rank-r decomposition for 3x3*3x3, we can use it for both blocks
    # with some sharing
    
    # Try rank = 2 * (rank for 3x3*3x3) - sharing
    # Best known for 3x3*3x3 is 23 (Strassen-like)
    
    for target_rank in range(30, 54):
        if time.time() > deadline - 5:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = _try_gradient_descent(n, m, p, target_rank, T, 
                                           max_iters=3000, lr=0.005, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p, atol=1e-8):
                    results.append((target_rank, U, V, W))
                    found = True
                    break
        if found:
            break
    
    return results

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    best_rank = schoolbook_rank
    
    # Strategy 1: Try gradient descent from various ranks
    # Start from a reasonable lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = 18
    # But realistically, for 3x3*3x6, try from 35 upward
    
    search_ranks = list(range(35, 54))
    
    for rank in search_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_gradient_descent(n, m, p, rank, T,
                                           max_iters=4000, lr=0.003, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p, atol=1e-8):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Try to reduce further with column operations
            U2, V2, W2, rank2 = _try_column_merge(best_U, best_V, best_W, n, m, p, T)
            if rank2 < best_rank:
                best_U, best_V, best_W = U2, V2, W2
                best_rank = rank2
            continue
    
    # Strategy 2: Start from schoolbook and try to reduce
    if time.time() < deadline - 20:
        U_s, V_s, W_s = _schoolbook(n, m, p)
        U_r, V_r, W_r, rank_r = _try_column_merge(U_s, V_s, W_s, n, m, p, T, max_rounds=100)
        if rank_r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r
    
    # Final verification
    if not _verify_fast(best_U, best_V, best_W, n, m, p, atol=1e-6):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = schoolbook_rank
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
