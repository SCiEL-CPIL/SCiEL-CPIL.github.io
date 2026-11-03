import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Build target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=float)
    V_fb = np.zeros((m*p, rank_fb), dtype=float)
    W_fb = np.zeros((p*n, rank_fb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb

    def compute_decomp(U, V, W):
        return np.einsum('at,bt,ct->abc', U, V, W, optimize=True)

    def check_decomp(U, V, W, atol=1e-9):
        T_decomp = compute_decomp(U, V, W)
        return np.allclose(T_decomp, T, atol=atol)

    def snap_to_half(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        return U_s, V_s, W_s

    def snap_to_quarter(U, V, W):
        U_s = np.round(U * 4) / 4
        V_s = np.round(V * 4) / 4
        W_s = np.round(W * 4) / 4
        return U_s, V_s, W_s

    def vectorized_als(U, V, W, rank, max_iters=200, tol=1e-12, verbose=False):
        """Fast vectorized alternating least squares."""
        size_u, size_v, size_w = n*m, m*p, p*n
        
        # Precompute the flattened target slices
        T_u = T.reshape(size_u, size_v * size_w)  # each row is T[a,:,:] flattened
        T_v = T.transpose(1,0,2).reshape(size_v, size_u * size_w)  # each row is T[:,b,:] flattened
        T_w = T.transpose(2,0,1).reshape(size_w, size_u * size_v)  # each row is T[:,:,c] flattened
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, solve (V^T ⊗ W^T) * U[a,:] = T[a,:,:]
            # The system is: sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
            # For fixed a: sum_t U[a,t] * (V[:,t]*W[:,t]) = T[a,:,:]
            # So the matrix is: for each t, column = V[:,t]*W[:,t] reshaped
            # A_U has shape (size_v*size_w, rank), column t = V[:,t]*W[:,t]
            A_U = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
            # Solve: A_U @ U[a,:] = T_u[a,:] for all a
            # This is a batched least squares: U = lstsq(A_U, T_u)
            try:
                U, residuals, rank_u, svals = np.linalg.lstsq(A_U, T_u, rcond=None)
            except:
                pass
            
            # Update V
            A_V = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
            try:
                V, residuals, rank_v, svals = np.linalg.lstsq(A_V, T_v, rcond=None)
            except:
                pass
            
            # Update W
            A_W = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
            try:
                W, residuals, rank_w, svals = np.linalg.lstsq(A_W, T_w, rcond=None)
            except:
                pass
            
            # Check convergence
            T_decomp = compute_decomp(U, V, W)
            err = np.sum((T - T_decomp) ** 2)
            if verbose:
                print(f"  ALS iter {it}: err={err:.2e}")
            if err < tol:
                break
            if abs(prev_err - err) < tol * 0.01:
                break
            prev_err = err
        
        return U, V, W

    def polish(U, V, W, rank, max_iters=50):
        """High-precision polish pass with tighter tolerance."""
        size_u, size_v, size_w = n*m, m*p, p*n
        T_u = T.reshape(size_u, size_v * size_w)
        T_v = T.transpose(1,0,2).reshape(size_v, size_u * size_w)
        T_w = T.transpose(2,0,1).reshape(size_w, size_u * size_v)
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            A_U = np.einsum('bt,ct->bct', V, W).reshape(size_v * size_w, rank)
            try:
                U, _, _, _ = np.linalg.lstsq(A_U, T_u, rcond=1e-15)
            except:
                pass
            
            A_V = np.einsum('at,ct->act', U, W).reshape(size_u * size_w, rank)
            try:
                V, _, _, _ = np.linalg.lstsq(A_V, T_v, rcond=1e-15)
            except:
                pass
            
            A_W = np.einsum('at,bt->abt', U, V).reshape(size_u * size_v, rank)
            try:
                W, _, _, _ = np.linalg.lstsq(A_W, T_w, rcond=1e-15)
            except:
                pass
            
            T_decomp = compute_decomp(U, V, W)
            err = np.sum((T - T_decomp) ** 2)
            if err < 1e-20:
                break
        
        return U, V, W

    def try_rank_als(rank, num_trials=30):
        """Try to find a rank-r decomposition using fast vectorized ALS."""
        size_u, size_v, size_w = n*m, m*p, p*n
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Multiple initialization strategies
            if trial % 5 == 0:
                scale = 0.3
            elif trial % 5 == 1:
                scale = 0.5
            elif trial % 5 == 2:
                scale = 0.7
            elif trial % 5 == 3:
                scale = 1.0
            else:
                scale = 0.4
            
            np.random.seed(trial * 7919 + rank * 31)
            U = np.random.randn(size_u, rank) * scale
            V = np.random.randn(size_v, rank) * scale
            W = np.random.randn(size_w, rank) * scale
            
            # Run fast ALS
            U, V, W = vectorized_als(U, V, W, rank, max_iters=100, tol=1e-14)
            
            # Check if we have an exact solution
            if check_decomp(U, V, W, atol=1e-10):
                # Polish for higher precision
                U, V, W = polish(U, V, W, rank, max_iters=20)
                
                if check_decomp(U, V, W, atol=1e-9):
                    # Try snapping to half-integers
                    U_s, V_s, W_s = snap_to_half(U, V, W)
                    if check_decomp(U_s, V_s, W_s, atol=1e-9):
                        return U_s, V_s, W_s
                    else:
                        U_s, V_s, W_s = snap_to_quarter(U, V, W)
                        if check_decomp(U_s, V_s, W_s, atol=1e-9):
                            return U_s, V_s, W_s
                        else:
                            return U, V, W
        
        return None

    def try_rank_structured(rank, num_trials=20):
        """Try structured initializations based on tensor symmetry."""
        size_u, size_v, size_w = n*m, m*p, p*n
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(50000 + trial * 103 + rank * 7)
            
            # Strategy: initialize with some zeros to exploit sparsity
            U = np.random.randn(size_u, rank) * 0.5
            V = np.random.randn(size_v, rank) * 0.5
            W = np.random.randn(size_w, rank) * 0.5
            
            # Add some structure: make some entries zero
            if trial % 4 == 1:
                U[np.random.rand(size_u, rank) < 0.3] = 0
            elif trial % 4 == 2:
                V[np.random.rand(size_v, rank) < 0.3] = 0
            elif trial % 4 == 3:
                W[np.random.rand(size_w, rank) < 0.3] = 0
            
            # Run fast ALS
            U, V, W = vectorized_als(U, V, W, rank, max_iters=100, tol=1e-14)
            
            if check_decomp(U, V, W, atol=1e-10):
                U, V, W = polish(U, V, W, rank, max_iters=20)
                if check_decomp(U, V, W, atol=1e-9):
                    U_s, V_s, W_s = snap_to_half(U, V, W)
                    if check_decomp(U_s, V_s, W_s, atol=1e-9):
                        return U_s, V_s, W_s
                    else:
                        return U, V, W
        
        return None

    # Try ranks from 26 to 35
    for rank in range(26, 36):
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        # Try structured approach first
        result = try_rank_structured(rank, num_trials=15)
        if result is None:
            # Then try generic ALS
            result = try_rank_als(rank, num_trials=25)
        
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r)
            best_rank = rank
            if rank == 26:
                break

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
