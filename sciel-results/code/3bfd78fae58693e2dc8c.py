import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    def get_tensor_error(U, V, W, n, m, p, rank):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        S = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
        err = np.sum((S - 1.0)**2)
        return err, S

    def verify(U, V, W, n, m, p, rank, tol=1e-8):
        err, S = get_tensor_error(U, V, W, n, m, p, rank)
        return err < tol

    def try_rank_als(rank, seed, max_iter=200):
        """Alternating Least Squares with half-integer rounding"""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.3
        V = rng.randn(m * p, rank) * 0.3
        W = rng.randn(p * n, rank) * 0.3
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Fix V, W, solve for U
            # S[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
            # For fixed V, W: U[i,j,:] = argmin sum_k (S[i,j,k] - sum_t U[i,j,t] V[j,k,t] W[k,i,t])^2
            # This is a linear system: A[i,j] @ U[i,j,:] = b[i,j]
            # where A[i,j] = sum_k W[k,i,:] V[j,k,:]^T (outer product summed over k)
            # b[i,j] = sum_k W[k,i,:] V[j,k,:] (element-wise, but actually we need...)
            
            # Actually, let's do it differently. For each (i,j), we want:
            # sum_t U[i,j,t] * (V[j,k,t] * W[k,i,t]) = 1 for all k
            # Let M[i,j,k,t] = V[j,k,t] * W[k,i,t]
            # Then sum_t U[i,j,t] * M[i,j,k,t] = 1 for all k
            # This is M[i,j] @ U[i,j,:] = ones(p)
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            for i in range(n):
                for j in range(m):
                    # Build A: (p, rank), b: (p,)
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V3[j, k, :] * W3[k, i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        U3[i, j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        U3[i, j, :] = rng.randn(rank) * 0.1
            
            U = U3.reshape(n*m, rank)
            
            # Fix U, W, solve for V
            V3 = V.reshape(m, p, rank)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U3[i, j, :] * W3[k, i, :]
                    b = np.ones(n)
                    try:
                        V3[j, k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        V3[j, k, :] = rng.randn(rank) * 0.1
            
            V = V3.reshape(m*p, rank)
            
            # Fix U, V, solve for W
            W3 = W.reshape(p, n, rank)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U3[i, j, :] * V3[j, k, :]
                    b = np.ones(m)
                    try:
                        W3[k, i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        W3[k, i, :] = rng.randn(rank) * 0.1
            
            W = W3.reshape(p*n, rank)
            
            # Check if we're close to half-integers
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            if err < 1e-10:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r, n, m, p, rank, tol=1e-6):
                    return U_r, V_r, W_r
        
        return None

    def try_rank_random_search(rank, seed):
        """Random search with local refinement"""
        if time.time() > deadline - 15:
            return None
            
        rng = np.random.RandomState(seed)
        
        # Start with random half-integer values
        for attempt in range(50):
            if time.time() > deadline - 10:
                return None
            
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            err, S = get_tensor_error(U, V, W, n, m, p, rank)
            if err < 1e-8:
                return U, V, W
            
            # Local refinement: try flipping individual entries
            for flip_round in range(20):
                if time.time() > deadline - 10:
                    return None
                
                improved = False
                # Try flipping random entries
                for _ in range(100):
                    mat = rng.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        idx = rng.randint(n*m*rank)
                        row, col = idx // rank, idx % rank
                        old_val = U[row, col]
                        new_val = -old_val if old_val != 0 else 0.5
                        U[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-12:
                            err = err_new
                            improved = True
                        else:
                            U[row, col] = old_val
                    
                    elif mat == 'V':
                        idx = rng.randint(m*p*rank)
                        row, col = idx // rank, idx % rank
                        old_val = V[row, col]
                        new_val = -old_val if old_val != 0 else 0.5
                        V[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-12:
                            err = err_new
                            improved = True
                        else:
                            V[row, col] = old_val
                    
                    else:
                        idx = rng.randint(p*n*rank)
                        row, col = idx // rank, idx % rank
                        old_val = W[row, col]
                        new_val = -old_val if old_val != 0 else 0.5
                        W[row, col] = new_val
                        err_new, _ = get_tensor_error(U, V, W, n, m, p, rank)
                        if err_new < err - 1e-12:
                            err = err_new
                            improved = True
                        else:
                            W[row, col] = old_val
                
                if not improved:
                    break
                
                if err < 1e-8:
                    return U, V, W
        
        return None

    best_result = None
    
    # Try ranks from low to high
    for rank in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        # Try ALS
        for seed in range(10):
            if time.time() > deadline - 15:
                break
            res = try_rank_als(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
        
        if best_result:
            break
        
        # Try random search
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            res = try_rank_random_search(rank, seed)
            if res is not None:
                U, V, W = res
                best_result = {
                    "U": [[float(x) for x in row] for row in U],
                    "V": [[float(x) for x in row] for row in V],
                    "W": [[float(x) for x in row] for row in W]
                }
                break
        
        if best_result:
            break

    if best_result:
        return best_result

    # Fallback schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
