"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k]."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_decompose(T, rank, n, m, p, seed=42, max_iter=50):
    """
    Alternating Least Squares for tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                # T[i,j,:] = sum_c U[i*m+j,c] * (V[j*p+0:p, c] * W[:, k*n+i] for each k)
                # For fixed i,j: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Let A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                # Then T[i,j,:] = U[i*m+j,:] @ A
                A = np.zeros((rank, p))
                for k in range(p):
                    A[:, k] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j,:] @ A = T[i,j,:]
                # U[i*m+j,:] = T[i,j,:] @ pinv(A)
                if rank <= p:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, T[i,j,:], rcond=None)[0]
                else:
                    U[i*m+j, :] = np.linalg.lstsq(A.T, T[i,j,:], rcond=None)[0]
        
        # Update V
        for j in range(m):
            for k in range(p):
                # T[:,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed j,k: T[i,j,k] = V[j*p+k,c] * sum_c' U[i*m+j,c'] * W[k*n+i,c']
                # Let B[c,i] = U[i*m+j,c] * W[k*n+i,c]
                # Then T[:,j,k] = V[j*p+k,:] @ B
                B = np.zeros((rank, n))
                for i in range(n):
                    B[:, i] = U[i*m+j, :] * W[k*n+i, :]
                V[j*p+k, :] = np.linalg.lstsq(B.T, T[:,j,k], rcond=None)[0]
        
        # Update W
        for k in range(p):
            for i in range(n):
                # T[i,:,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # For fixed k,i: T[i,j,k] = W[k*n+i,c] * sum_c' U[i*m+j,c'] * V[j*p+k,c']
                # Let C[c,j] = U[i*m+j,c] * V[j*p+k,c]
                # Then T[i,:,k] = W[k*n+i,:] @ C
                C = np.zeros((rank, m))
                for j in range(m):
                    C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                W[k*n+i, :] = np.linalg.lstsq(C.T, T[i,j,k], rcond=None)[0]
    
    # Compute error
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pred = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (pred - T[i,j,k])**2
    err = np.sqrt(err)
    
    return U, V, W, err

def _check_exact(U, V, W, n, m, p, tol=1e-6):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _to_half_int_list(arr):
    result = []
    for row in arr:
        new_row = [round(float(x) * 2) / 2.0 for x in row]
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    T = _build_tensor(n, m, p)
    
    schoolbook = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks
    for rank in [32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]:
        if time.time() > deadline - 20:
            break
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            U, V, W, err = _als_decompose(T, rank, n, m, p, seed=seed, max_iter=30)
            if err < 1e-6:
                if _check_exact(U, V, W, n, m, p, tol=1e-6):
                    return {"U": _to_half_int_list(U), "V": _to_half_int_list(V), "W": _to_half_int_list(W)}
    
    return schoolbook
# EVOLVE-BLOCK-END

