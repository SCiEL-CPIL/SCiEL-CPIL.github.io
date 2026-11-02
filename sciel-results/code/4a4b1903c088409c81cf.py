import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Check if entries are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_vectorized(rank, time_limit=5.0, max_restarts=10):
    """
    Try to find a valid decomposition of the given rank using vectorized local search.
    """
    n, m, p = 3, 3, 6
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small random integers or half-integers
        rng = np.random.RandomState(restart + 12345)
        U = rng.choice([0, 0.5, -0.5, 1, -1], size=(n*m, rank))
        V = rng.choice([0, 0.5, -0.5, 1, -1], size=(m*p, rank))
        W = rng.choice([0, 0.5, -0.5, 1, -1], size=(p*n, rank))
        
        lr = 0.01
        best_loss = float('inf')
        
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Compute residuals vectorized
            # Residual[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
            residual = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, :] is (rank,)
                    # V[j*p: (j+1)*p, :] is (p, rank)
                    # W[:, k*n+i] is (p,)
                    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is element-wise product sum over r.
                    # Let's compute for all k at once for fixed i, j.
                    # U_vec = U[i*m+j, :] (rank,)
                    # V_mat = V[j*p:(j+1)*p, :] (p, rank)
                    # W_vec_k = W[:, k*n+i] (p,) -> need to broadcast
                    # Result[k] = sum_r U_vec[r] * V_mat[k, r] * W_vec[k, r]
                    # This is not a simple dot product because of W_vec[k, r] dependence on k.
                    # We can use einsum or explicit loop over k if p is small.
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        residual[i, j, k] = s - 1.0
            
            loss = np.sum(residual ** 2)
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-12:
                    break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r_ijk = residual[i, j, k]
                        # dU[i*m+j, r] += 2 * r_ijk * V[j*p+k, r] * W[k*n+i, r]
                        dU[i*m+j, :] += 2 * r_ijk * (V[j*p+k, :] * W[k*n+i, :])
                        dV[j*p+k, :] += 2 * r_ijk * (U[i*m+j, :] * W[k*n+i, :])
                        dW[k*n+i, :] += 2 * r_ijk * (U[i*m+j, :] * V[j*p+k, :])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            lr *= 0.95
            
            # Optional: snap to nearest half-integer every few steps to encourage discrete solution
            if it % 10 == 9:
                U = np.round(U * 2) / 2.0
                V = np.round(V * 2) / 2.0
                W = np.round(W * 2) / 2.0
        
        if best_loss < 1e-9:
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Leave 5s margin
    
    # Try to find lower ranks
    # Start from 53 and go down
    # Given the difficulty, we might only check a few
    for rank in range(53, 30, -1):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Allocate time based on how many ranks are left
        # Simple allocation: fixed 5s per rank attempt
        result = _try_rank_vectorized(rank, time_limit=min(5.0, remaining_time - 1.0), max_restarts=10)
        
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook()
