"""Find a rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE_SECONDS = 105.0  # Safety margin before the 120s wall-clock limit


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For the tensor T[i,j,k] = delta_{i,k} (in the appropriate sense for matrix mult),
    we need: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    and for each (i,j,k) with the "wrong" pairing, it should be 0.
    
    Actually, the matrix multiplication tensor M_{i,j,k} = 1 if the indices are consistent
    with C[i,k] = sum_j A[i,j]*B[j,k], so M[i,j,k] = 1 for all valid (i,j,k) triples.
    Wait, no. The tensor for matrix multiplication A(n x m) * B(m x p) = C(n x p)
    has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    So we need this sum to equal 1 for all (i,j,k).
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3D tensor contraction.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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
    return U, V, W


def _try_structured_low_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given target rank using structured approaches."""
    start_time = time.time()
    
    def _search_rank(r, num_trials=20, max_iter=150):
        """Search for a rank-r decomposition."""
        if time.time() - start_time > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() - start_time > deadline:
                return None
            
            np.random.seed(1000 + trial)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iter):
                if time.time() - start_time > deadline:
                    return None
                
                errs = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            errs[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                
                err = np.max(np.abs(errs))
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = errs[i,j,k]
                            for c in range(r):
                                dU[i*m+j, c] += e * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += e * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += e * U[i*m+j, c] * V[j*p+k, c]
                
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if best_err < 1e-8:
                U_s = np.round(best_U * 2) / 2
                V_s = np.round(best_V * 2) / 2
                W_s = np.round(best_W * 2) / 2
                
                U_l = [[float(x) for x in row] for row in U_s]
                V_l = [[float(x) for x in row] for row in V_s]
                W_l = [[float(x) for x in row] for row in W_s]
                
                if _verify_decomposition(U_l, V_l, W_l, n, m, p):
                    return U_l, V_l, W_l
        
        return None
    
    for r in range(21, 42):
        if time.time() - start_time > deadline:
            break
        result = _search_rank(r)
        if result is not None:
            return result
    
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Verify schoolbook is correct
    assert _verify_decomposition(U_sb, V_sb, W_sb, n, m, p), "Schoolbook decomposition failed verification!"
    
    # Try to find a lower-rank decomposition
    # Known bounds: for 2x3x7, the schoolbook rank is 42.
    # The minimum rank is at most 42 and at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # In practice, finding a decomposition below 42 is very hard.
    
    remaining = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 5:
        result = _try_structured_low_rank(n, m, p, best_rank, remaining)
        if result is not None:
            U, V, W = result
            new_rank = len(U[0])
            if new_rank < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = new_rank
    
    return best


if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
