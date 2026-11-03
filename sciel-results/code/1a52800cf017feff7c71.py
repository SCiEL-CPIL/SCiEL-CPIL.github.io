import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build T1 (3x3x3 part)
    T1 = np.zeros((n*m, m*3, 3*n))
    for i in range(n):
        for j in range(m):
            for k in range(3):
                T1[i*m+j, j*3+k, k*n+i] = 1
                
    # Build T2 (3x3x4 part)
    # B is 3x4, C is 4x3
    # Indices: i (0..2), j (0..2), k (0..3) corresponding to original k (3..6)
    T2 = np.zeros((n*m, m*4, 4*n))
    for i in range(n):
        for j in range(m):
            for k in range(4):
                T2[i*m+j, j*4+k, k*n+i] = 1

    def find_rank_decomposition(T, target_rank, max_trials=10, time_limit=10):
        dim1, dim2, dim3 = T.shape
        start_time = time.time()
        for trial in range(max_trials):
            if time.time() > time_limit or time.time() > deadline - 5:
                break
                
            U = np.random.randn(dim1, target_rank)
            V = np.random.randn(dim2, target_rank)
            W = np.random.randn(dim3, target_rank)
            
            def residual(params):
                U = params[:dim1*target_rank].reshape(dim1, target_rank)
                V = params[dim1*target_rank:dim1*target_rank+dim2*target_rank].reshape(dim2, target_rank)
                W = params[dim1*target_rank+dim2*target_rank:].reshape(dim3, target_rank)
                result = np.einsum('ar,br,cr->abc', U, V, W)
                return (result - T).flatten()
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = least_squares(residual, params0, method='lm', max_nfev=100)
                U = result.x[:dim1*target_rank].reshape(dim1, target_rank)
                V = result.x[dim1*target_rank:dim1*target_rank+dim2*target_rank].reshape(dim2, target_rank)
                W = result.x[dim1*target_rank+dim2*target_rank:].reshape(dim3, target_rank)
                
                result_tensor = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.max(np.abs(result_tensor - T))
                
                if error < 1e-6:
                    return U, V, W
            except:
                pass
        return None

    best_U1 = None
    best_V1 = None
    best_W1 = None
    best_rank1 = None
    
    # Try to find rank for T1
    for r1 in [23, 24, 25, 26]:
        if time.time() > deadline - 20:
            break
        res = find_rank_decomposition(T1, r1)
        if res:
            best_U1, best_V1, best_W1 = res
            best_rank1 = r1
            break
            
    if not best_U1:
        # Fallback to schoolbook for T1 (rank 27)
        best_rank1 = 27
        best_U1 = np.zeros((n*m, 27))
        best_V1 = np.zeros((m*3, 27))
        best_W1 = np.zeros((3*n, 27))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    best_U1[i*m+j, col] = 1
                    best_V1[j*3+k, col] = 1
                    best_W1[k*n+i, col] = 1
                    col += 1

    best_U2 = None
    best_V2 = None
    best_W2 = None
    best_rank2 = None
    
    # Try to find rank for T2
    for r2 in [32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
        res = find_rank_decomposition(T2, r2)
        if res:
            best_U2, best_V2, best_W2 = res
            best_rank2 = r2
            break
            
    if not best_U2:
        # Fallback to schoolbook for T2 (rank 36)
        best_rank2 = 36
        best_U2 = np.zeros((n*m, 36))
        best_V2 = np.zeros((m*4, 36))
        best_W2 = np.zeros((4*n, 36))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(4):
                    best_U2[i*m+j, col] = 1
                    best_V2[j*4+k, col] = 1
                    best_W2[k*n+i, col] = 1
                    col += 1

    total_rank = best_rank1 + best_rank2
    
    # Construct full U, V, W
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Copy T1 part
    U[:, :best_rank1] = best_U1
    V[:, :best_rank1] = best_V1
    W[:, :best_rank1] = best_W1
    
    # Copy T2 part
    # Note: V and W for T2 have different dimensions than the full V and W
    # V_full has shape (m*p, rank) = (21, rank)
    # V2 has shape (m*4, rank) = (12, rank)
    # The indices for T2 correspond to k=3..6 in the original tensor
    # In V_full, the index is j*p+k. For k=3..6, this is j*7+3 .. j*7+6
    # In V2, the index is j*4+k'. For k'=0..3, this is j*4+k'
    # So we need to map V2 indices to V_full indices
    
    col = best_rank1
    for i in range(n):
        U[i*m+j, col] # Wait, U is same for both parts?
        # U is (n*m, rank). The i,j indices are the same.
        # So U[:, best_rank1:total_rank] = best_U2
        
    U[:, best_rank1:total_rank] = best_U2
    
    # Map V2 to V
    for j in range(m):
        for k_prime in range(4):
            # V2 index: j*4 + k_prime
            # V_full index: j*p + (k_prime + 3)
            V[j*p + (k_prime + 3), best_rank1:total_rank] = best_V2[j*4 + k_prime, :]
            
    # Map W2 to W
    # W2 has shape (4*n, rank). Index k_prime*n + i
    # W_full has shape (p*n, rank). Index k*n + i
    # For T2, k = k_prime + 3
    for k_prime in range(4):
        for i in range(n):
            # W2 index: k_prime*n + i
            # W_full index: (k_prime + 3)*n + i
            W[(k_prime + 3)*n + i, best_rank1:total_rank] = best_W2[k_prime*n + i, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
