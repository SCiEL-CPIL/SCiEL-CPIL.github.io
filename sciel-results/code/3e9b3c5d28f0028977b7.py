import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin from 150s
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i for some i,j,k
    # Else 0
    T = np.zeros((size_u, size_v, size_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Function to check if U, V, W exactly decompose T
    def check_decomposition(U, V, W, tol=1e-9):
        R = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(R, T, atol=tol):
            return False
        # Check if entries are integer/half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(R_r, T, atol=1e-9):
            return True
        return False

    # Function to find integer/half-integer solution via ALS
    def find_decomposition(rank):
        if time.time() > deadline:
            return None
            
        # Try multiple random starts
        # Increase attempts for lower ranks
        num_attempts = 200 if rank <= 23 else 50
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
                
            # Use different seeds for diversity
            np.random.seed(42 + attempt)
            U = np.random.randn(size_u, rank)
            V = np.random.randn(size_v, rank)
            W = np.random.randn(size_w, rank)
            
            # Increase iterations for critical ranks
            max_iter = 300 if rank <= 23 else 100
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                    
                # Update U
                M_vw = np.einsum('br,cr->bcr', V, W).reshape(size_v * size_w, rank)
                T_1 = T.reshape(size_u, size_v * size_w)
                U, residuals, rank_u, s = np.linalg.lstsq(M_vw.T, T_1.T, rcond=None)
                U = U.T
                
                # Update V
                M_ua = np.einsum('ar,cr->acr', U, W).reshape(size_u * size_w, rank)
                T_2 = T.transpose(1, 0, 2).reshape(size_v, size_u * size_w)
                V, residuals, rank_v, s = np.linalg.lstsq(M_ua.T, T_2.T, rcond=None)
                V = V.T
                
                # Update W
                M_uv = np.einsum('ar,br->abr', U, V).reshape(size_u * size_v, rank)
                T_3 = T.transpose(0, 2, 1).reshape(size_w, size_u * size_v)
                W, residuals, rank_w, s = np.linalg.lstsq(M_uv.T, T_3.T, rcond=None)
                W = W.T
                
            # Check convergence
            if check_decomposition(U, V, W):
                # Round to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None

    best_result = None
    # Try ranks from 23 to 26 (Skipping 20-22 as they are known impossible for 3x3)
    for rank in range(23, 27):
        if time.time() > deadline - 10: # Leave 10s for cleanup/return
            break
        res = find_decomposition(rank)
        if res is not None:
            best_result = res
            break # Found minimal rank, stop

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
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
