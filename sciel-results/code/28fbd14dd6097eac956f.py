import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None

    def check_validity(U, V, W, atol=1e-9):
        res = np.zeros_like(T)
        for alpha in range(U.shape[1]):
            res += U[:, alpha][:, None, None] * V[:, alpha][None, :, None] * W[:, alpha][None, None, :]
        return np.allclose(res, T, atol=atol)

    def snap_to_half_integers(U, V, W):
        """
        Attempts to snap values to nearest half-integers if they are close.
        Returns True if the snapped matrices still satisfy the decomposition.
        """
        def snap_matrix(M):
            # Multiply by 2, round, divide by 2
            M_snapped = np.round(M * 2) / 2
            # Check if close to original
            if not np.allclose(M, M_snapped, atol=1e-5):
                return None
            return M_snapped

        U_s = snap_matrix(U)
        V_s = snap_matrix(V)
        W_s = snap_matrix(W)
        
        if U_s is None or V_s is None or W_s is None:
            return None
            
        # Verify the snapped matrices
        if check_validity(U_s, V_s, W_s, atol=1e-6):
            return U_s, V_s, W_s
        else:
            return None

    def run_als(r, init_V, init_W, max_iters=100):
        U = np.random.randn(n*m, r)
        V = init_V.copy()
        W = init_W.copy()
        
        # Precompute T reshapes
        T_flat_U = T.reshape(n*m, -1) # (nm, mp*pn)
        T_flat_V = T.transpose(1, 0, 2).reshape(m*p, -1) # (mp, nm*pn)
        T_flat_W = T.transpose(2, 0, 1).reshape(p*n, -1) # (pn, nm*mp)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # B[j*p*n + k, alpha] = V[j, alpha] * W[k, alpha]
            # Vectorized construction
            V_exp = V[:, np.newaxis, :] # (mp, 1, r)
            W_exp = W[np.newaxis, :, :] # (1, pn, r)
            B = (V_exp * W_exp).reshape(m*p*p*n, r)
            
            # Solve U @ B = T_flat_U
            # U = T_flat_U @ B^+
            # Using lstsq: B.T @ U.T = T_flat_U.T
            try:
                U_T, _, _, _ = np.linalg.lstsq(B.T, T_flat_U.T, rcond=None)
                U = U_T.T
            except:
                return None
                
            # Update V
            # C[i*p*n + k, alpha] = U[i, alpha] * W[k, alpha]
            U_exp = U[:, np.newaxis, :] # (nm, 1, r)
            W_exp = W[np.newaxis, :, :] # (1, pn, r)
            C = (U_exp * W_exp).reshape(n*m*p*n, r)
            
            # Solve V @ C = T_flat_V
            # V = T_flat_V @ C^+
            try:
                V_T, _, _, _ = np.linalg.lstsq(C.T, T_flat_V.T, rcond=None)
                V = V_T.T
            except:
                return None
                
            # Update W
            # D[i*m*p + j, alpha] = U[i, alpha] * V[j, alpha]
            U_exp = U[:, np.newaxis, :] # (nm, 1, r)
            V_exp = V[np.newaxis, :, :] # (1, mp, r)
            D = (U_exp * V_exp).reshape(n*m*m*p, r)
            
            # Solve W @ D = T_flat_W
            # W = T_flat_W @ D^+
            try:
                W_T, _, _, _ = np.linalg.lstsq(D.T, T_flat_W.T, rcond=None)
                W = W_T.T
            except:
                return None
                
            # Check convergence
            if check_validity(U, V, W):
                # Try to snap
                snapped = snap_to_half_integers(U, V, W)
                if snapped:
                    return snapped
                else:
                    # If snapping fails but validity holds, return as is? 
                    # The prompt says "integer/half-integer entries". 
                    # If we can't snap, it might be invalid for the strict interface.
                    # However, if the values are e.g. 0.500000001, snapping should work.
                    # If they are irrational, we return None to keep searching.
                    return None

        return None

    # Search
    # Start with low ranks. The rank for 3x5x6 is likely around 15-25.
    # Schoolbook is 90.
    # We will iterate ranks from 1 up to 90.
    # To save time, we can skip some very low ranks if we know they are impossible,
    # but for now, let's just run.
    
    # Use a fixed seed for reproducibility in the search process if we want, 
    # but random restarts are better.
    
    for r in range(1, 91):
        if time.time() > deadline:
            break
            
        # Number of restarts: decrease as rank increases?
        # Low ranks are harder to find but more valuable.
        num_restarts = 50 if r < 20 else 20 if r < 50 else 5
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            V_init = np.random.randn(m*p, r)
            W_init = np.random.randn(p*n, r)
            
            res = run_als(r, V_init, W_init, max_iters=50)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
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
