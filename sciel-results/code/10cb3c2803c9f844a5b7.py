import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety
    
    n, m, p = 3, 5, 5
    size_U = n * m  # 15
    size_V = m * p  # 25
    size_W = p * n  # 15
    
    # Precompute target tensor
    T_target = np.zeros((size_U, size_V, size_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T_target[I, J, K] = 1.0
    
    def verify(U, V, W, tol=1e-6):
        """Verify that U, V, W exactly decompose the tensor."""
        R = U.shape[1]
        Res = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(Res - T_target)
        return err < tol, err
    
    def snap_to_halfint(M):
        """Snap matrix to nearest half-integer and verify."""
        M_snap = np.round(M * 2) / 2
        return M_snap
    
    def solve_W(U, V, T):
        """Given U and V, solve for W via linear least squares."""
        # Res[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
        # For each K, this is: sum_c (sum_{I,J} U[I,c] V[J,c] * delta) W[K,c] = T[K]
        # Actually: T[I,J,K] = sum_c (U[I,c] V[J,c]) W[K,c]
        # Let A[I,J,c] = U[I,c] * V[J,c]
        # Then T[I,J,K] = sum_c A[I,J,c] * W[K,c]
        # This is linear in W: for each K, W[K,:] is determined by all (I,J)
        
        R = U.shape[1]
        # Build system: for each K, sum_c A[I,J,c] * W[K,c] = T[I,J,K] for all I,J
        # Stack: let's form a matrix M of size (size_U * size_V) x R
        # M[row, c] where row = I * size_V + J
        # M[row, c] = U[I,c] * V[J,c]
        
        M = np.zeros((size_U * size_V, R), dtype=np.float64)
        for I in range(size_U):
            for J in range(size_V):
                row = I * size_V + J
                for c in range(R):
                    M[row, c] = U[I, c] * V[J, c]
        
        # Now T_flat[row, K] = sum_c M[row, c] * W[K, c]
        # T_flat is size_U * size_V x size_W
        T_flat = T.reshape(size_U * size_V, size_W)
        
        # We want: M @ W.T = T_flat.T  =>  W.T = M^{-1} T_flat.T (in least squares sense)
        # So W = (M.T @ T_flat.T).T ... let me think again
        # T_flat[row, K] = sum_c M[row, c] W[K, c]
        # Let's define W_mat = W (size_W x R)
        # Then T_flat = M @ W_mat.T  (size_U*size_V x size_W)
        # So W_mat.T = M^{-1} T_flat  =>  W_mat = (M^{-1} T_flat).T
        # In least squares: W_mat.T = np.linalg.lstsq(M, T_flat, rcond=None)[0]
        
        try:
            W_T = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            W = W_T.T
        except:
            return None
        
        return W
    
    def solve_V(U, W, T):
        """Given U and W, solve for V via linear least squares."""
        # T[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
        # For fixed U, W: T[I,J,K] = sum_c (U[I,c] W[K,c]) V[J,c]
        # Let B[I,J,K] ... let's rearrange
        # T[I,J,K] = sum_c V[J,c] * (U[I,c] W[K,c])
        # For each J: sum_c V[J,c] * C[I,K,c] = T[I,J,K] where C[I,K,c] = U[I,c]*W[K,c]
        # Stack over I,K: for each J, we have size_U * size_W equations
        # M2[row, c] where row = I * size_W + K
        # M2[row, c] = U[I,c] * W[K,c]
        # T2[row, J] = sum_c M2[row, c] * V[J, c]
        # So T2 = M2 @ V.T => V.T = M2^{-1} T2 => V = (M2^{-1} T2).T
        
        R = U.shape[1]
        M2 = np.zeros((size_U * size_W, R), dtype=np.float64)
        for I in range(size_U):
            for K in range(size_W):
                row = I * size_W + K
                for c in range(R):
                    M2[row, c] = U[I, c] * W[K, c]
        
        T2 = T.reshape(size_U, size_W, size_V).transpose(0, 2, 1).reshape(size_U * size_W, size_V)
        # Wait, T is (size_U, size_V, size_W). We need T2[row, J] where row = I*size_W+K
        # T2[I*size_W+K, J] = T[I, J, K]
        
        T2 = np.zeros((size_U * size_W, size_V), dtype=np.float64)
        for I in range(size_U):
            for J in range(size_V):
                for K in range(size_W):
                    T2[I * size_W + K, J] = T[I, J, K]
        
        try:
            V_T = np.linalg.lstsq(M2, T2, rcond=None)[0]
            V = V_T.T
        except:
            return None
        
        return V
    
    def solve_U(V, W, T):
        """Given V and W, solve for U via linear least squares."""
        # T[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
        # For fixed V, W: T[I,J,K] = sum_c U[I,c] * (V[J,c] W[K,c])
        # For each I: sum_c U[I,c] * D[J,K,c] = T[I,J,K] where D[J,K,c] = V[J,c]*W[K,c]
        # Stack over J,K: for each I, we have size_V * size_W equations
        # M3[row, c] where row = J * size_W + K
        # M3[row, c] = V[J,c] * W[K,c]
        # T3[row, I] = sum_c M3[row, c] * U[I, c]
        # So T3 = M3 @ U.T => U.T = M3^{-1} T3 => U = (M3^{-1} T3).T
        
        R = V.shape[1]
        M3 = np.zeros((size_V * size_W, R), dtype=np.float64)
        for J in range(size_V):
            for K in range(size_W):
                row = J * size_W + K
                for c in range(R):
                    M3[row, c] = V[J, c] * W[K, c]
        
        T3 = np.zeros((size_V * size_W, size_U), dtype=np.float64)
        for I in range(size_U):
            for J in range(size_V):
                for K in range(size_W):
                    T3[J * size_W + K, I] = T[I, J, K]
        
        try:
            U_T = np.linalg.lstsq(M3, T3, rcond=None)[0]
            U = U_T.T
        except:
            return None
        
        return U
    
    def als_decompose(R, max_iters=50, tol=1e-8):
        """Alternating Least Squares for tensor decomposition."""
        # Initialize randomly
        U = np.random.randn(size_U, R) * 0.5
        V = np.random.randn(size_V, R) * 0.5
        W = np.random.randn(size_W, R) * 0.5
        
        prev_cost = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Solve for W
            W_new = solve_W(U, V, T_target)
            if W_new is None:
                return None, None, None, np.inf
            W = W_new
            
            # Solve for V
            V_new = solve_V(U, W, T_target)
            if V_new is None:
                return None, None, None, np.inf
            V = V_new
            
            # Solve for U
            U_new = solve_U(V, W, T_target)
            if U_new is None:
                return None, None, None, np.inf
            U = U_new
            
            # Compute cost
            Res = np.einsum('ic,jc,kc->ijk', U, V, W)
            cost = np.linalg.norm(Res - T_target)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return U, V, W, prev_cost
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    # Known: rank 75 is trivial. Try to find lower.
    # Based on literature, the rank for 3x5 * 5x5 is likely around 56-65
    ranks_to_try = list(range(56, 75)) + [75]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 10
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            U, V, W, cost = als_decompose(R, max_iters=30, tol=1e-9)
            
            if U is not None and cost < 1e-6:
                # Snap to half-integers
                U_snap = snap_to_halfint(U)
                V_snap = snap_to_halfint(V)
                W_snap = snap_to_halfint(W)
                
                # Verify snapped
                valid, err = verify(U_snap, V_snap, W_snap, tol=1e-5)
                if valid:
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found = True
                    break
        
        if found:
            break
    
    # If no better found, return seed (rank 75)
    if best_rank < n * m * p:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Return Seed
        rank = n * m * p
        U = [[0] * rank for _ in range(size_U)]
        V = [[0] * rank for _ in range(size_V)]
        W = [[0] * rank for _ in range(size_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
