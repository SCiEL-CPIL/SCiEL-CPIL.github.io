import numpy as np
import scipy.optimize
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
        R = U.shape[1]
        M = np.zeros((size_U * size_V, R), dtype=np.float64)
        for I in range(size_U):
            for J in range(size_V):
                row = I * size_V + J
                for c in range(R):
                    M[row, c] = U[I, c] * V[J, c]
        
        T_flat = T.reshape(size_U * size_V, size_W)
        
        try:
            W_T = np.linalg.lstsq(M, T_flat, rcond=None)[0]
            W = W_T.T
        except:
            return None
        
        return W
    
    def solve_V(U, W, T):
        """Given U and W, solve for V via linear least squares."""
        R = U.shape[1]
        M2 = np.zeros((size_U * size_W, R), dtype=np.float64)
        for I in range(size_U):
            for K in range(size_W):
                row = I * size_W + K
                for c in range(R):
                    M2[row, c] = U[I, c] * W[K, c]
        
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
    
    def als_decompose(U_init, V_init, W_init, R, max_iters=50, tol=1e-8):
        """Alternating Least Squares for tensor decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_cost = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            W_new = solve_W(U, V, T_target)
            if W_new is None:
                return None, None, None, np.inf
            W = W_new
            
            V_new = solve_V(U, W, T_target)
            if V_new is None:
                return None, None, None, np.inf
            V = V_new
            
            U_new = solve_U(V, W, T_target)
            if U_new is None:
                return None, None, None, np.inf
            U = U_new
            
            Res = np.einsum('ic,jc,kc->ijk', U, V, W)
            cost = np.linalg.norm(Res - T_target)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        return U, V, W, prev_cost
    
    def polish(U, V, W, R):
        """Polish the solution using scipy.optimize.least_squares."""
        size_U, size_V, size_W = U.shape[0], V.shape[0], W.shape[0]
        
        def residual(vec):
            U_flat = vec[:size_U*R]
            V_flat = vec[size_U*R:size_U*R + size_V*R]
            W_flat = vec[size_U*R + size_V*R:]
            
            U = U_flat.reshape(size_U, R)
            V = V_flat.reshape(size_V, R)
            W = W_flat.reshape(size_W, R)
            
            Res = np.einsum('ic,jc,kc->ijk', U, V, W)
            return (Res - T_target).flatten()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = scipy.optimize.least_squares(residual, x0, method='trf', 
                                                ftol=1e-12, xtol=1e-12, gtol=1e-12,
                                                max_nfev=1000)
            if res.cost < 1e-8:
                U_opt = res.x[:size_U*R].reshape(size_U, R)
                V_opt = res.x[size_U*R:size_U*R + size_V*R].reshape(size_V, R)
                W_opt = res.x[size_U*R + size_V*R:].reshape(size_W, R)
                return U_opt, V_opt, W_opt
        except:
            pass
        
        return None, None, None
    
    def create_structured_init(R):
        """Create a structured initialization based on the schoolbook decomposition."""
        # Start from schoolbook and try to merge terms
        # This is a heuristic: initialize with patterns that might allow rank reduction
        
        U = np.zeros((size_U, R), dtype=np.float64)
        V = np.zeros((size_V, R), dtype=np.float64)
        W = np.zeros((size_W, R), dtype=np.float64)
        
        # Use a pattern based on the algebraic structure
        # For each (i,j,k), we have a term. Try to group them.
        # Simple approach: use random but with structure
        
        # Initialize with small random values
        U = np.random.randn(size_U, R) * 0.3
        V = np.random.randn(size_V, R) * 0.3
        W = np.random.randn(size_W, R) * 0.3
        
        # Add some structure: make certain rows/columns similar
        for c in range(R):
            # Randomly permute which (i,j) pairs share structure
            pass
        
        return U, V, W
    
    def create_schoolbook_scaled_init(R):
        """Scale the schoolbook decomposition to rank R (if R < 75, use projection)."""
        # Schoolbook has rank 75
        U_school = np.zeros((size_U, 75), dtype=np.float64)
        V_school = np.zeros((size_V, 75), dtype=np.float64)
        W_school = np.zeros((size_W, 75), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        
        if R >= 75:
            # Take first R columns
            return U_school[:, :R], V_school[:, :R], W_school[:, :R]
        else:
            # Project to lower rank using SVD
            # This won't be exact, but gives a good starting point
            U_trunc, _, _ = np.linalg.svd(U_school, full_matrices=False)
            V_trunc, _, _ = np.linalg.svd(V_school, full_matrices=False)
            W_trunc, _, _ = np.linalg.svd(W_school, full_matrices=False)
            
            return U_trunc[:, :R], V_trunc[:, :R], W_trunc[:, :R]
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    # Based on literature, the rank for 3x5 * 5x5 might be around 56-65
    ranks_to_try = list(range(56, 75)) + [75]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        found = False
        num_restarts = 8
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random init
                U_init = np.random.randn(size_U, R) * 0.5
                V_init = np.random.randn(size_V, R) * 0.5
                W_init = np.random.randn(size_W, R) * 0.5
            elif restart % 3 == 1:
                # Structured init
                U_init, V_init, W_init = create_structured_init(R)
            else:
                # Schoolbook scaled init
                U_init, V_init, W_init = create_schoolbook_scaled_init(R)
            
            U, V, W, cost = als_decompose(U_init, V_init, W_init, R, max_iters=40, tol=1e-9)
            
            if U is not None and cost < 1e-4:
                # Try to polish
                U_pol, V_pol, W_pol = polish(U, V, W, R)
                
                if U_pol is not None:
                    U, V, W = U_pol, V_pol, W_pol
                    Res = np.einsum('ic,jc,kc->ijk', U, V, W)
                    cost = np.linalg.norm(Res - T_target)
                
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
