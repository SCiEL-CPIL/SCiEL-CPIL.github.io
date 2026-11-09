import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140  # 140s safety margin
    
    # Schoolbook rank-36 decomposition as baseline
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_U, best_V, best_W = U0, V0, W0
    best_rank = n * m * p
    
    # Verify decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    def verify(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by zeroing out some columns and re-solving
    # Strategy: try rank r from 35 down to 25
    # For each rank, try to find a decomposition by optimizing
    
    def try_rank(r):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with first r columns of schoolbook
        U = U0[:, :r].copy()
        V = V0[:, :r].copy()
        W = W0[:, :r].copy()
        
        # The constraint is: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is a system of n*m*p = 36 equations in r*9 + r*12 + r*12 = 33r variables
        # For r=32: 33*32=1056 vars, 36 eqs - very underdetermined
        # We need to find ANY solution
        
        # Use iterative approach: fix two factors, solve for third linearly
        # Then cycle through factors
        
        def solve_W(U, V, r):
            """Given U and V, solve for W such that constraints hold."""
            # For each (i,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
            # Actually: C[i,k] = sum_j A[i,j]B[j,k] means
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
            # 
            # Fix U and V. For each (i,k), we need:
            # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is: sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # 
            # So for each (i,k): let S[i,k,r] = sum_j U[i*m+j,r]*V[j*p+k,r]
            # Then: sum_r S[i,k,r] * W[k*n+i,r] = 1
            # This gives n*p = 12 linear equations in r unknowns (W[k*n+i,r] for fixed i,k)
            # Wait, W has shape (p*n, r), so W[k*n+i, r] for fixed (i,k) is one row of W
            # We have 12 rows of W, each with r entries = 12r unknowns
            # And 12 equations (one per (i,k) pair)
            # Underdetermined if 12r > 12, i.e., r > 1
            
            # Actually let me reconsider. The constraint is:
            # For each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # That's 36 equations.
            # W has 12*r unknowns.
            # For fixed U,V: 36 eqs in 12r unknowns.
            # For r>=3: underdetermined, many solutions.
            
            # Let's set up the linear system
            # Unknowns: W[0,0], W[0,1], ..., W[11, r-1] (flattened)
            # Equation for (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            A_mat = np.zeros((n*m*p, p*n*r))
            b_vec = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            # W[k*n+i, r_idx] is at position (k*n+i)*r + r_idx
                            A_mat[eq_idx, (k*n+i)*r + r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                        eq_idx += 1
            
            # Solve least squares (or find any solution)
            try:
                sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_sol = sol.reshape(p*n, r)
                # Check residual
                residual = np.linalg.norm(A_mat @ sol - b_vec)
                return W_sol, residual
            except:
                return None, float('inf')
            
        def solve_U(V, W, r):
            """Given V and W, solve for U."""
            # Equation: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # Unknowns: U[i*m+j,r] -> n*m*r = 9r unknowns
            # 36 equations
            
            A_mat = np.zeros((n*m*p, n*m*r))
            b_vec = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            # U[i*m+j, r_idx] is at position (i*m+j)*r + r_idx
                            A_mat[eq_idx, (i*m+j)*r + r_idx] = V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        eq_idx += 1
            
            try:
                sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U_sol = sol.reshape(n*m, r)
                residual = np.linalg.norm(A_mat @ sol - b_vec)
                return U_sol, residual
            except:
                return None, float('inf')
        
        def solve_V(U, W, r):
            """Given U and W, solve for V."""
            # Unknowns: V[j*p+k,r] -> m*p*r = 12r unknowns
            # 36 equations
            
            A_mat = np.zeros((n*m*p, m*p*r))
            b_vec = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            # V[j*p+k, r_idx] is at position (j*p+k)*r + r_idx
                            A_mat[eq_idx, (j*p+k)*r + r_idx] = U[i*m+j, r_idx] * W[k*n+i, r_idx]
                        eq_idx += 1
            
            try:
                sol, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V_sol = sol.reshape(m*p, r)
                residual = np.linalg.norm(A_mat @ sol - b_vec)
                return V_sol, residual
            except:
                return None, float('inf')
        
        # Alternating minimization
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            W_new, res_w = solve_W(U, V, r)
            if W_new is not None:
                W = W_new
            
            U_new, res_u = solve_U(V, W, r)
            if U_new is not None:
                U = U_new
            
            V_new, res_v = solve_U(V, W, r)  # placeholder
            V_new, res_v = solve_V(U, W, r)
            if V_new is not None:
                V = V_new
        
        # Check if solution is valid
        if verify(U, V, W, tol=1e-4):
            return U, V, W
        
        return None
    
    # Try ranks from 35 down
    for r in range(n*m*p - 1, 24, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = r
            # Try to go lower
            continue
        else:
            # If we can't find rank r, stop
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-4):
        # Fall back to schoolbook
        best_U, best_V, best_W = U0, V0, W0
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
