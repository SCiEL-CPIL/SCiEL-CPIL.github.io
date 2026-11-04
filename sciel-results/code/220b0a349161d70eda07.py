import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 130  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Build the target tensor T[i, j, k]
    # T[i, j, k] = 1 for all valid i, j, k
    T = np.ones((n, m, p))
    
    def try_rank_als(R):
        """
        Try to find a decomposition of rank R using Alternating Least Squares.
        """
        # Parameters:
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        
        # Reshape helper
        def reshape_U(U):
            return U.reshape(n, m, R)
        def reshape_V(V):
            return V.reshape(m, p, R)
        def reshape_W(W):
            return W.reshape(p, n, R)
        
        def compute_tensor(U, V, W):
            U3 = reshape_U(U)
            V3 = reshape_V(V)
            W3 = reshape_W(W)
            # T_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
            # Note: W3 is (p, n, R), so indices are k, i, c.
            return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        def solve_U(U, V, W):
            # Fix V, W. Solve for U.
            # We want min || T - T_approx(U) ||_F
            # T_approx[i, j, k] = sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c])
            # Let M[i, j, k, c] = V[j, k, c] * W[k, i, c]
            # Then T_approx[i, j, k] = sum_c U[i, j, c] M[i, j, k, c]
            # This is a linear system for each (i, j) pair? No, for each (i, j), we have p equations.
            # Actually, for fixed i, j, we have a vector of length p: T[i, j, :]
            # And we want to find U[i, j, :] (length R) such that:
            # sum_c U[i, j, c] * (sum_k V[j, k, c] W[k, i, c]) ... wait.
            # The term is V[j, k, c] * W[k, i, c].
            # For fixed i, j, and c, let B[i, j, c] = sum_k V[j, k, c] * W[k, i, c].
            # Then T_approx[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c].
            # This is not a simple linear solve for U[i,j,:] because the coefficient depends on k.
            # We need to solve:
            # min_U sum_{i,j,k} (T[i,j,k] - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
            # This is a linear least squares problem for U.
            # We can flatten U to a vector u of length n*m*R.
            # The matrix A of the linear system A u = t will be sparse.
            # It's easier to just use scipy.optimize.least_squares on the residuals for U, with V, W fixed.
            # Or, we can construct the linear system.
            
            # Let's use a simple gradient step or least squares on the linearized problem.
            # Actually, for fixed V, W, the problem is linear in U.
            # We can form the design matrix.
            # Number of equations: n*m*p = 24.
            # Number of unknowns: n*m*R.
            # For R=23, 6*23 = 138 unknowns. 24 equations.
            # Under-determined. We want the minimum norm solution?
            # No, we want ANY solution that fits.
            # Since it's under-determined, we can use `np.linalg.lstsq`.
            
            # Construct the linear system for U.
            # Reshape U to (n, m, R).
            # For each i, j, we have p equations.
            # Equation for (i, j, k):
            # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            # We can solve for each (i, j) independently?
            # Yes! The equations for different (i, j) do not share U variables.
            # U[i, j, :] only appears in equations for that specific (i, j).
            
            U_new = np.zeros((n, m, R))
            for i in range(n):
                for j in range(m):
                    # Unknowns: U[i, j, :] (length R)
                    # Equations: k in 0..p-1 (length p)
                    # Coefficients: A[k, c] = V[j, k, c] * W[k, i, c]
                    # Target: T[i, j, k]
                    
                    # V is (m, p, R) -> V[j, :, :] is (p, R)
                    # W is (p, n, R) -> W[:, i, :] is (p, R)
                    V_j = V.reshape(m, p, R)[j] # (p, R)
                    W_i = W.reshape(p, n, R)[:, i] # (p, R)
                    
                    A = V_j * W_i # (p, R)
                    b = T[i, j, :] # (p,)
                    
                    # Solve A x = b for x = U[i, j, :]
                    # Use lstsq
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i, j, :] = x
            
            return U_new.reshape(n * m, R)
        
        def solve_V(U, V, W):
            # Similar logic. Fix U, W. Solve for V.
            # V[j, k, c] appears in equations for (i, j, k).
            # For fixed j, k, we have n equations (i=0..n-1).
            # Unknowns: V[j, k, :] (length R).
            # Equation for (i, j, k):
            # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            V_new = np.zeros((m, p, R))
            for j in range(m):
                for k in range(p):
                    # Unknowns: V[j, k, :]
                    # Equations: i in 0..n-1
                    # Coefficients: B[i, c] = U[i, j, c] * W[k, i, c]
                    # Target: T[:, j, k]
                    
                    U_j = U.reshape(n, m, R)[:, j] # (n, R)
                    W_k = W.reshape(p, n, R)[k] # (n, R)
                    
                    B = U_j * W_k # (n, R)
                    b = T[:, j, k] # (n,)
                    
                    x, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    V_new[j, k, :] = x
            
            return V_new.reshape(m * p, R)
        
        def solve_W(U, V, W):
            # Fix U, V. Solve for W.
            # W[k, i, c] appears in equations for (i, j, k).
            # For fixed k, i, we have m equations (j=0..m-1).
            # Unknowns: W[k, i, :] (length R).
            # Equation for (i, j, k):
            # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            W_new = np.zeros((p, n, R))
            for k in range(p):
                for i in range(n):
                    # Unknowns: W[k, i, :]
                    # Equations: j in 0..m-1
                    # Coefficients: C[j, c] = U[i, j, c] * V[j, k, c]
                    # Target: T[i, :, k]
                    
                    U_i = U.reshape(n, m, R)[i] # (m, R)
                    V_k = V.reshape(m, p, R)[:, k] # (m, R)
                    
                    C = U_i * V_k # (m, R)
                    b = T[i, :, k] # (m,)
                    
                    x, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                    W_new[k, i, :] = x
            
            return W_new.reshape(p * n, R)
        
        # ALS Loop
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random init
            np.random.seed(r)
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            W = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            # Run ALS
            for it in range(50):
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                # Check convergence / exactness
                T_approx = compute_tensor(U, V, W)
                if np.allclose(T_approx, T, atol=1e-6):
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    T_check = compute_tensor(U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist(),
                            "rank": R
                        }
                        
        return None

    # Try lower ranks
    # We know rank 24 is always possible.
    # Try 23, 22, 21, 20, 19, 18, 17
    for R in range(23, 16, -1):
        if time.time() - start_time > max_time - 10:
            break
        res = try_rank_als(R)
        if res:
            return res

    # Fallback: Schoolbook decomposition
    rank = target_rank
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
