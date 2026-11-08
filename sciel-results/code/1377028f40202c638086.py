import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    n, m, p = DIMENSIONS
    
    # Target: for all i,k, sum_c W[k*n+i,c] * outer(U[i*m:(i+1)*m,c], V[k*p:(k+1)*p,c]) = I_m
    # Equivalently: for all i,k,j,j': sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
    
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for c in range(R):
                    wc = W[k*n+i, c]
                    if wc == 0:
                        continue
                    uc = U[i*m:(i+1)*m, c]
                    vc = V[k*p:(k+1)*p, c]
                    M += wc * np.outer(uc, vc)
                if not np.allclose(M, np.eye(m), atol=1e-8):
                    return False
        return True
    
    def solve_linear(fixed1, fixed2, which):
        """
        Solve for the variable matrix given two fixed matrices.
        which: 'U', 'V', or 'W'
        fixed1, fixed2: the other two matrices (order depends on which)
        """
        R = fixed1.shape[1]
        
        if which == 'W':
            # U, V fixed. For each (i,k,j,j'): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'}
            # Group by (i,k): for each (i,k), we have m*m equations for W[k*n+i, :]
            # Each equation: sum_c W[k*n+i,c] * A_c = b where A_c = U[i*m+j,c]*V[j'*p+k,c]
            # This is: W[k*n+i, :] @ A = b where A is (m*m, R), b is (m*m,)
            W = np.zeros((p*n, R))
            for i in range(n):
                for k in range(p):
                    # Build A: (m*m, R) where A[j*m+j2, c] = U[i*m+j,c] * V[j'*p+k,c] with j'=j2
                    # Wait: equation is for (j, j2): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
                    # So for fixed (i,k), the unknown is W[k*n+i, :] of length R
                    # Equation for (j,j2): sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j2*p+k,c]) = delta_{j,j2}
                    # Let A[j*m+j2, c] = U[i*m+j,c] * V[j2*p+k,c]
                    # Then A @ W[k*n+i, :].T = b where b[j*m+j2] = delta_{j,j2}
                    A = np.zeros((m*m, R))
                    b = np.zeros(m*m)
                    for j in range(m):
                        for j2 in range(m):
                            A[j*m+j2, :] = U[i*m+j, :] * V[j2*p+k, :]
                            b[j*m+j2] = 1.0 if j == j2 else 0.0
                    # Solve A @ x = b for x = W[k*n+i, :].T
                    # Least squares
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = sol
            return W
        
        elif which == 'U':
            # V, W fixed. For each (i,j): equations involve U[i*m+j, :]
            # Equation for (i,k,j,j2): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
            # For fixed (i,j): sum over k,j2: W[k*n+i,c] * V[j2*p+k,c] * U[i*m+j,c] = delta_{j,j2}
            # Let A[k*m+j2, c] = W[k*n+i,c] * V[j2*p+k,c]
            # b[k*m+j2] = delta_{j,j2}
            # A @ U[i*m+j, :].T = b
            U = np.zeros((n*m, R))
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p*m, R))
                    b = np.zeros(p*m)
                    for k in range(p):
                        for j2 in range(m):
                            A[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                            b[k*m+j2] = 1.0 if j == j2 else 0.0
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = sol
            return U
        
        elif which == 'V':
            # U, W fixed. For each (k,j2): equations involve V[j2*p+k, :]
            # Equation for (i,k,j,j2): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
            # For fixed (k,j2): sum over i,j: W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
            # Let A[i*m+j, c] = W[k*n+i,c] * U[i*m+j,c]
            # b[i*m+j] = delta_{j,j2}
            # A @ V[j2*p+k, :].T = b
            V = np.zeros((m*p, R))
            for k in range(p):
                for j2 in range(m):
                    A = np.zeros((n*m, R))
                    b = np.zeros(n*m)
                    for i in range(n):
                        for j in range(m):
                            A[i*m+j, :] = W[k*n+i, :] * U[i*m+j, :]
                            b[i*m+j] = 1.0 if j == j2 else 0.0
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j2*p+k, :] = sol
            return V
    
    def alternating_minimization(r, max_iter=200, tol=1e-10):
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 20.0:
                break
            
            # Solve for W given U, V
            W = solve_linear(U, V, 'W')
            # Solve for U given V, W
            U = solve_linear(V, W, 'U')
            # Solve for V given U, W
            V = solve_linear(U, W, 'V')
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, m))
                    for c in range(r):
                        wc = W[k*n+i, c]
                        if wc != 0:
                            uc = U[i*m:(i+1)*m, c]
                            vc = V[k*p:(k+1)*p, c]
                            M += wc * np.outer(uc, vc)
                    err += np.linalg.norm(M - np.eye(m))**2
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        # Verify
        if verify(U, V, W):
            return U, V, W
        return None
    
    # Try ranks from low to high
    for r in range(10, 121):
        if time.time() > deadline - 10.0:
            break
        
        result = alternating_minimization(r)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries
            # Round to nearest 0.5 if close
            def to_list(mat):
                out = []
                for row in mat:
                    new_row = []
                    for val in row:
                        # Round to nearest 0.5
                        rounded = round(val * 2) / 2
                        if abs(val - rounded) < 1e-6:
                            new_row.append(rounded)
                        else:
                            new_row.append(val)
                    out.append(new_row)
                return out
            
            U_list = to_list(U)
            V_list = to_list(V)
            W_list = to_list(W)
            
            # Final verification with the list version
            def verify_list(U, V, W):
                R = len(U[0])
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, m))
                        for c in range(R):
                            wc = W[k*n+i][c]
                            if wc == 0:
                                continue
                            uc = [U[i*m+j][c] for j in range(m)]
                            vc = [V[j2*p+k][c] for j2 in range(m)]
                            M += wc * np.outer(uc, vc)
                        if not np.allclose(M, np.eye(m), atol=1e-8):
                            return False
                return True
            
            if verify_list(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
