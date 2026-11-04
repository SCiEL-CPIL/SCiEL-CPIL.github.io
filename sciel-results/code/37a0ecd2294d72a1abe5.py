import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor components T[i, j, k] = 1 for all i, j, k
    # Condition: sum_t W[i, k, t] * U[i, j, t] * V[j, k, t] == 1
    
    # We search for rank r
    # Start with a reasonable guess
    # For n=2, m=4, p=4, rank 8 is a good guess (2*4)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    deadline = time.time() + 100
    
    for r in range(2, 33):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        for attempt in range(10):
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # We need to solve for W given U, V
            # T[i, j, k] = sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t]
            # Let A_t[i, j] = U[i*m+j, t]
            # Let B_t[j, k] = V[j*p+k, t]
            # C_t[i, k] = sum_j A_t[i, j] B_t[j, k]
            # Then T[i, j, k] is not directly C.
            # The condition is sum_t W[i, k, t] * (something) = 1?
            # No, the condition is:
            # For each i, j, k: sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t] = 1
            
            # This is linear in W.
            # Let's form the system for W.
            # Variables: W[k*n+i, t] for all k, i, t.
            # Equations: for each i, j, k:
            # sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t] = 1
            
            # Note that W index is k*n+i.
            # Let's group equations by (i, k).
            # For fixed i, k, we have m equations (one for each j).
            # sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t] = 1
            
            # Let w_ik be the vector of W[k*n+i, :] of length r.
            # Let u_ij be U[i*m+j, :] of length r.
            # Let v_jk be V[j*p+k, :] of length r.
            # The equation is: w_ik^T (u_ij * v_jk) = 1? No.
            # sum_t w_ik[t] u_ij[t] v_jk[t] = 1.
            # This is a trilinear form.
            
            # This is hard to solve directly.
            # Let's use the fact that we can fix U and V and solve for W?
            # No, it's not linear in W if U, V are fixed?
            # It IS linear in W if U, V are fixed.
            # sum_t W[idx, t] * (U[i*m+j, t] * V[j*p+k, t]) = 1.
            # Let M_jk[i, t] = U[i*m+j, t] * V[j*p+k, t].
            # Then sum_t W[k*n+i, t] M_jk[i, t] = 1.
            # This is a linear system for W[k*n+i, :].
            
            # For each i, k, we have a system of m equations:
            # sum_{j=0}^{m-1} ... No, j is the index of the equation.
            # For fixed i, k, we have m equations (for j=0..m-1).
            # Unknowns: W[k*n+i, t] for t=0..r-1.
            # Equation j: sum_t W[k*n+i, t] U[i*m+j, t] V[j*p+k, t] = 1.
            
            # Let's build the matrix A_ik of size m x r.
            # A_ik[j, t] = U[i*m+j, t] * V[j*p+k, t].
            # Then A_ik @ W[k*n+i, :] = 1 (vector of ones of length m).
            
            # We can solve this least squares for each i, k.
            
            W_sol = np.zeros((p * n, r))
            valid = True
            for i in range(n):
                for k in range(p):
                    A_ik = np.zeros((m, r))
                    for j in range(m):
                        A_ik[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve A_ik @ w = 1
                    try:
                        w, _, _, _ = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        W_sol[k*n+i, :] = w
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Check if the solution is exact
            # Residual
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(W_sol[k*n+i, :], U[i*m+j, :] * V[j*p+k, :])
                        if abs(val - 1.0) > 1e-6:
                            err = 1
                            break
                    if err: break
                if err: break
            if err == 0:
                # Found a decomposition of rank r
                # Convert to list of lists with integer/half-integer if possible
                # The problem allows integer/half-integer
                # We can just return the floats if they are close to half-integers
                # Or just return the floats? "integer/half-integer entries"
                # We should try to snap to half-integers
                
                U_out = [[0]*r for _ in range(n*m)]
                V_out = [[0]*r for _ in range(m*p)]
                W_out = [[0]*r for _ in range(p*n)]
                
                for i in range(n*m):
                    for t in range(r):
                        val = U[i, t]
                        # Snap to nearest half-integer
                        snapped = round(val * 2) / 2
                        if abs(val - snapped) < 1e-5:
                            U_out[i][t] = snapped
                        else:
                            U_out[i][t] = val
                            
                for i in range(m*p):
                    for t in range(r):
                        val = V[i, t]
                        snapped = round(val * 2) / 2
                        if abs(val - snapped) < 1e-5:
                            V_out[i][t] = snapped
                        else:
                            V_out[i][t] = val
                            
                for i in range(p*n):
                    for t in range(r):
                        val = W_sol[i, t]
                        snapped = round(val * 2) / 2
                        if abs(val - snapped) < 1e-5:
                            W_out[i][t] = snapped
                        else:
                            W_out[i][t] = val
                            
                return {"U": U_out, "V": V_out, "W": W_out}
                
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
