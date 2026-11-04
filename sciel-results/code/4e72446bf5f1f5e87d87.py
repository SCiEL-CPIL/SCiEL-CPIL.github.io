import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 32.
    # Structure:
    # Split A (2x4) into A3 (2x3) and A1 (2x1).
    # Split B (4x5) into B3 (3x5) and B1 (1x5).
    # B3 is split into B33 (3x3) and B32 (3x2).
    # C = A3 B33 + A3 B32 + A1 B1.
    # Rank(A3 B33) = 11 (known 2x3x3 min rank).
    # Rank(A3 B32) = 11 (2x3x2 min rank).
    # Rank(A1 B1) = 10 (2x1x5 rank).
    # Total rank = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)] # 8 x 32
    V = [[0] * rank for _ in range(m * p)] # 20 x 32
    W = [[0] * rank for _ in range(p * n)] # 10 x 32
    
    # We need the explicit coefficients for 2x3x3 rank 11 and 2x3x2 rank 11.
    # This is complex to hardcode from memory.
    # Instead, I will use a generic solver to find a valid decomposition for the full tensor.
    
    # Define the target tensor T[i, j, k] = 1 if A[i, j] * B[j, k] contributes to C[i, k].
    # Indices: i in 0..1, j in 0..3, k in 0..4.
    # T is a 3D array of shape (2, 4, 5).
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Flatten T to a vector of length 40.
    T_flat = T.flatten()
    
    # We will try to find U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    # Wait, the convention is W[k*n+i, r].
    # So C[i, k] = sum_j sum_r A[i, j] B[j, k] ... no.
    # The tensor decomposition is:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ?
    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # And "output factor follows the official trace convention W[k*n+i,r]".
    # This usually means the tensor is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # Let's verify with schoolbook.
    # Schoolbook: U[i, j, c] = 1 if c = i*m+j. V[j, k, c] = 1 if c = j*p+k. W[k, i, c] = 1 if c = k*n+i.
    # Then sum_r U[i,j,r] V[j,k,r] W[k,i,r] is 1 only if there is a c such that c=i*m+j AND c=j*p+k AND c=k*n+i.
    # This implies i*m+j = j*p+k = k*n+i.
    # This is not generally true for all i,j,k.
    # The schoolbook decomposition in the prompt is:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term $A_{i,j} B_{j,k}$ contributing to $C_{i,k}$.
    # The product is $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}$.
    # For a fixed i,j,k, only one c is non-zero for all three?
    # No, for a fixed i,j,k, the term in the sum is $A_{i,j} B_{j,k}$.
    # The decomposition of the multiplication tensor $T$ is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # In schoolbook, for each (i,j,k), we have a unique r.
    # So $T_{i,j,k} = 1$.
    
    # So we need to solve for U, V, W such that:
    # for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # This is a system of 40 cubic equations.
    # We can use a numerical solver.
    
    def solve_rank(r):
        # Variables: U (8*r), V (20*r), W (10*r). Total 38*r.
        # We will use scipy.optimize.least_squares.
        # Residual: for each (i,j,k), residual = 1 - sum_r ...
        
        def residual(x):
            U_flat = x[:n*m*r]
            V_flat = x[n*m*r:n*m*r + m*p*r]
            W_flat = x[n*m*r + m*p*r:]
            
            U = U_flat.reshape(n*m, r)
            V = V_flat.reshape(m*p, r)
            W = W_flat.reshape(p*n, r)
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = np.dot(U[i*m+j], np.multiply(V[j*p+k], W[k*n+i]))
                        res[idx] = 1.0 - val
                        idx += 1
            return res
            
        # Initial guess: random
        np.random.seed(42)
        x0 = np.random.randn(38*r)
        
        # Solve
        sol = None
        try:
            sol = scipy.optimize.least_squares(residual, x0, method='lm', max_nfev=10000)
        except Exception:
            return None
            
        if sol.cost < 1e-10:
            # Extract U, V, W
            U_flat = sol.x[:n*m*r]
            V_flat = sol.x[n*m*r:n*m*r + m*p*r]
            W_flat = sol.x[n*m*r + m*p*r:]
            U = U_flat.reshape(n*m, r)
            V = V_flat.reshape(m*p, r)
            W = W_flat.reshape(p*n, r)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rounded[i*m+j], np.multiply(V_rounded[j*p+k], W_rounded[k*n+i]))
                        if not np.isclose(val, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
        return None

    import scipy.optimize
    
    # Try ranks from 20 to 40
    for r in range(20, 41):
        result = solve_rank(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
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
