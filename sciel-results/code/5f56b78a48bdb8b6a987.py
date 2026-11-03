import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute T_slices: for each c in 0..pn-1, T_c is a flattened (nm x mp) matrix
    T_slices = []
    for c in range(pn):
        T_c = np.zeros((nm, mp))
        k = c // n
        i = c % n
        for j in range(m):
            a = i * m + j
            b = j * p + k
            T_c[a, b] = 1
        T_slices.append(T_c.flatten())
    
    # The known lower bound for R(2,3,5) is 15, but we start from a safe lower bound.
    # Let's try ranks from 15 up to 30.
    for R in range(15, 31):
        # We will try to find a decomposition of rank R.
        # Strategy:
        # 1. Randomly initialize U (nm x R) and V (mp x R) with entries from {-1, 0, 1}.
        # 2. Construct the matrix B (nm*mp x R) where B[:, r] = vec(U[:, r] * V[:, r]^T).
        # 3. For each c, solve B * W[c, :] = T_slices[c] using least squares.
        # 4. Check if the solution is exact (residual is close to 0).
        # 5. Check if U, V, W have integer or half-integer entries.
        
        for trial in range(5000):
            # Generate random U, V
            # Using a mix of -1, 0, 1 with some bias towards non-zero to improve rank
            U = np.random.choice([-1, 0, 1], size=(nm, R))
            V = np.random.choice([-1, 0, 1], size=(mp, R))
            
            # Ensure U and V are not trivially zero
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            # Construct B
            B = np.zeros((nm * mp, R))
            for r in range(R):
                outer = np.outer(U[:, r], V[:, r])
                B[:, r] = outer.flatten()
            
            # Check if B has full column rank
            if np.linalg.matrix_rank(B) < R:
                continue
            
            # Solve for W
            W = np.zeros((pn, R))
            valid = True
            for c in range(pn):
                # Solve B * W_c = T_slices[c]
                W_c, residuals, rank, sv = np.linalg.lstsq(B, T_slices[c], rcond=None)
                W[c, :] = W_c
                
                # Check if solution is exact
                if not np.allclose(B @ W[c, :], T_slices[c], atol=1e-9):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if U, V, W have integer/half-integer entries
            # We check if they are close to k/2 for some integer k
            U_check = np.round(U * 2) / 2
            if not np.allclose(U, U_check, atol=1e-9):
                continue
            V_check = np.round(V * 2) / 2
            if not np.allclose(V, V_check, atol=1e-9):
                continue
            W_check = np.round(W * 2) / 2
            if not np.allclose(W, W_check, atol=1e-9):
                continue
            
            # If we get here, we have a valid decomposition
            return {
                "U": U_check.tolist(),
                "V": V_check.tolist(),
                "W": W_check.tolist()
            }
        
        # If no solution found for this R, try next R

    # Fallback to schoolbook (rank = n*m*p = 30)
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
# EVOLVE-BLOCK-END
