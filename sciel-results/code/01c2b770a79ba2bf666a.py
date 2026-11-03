import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor slices
    # T_c is a matrix of size nm x mp
    # T_c[a, b] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
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
    
    # The schoolbook decomposition has rank n*m*p = 30
    # We try to find a lower rank decomposition
    
    # For small dimensions, we can try to use known results or search
    # The lower bound for 2x3x5 is at least 15 (geometric mean)
    # Known: R(2,3,3) = 10, R(3,3,3) = 23 (Strassen-like)
    # For 2,3,5: we try ranks from 15 to 29
    
    # We use a structured approach:
    # Try to find U, V with entries in {-1, 0, 1} such that
    # the linear system for W has a solution with half-integer entries
    
    # To make this more efficient, we use the fact that for each c,
    # W[c, :] is determined by the equation:
    # B @ W[c, :] = T_slices[c]
    # where B[:, r] = vec(U[:, r] outer V[:, r])
    
    # We'll try random U, V but with a bias towards structured patterns
    
    for R in range(15, 30):
        for trial in range(500):
            # Generate U, V with entries in {-1, 0, 1}
            # Use a mix of random and structured patterns
            if trial % 3 == 0:
                # Pure random
                U = np.random.choice([-1, 0, 1], size=(nm, R))
                V = np.random.choice([-1, 0, 1], size=(mp, R))
            elif trial % 3 == 1:
                # Mostly zeros with some ones
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                for r in range(R):
                    # Pick random non-zero entries
                    nz_U = np.random.choice(nm, size=np.random.randint(2, 6), replace=False)
                    nz_V = np.random.choice(mp, size=np.random.randint(2, 6), replace=False)
                    U[nz_U, r] = np.random.choice([-1, 1], size=len(nz_U))
                    V[nz_V, r] = np.random.choice([-1, 1], size=len(nz_V))
            else:
                # Block structure
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                for r in range(R):
                    # Each column has a block of non-zeros
                    block_size = np.random.randint(2, 4)
                    start_U = np.random.randint(0, nm - block_size + 1)
                    start_V = np.random.randint(0, mp - block_size + 1)
                    U[start_U:start_U+block_size, r] = np.random.choice([-1, 1], size=block_size)
                    V[start_V:start_V+block_size, r] = np.random.choice([-1, 1], size=block_size)
            
            # Construct B matrix
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
                W_c, residuals, rank, sv = np.linalg.lstsq(B, T_slices[c], rcond=None)
                W[c, :] = W_c
                
                # Check if solution is exact
                if not np.allclose(B @ W[c, :], T_slices[c], atol=1e-10):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if U, V, W have integer/half-integer entries
            U_check = np.round(U * 2) / 2
            if not np.allclose(U, U_check, atol=1e-10):
                continue
            V_check = np.round(V * 2) / 2
            if not np.allclose(V, V_check, atol=1e-10):
                continue
            W_check = np.round(W * 2) / 2
            if not np.allclose(W, W_check, atol=1e-10):
                continue
            
            # Verify the decomposition
            for c in range(pn):
                reconstructed = sum(W_check[c, r] * np.outer(U_check[:, r], V_check[:, r]) for r in range(R)).flatten()
                if not np.allclose(reconstructed, T_slices[c], atol=1e-10):
                    valid = False
                    break
            
            if valid:
                return {
                    "U": U_check.tolist(),
                    "V": V_check.tolist(),
                    "W": W_check.tolist()
                }
    
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
# EVOLVE-BLOCK-END
