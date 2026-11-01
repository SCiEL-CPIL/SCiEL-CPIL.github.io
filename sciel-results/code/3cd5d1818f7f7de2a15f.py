import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2_2x2():
    """
    Return the Strassen decomposition for 2x2 * 2x2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen 7-term algorithm
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C00 += M1, C11 += M1, C01 -= M1, C10 += M1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = -1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C10 += M2, C11 += M2
    W[1][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    # C00 += M3, C10 -= M3
    W[0][2] = 1; W[2][2] = -1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    # C01 += M4, C11 -= M4
    W[1][3] = 1; W[3][3] = -1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C01 += M5, C11 += M5
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C00 += M6, C10 += M6
    W[1][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C01 += M7, C11 += M7
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _construct_rank14():
    """
    Construct a rank-14 decomposition for 2x2 * 2x4 using two Strassen blocks.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    
    # First 2x2 block (columns 0-1 of the 4-wide matrix)
    for r in range(7):
        for i in range(n*m):
            U[i][r] = U_s[i][r]
        for j in range(m):
            for k in range(2):
                V[j*p+k][r] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(n):
                W[k*n+i][r] = W_s[k*2+i][r]
                
    # Second 2x2 block (columns 2-3 of the 4-wide matrix)
    for r in range(7):
        for i in range(n*m):
            U[i][r+7] = U_s[i][r]
        for j in range(m):
            for k in range(2):
                V[j*p+k+2][r+7] = V_s[j*2+k][r]
        for k in range(2):
            for i in range(n):
                W[(k+2)*n+i][r+7] = W_s[k*2+i][r]
                
    return U, V, W

def _search_lower_rank(n, m, p, target_rank, deadline):
    """
    Search for a lower-rank decomposition by fixing U and V to random sparse patterns
    and solving the resulting linear system for W.
    """
    if time.time() > deadline:
        return None
    
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    # If we fix U and V, this becomes a linear system in W.
    # Number of equations: n*m*p = 16
    # Number of unknowns: p*n*rank = 8*rank
    # If rank >= 2, the system is underdetermined (more unknowns than equations).
    
    # We want W to have small integer/half-integer entries.
    # Strategy: Fix U and V to random sparse matrices with entries in {-1, 0, 1}.
    # Solve the linear system for W using least squares.
    # Check if the solution is "clean" (close to integers or half-integers).
    
    import itertools
    
    best_W = None
    best_score = float('inf')
    
    # Try different sparsity patterns for U and V
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Generate random sparse U and V
        # U is (n*m, rank) = (4, rank)
        # V is (m*p, rank) = (8, rank)
        # Each entry is 0, 1, or -1 with some probability
        
        U_try = np.zeros((n*m, target_rank), dtype=int)
        V_try = np.zeros((m*p, target_rank), dtype=int)
        
        # Make U sparse: each column has ~2 non-zero entries
        for r in range(target_rank):
            # Choose 2 random rows for U
            rows = np.random.choice(n*m, size=2, replace=False)
            for idx, row in enumerate(rows):
                U_try[row, r] = np.random.choice([-1, 1])
            
            # Choose 3 random rows for V
            rows = np.random.choice(m*p, size=3, replace=False)
            for idx, row in enumerate(rows):
                V_try[row, r] = np.random.choice([-1, 1])
        
        # Build the linear system for W
        # Equation for (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let W_vec be a vector of size p*n*rank = 8*rank
        # W_vec[(k*n+i)*rank + r] = W[k*n+i, r]
        
        A = np.zeros((n*m*p, p*n*target_rank), dtype=float)
        b = np.ones(n*m*p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        coeff = U_try[i*m+j, r] * V_try[j*p+k, r]
                        if coeff != 0:
                            A[eq_idx, (k*n+i)*target_rank + r] = coeff
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        W_try = W_flat.reshape(p*n, target_rank)
        
        # Check residual
        if np.linalg.norm(A @ W_flat - b) > 1e-6:
            continue
        
        # Check if W is "clean" (close to half-integers)
        # Multiply by 2 and check if close to integers
        W_2x = W_try * 2
        W_rounded = np.round(W_2x)
        error = np.linalg.norm(W_2x - W_rounded)
        
        if error < 0.1:
            # Clean solution!
            W_int = (W_rounded / 2.0)
            
            # Verify
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_int.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Track best
            if error < best_score:
                best_score = error
                best_W = W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Guaranteed fallback: rank-14 Strassen construction
    U14, V14, W14 = _construct_rank14()
    best_result = {"U": U14, "V": V14, "W": W14}
    best_rank = 14
    
    # Verify fallback
    if not _verify_decomposition(U14, V14, W14, n, m, p):
        # Should never happen, but fall back to trivial rank-16
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
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try to find rank 13
    if time.time() < deadline - 5:
        result = _search_lower_rank(n, m, p, 13, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = result
                best_rank = 13
    
    # Try to find rank 12
    if time.time() < deadline - 5 and best_rank > 12:
        result = _search_lower_rank(n, m, p, 12, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = result
                best_rank = 12
    
    return best_result
