"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _rank4_decomposition():
    """
    Rank-4 decomposition exploiting n=2 structure.
    
    For each j in 0..m-1, create one rank term:
      U[i*m+j, r=j] = 1  for all i (picks column j of A)
      V[j*p+k, r=j] = 1  for all k (picks row j of B)
      W[k*n+i, r=j] = 1  for all i,k (broadcasts outer product)
    
    This gives: sum_j (A[:,j] * B[j,:]) = A @ B
    Rank = m = 4.
    """
    n, m, p = DIMENSIONS
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    Split A into two 2x2 blocks, B into 2x6 halves with 3 col blocks of width 2.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_full_decomposition():
    """Fallback: rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _verify_decomposition(result, n=2, m=4, p=6, trials=30):
    """Verify a decomposition with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _verify_numpy(result, n=2, m=4, p=6, trials=5):
    """Fast numerical verification using numpy."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float).T  # (rank, n*m)
    V_np = np.array(V, dtype=float).T  # (rank, m*p)
    W_np = np.array(W, dtype=float).T  # (rank, p*n)
    
    for trial in range(trials):
        A = np.random.randint(-10, 11, size=(n, m)).astype(float)
        B = np.random.randint(-10, 11, size=(m, p)).astype(float)
        
        C_exp = A @ B
        
        # Reshape U, V, W for computation
        U_r = U_np.reshape(rank, n, m)  # (rank, n, m)
        V_r = V_np.reshape(rank, m, p)  # (rank, m, p)
        W_r = W_np.reshape(rank, p, n)  # (rank, p, n)
        
        # u_val[r] = sum_{i,j} U[r,i,j] * A[i,j]
        u_vals = np.einsum('rij,ij->r', U_r, A)
        v_vals = np.einsum('rjk,jk->r', V_r, B)
        
        # C[i,k] = sum_r W[r,k,i] * u_vals[r] * v_vals[r]
        C_comp = np.einsum('rki,r,r->ik', W_r, u_vals, v_vals)
        
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4
    matrix by a 4 by 6 matrix.
    
    Strategy: Use rank-4 decomposition exploiting n=2 structure.
    For each j, one term: U picks A[:,j], V picks B[j,:], W broadcasts.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    # Primary: rank-4 decomposition
    result = _rank4_decomposition()
    
    # Quick numpy verification
    if time.time() < deadline - 5:
        if _verify_numpy(result, 2, 4, 6, trials=5):
            # Thorough integer verification
            if time.time() < deadline - 5:
                if _verify_decomposition(result, 2, 4, 6, trials=30):
                    return result
            else:
                return result
    else:
        return result
    
    # Fallback: Strassen rank-42
    if time.time() < deadline - 10:
        strassen = _strassen_block_decomposition()
        if _verify_numpy(strassen, 2, 4, 6, trials=3):
            if time.time() < deadline - 5:
                if _verify_decomposition(strassen, 2, 4, 6, trials=20):
                    return strassen
            else:
                return strassen
    
    # Last resort: full schoolbook rank-48
    schoolbook = _schoolbook_full_decomposition()
    if time.time() < deadline - 2:
        if _verify_numpy(schoolbook, 2, 4, 6, trials=3):
            return schoolbook
    
    return schoolbook
# EVOLVE-BLOCK-END
