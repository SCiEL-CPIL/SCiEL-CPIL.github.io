"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys

def _combine_factors(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions into one."""
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We concatenate along the rank dimension (columns)
    import numpy as np
    U = np.hstack((U1, U2))
    V = np.hstack((V1, V2))
    W = np.hstack((W1, W2))
    return U, V, W

def _schoolbook_factors(n, m, p):
    """Generate schoolbook factors for n x m x p."""
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

def _strassen_2x2x2_factors():
    """Generate Strassen factors for 2x2x2. Rank 7."""
    # U, V, W matrices of shape (4, 7)
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Indices for U (A factors): 0:a11, 1:a12, 2:a21, 3:a22
    # Indices for V (B factors): 0:b11, 1:b12, 2:b21, 3:b22
    # Indices for W (C factors): 0:c11, 1:c12, 2:c21, 3:c22
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (a21 + a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = a11(b12 - b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21 - b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11 + a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (a21 - a11)(b11 + b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = -1
    
    # M7 = (a12 - a22)(b21 + b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1
    
    return U, V, W

def _get_factors(n, m, p):
    """Recursively get factors for n x m x p."""
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2_factors()
    
    # Choose a dimension to split if possible, prefer splitting larger dims into 2+rest
    # If all dims are small (<=2), use schoolbook
    if n <= 2 and m <= 2 and p <= 2:
        return _schoolbook_factors(n, m, p)
        
    # Split p if p > 2
    if p > 2:
        p1 = min(2, p)
        p2 = p - p1
        U1, V1, W1 = _get_factors(n, m, p1)
        U2, V2, W2 = _get_factors(n, m, p2)
        # V1 corresponds to m x p1, V2 to m x p2.
        # We need to embed them in m x p space.
        # V1 indices: j*p1 + k. V2 indices: j*p1 + (k+p1).
        # This is tricky with simple hstack because V1 and V2 have different row structures?
        # No, V is (m*p, rank).
        # For sub-problem 1: V1 is (m*p1, rank1). We need to expand to (m*p, rank1) by placing at rows j*p1..j*p1+p1-1.
        # For sub-problem 2: V2 is (m*p2, rank2). Place at rows j*p1..
        
        V_full = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
        # Fill V_full for part 1
        for j in range(m):
            start = j * p
            V_full[start:start+p1, 0:U1.shape[1]] = V1[j*p1:(j+1)*p1, :]
        # Fill V_full for part 2
        for j in range(m):
            start = j * p + p1
            V_full[start:start+p2, U1.shape[1]:] = V2[j*p2:(j+1)*p2, :]
            
        # U is same n*m, just hstack
        U = np.hstack((U1, U2))
        
        # W is (p*n, rank).
        # W1 is (p1*n, rank1). W2 is (p2*n, rank2).
        W_full = np.zeros((p * n, U1.shape[1] + U2.shape[1]))
        for i in range(n):
            start1 = i * p
            W_full[start1:start1+p1, 0:U1.shape[1]] = W1[i*p1:(i+1)*p1, :]
            start2 = i * p + p1
            W_full[start2:start2+p2, U1.shape[1]:] = W2[i*p2:(i+1)*p2, :]
            
        return U, V_full, W_full

    # Split m if m > 2
    if m > 2:
        m1 = min(2, m)
        m2 = m - m1
        U1, V1, W1 = _get_factors(n, m1, p)
        U2, V2, W2 = _get_factors(n, m2, p)
        
        # U is (n*m, rank).
        U_full = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
        for i in range(n):
            start1 = i * m
            U_full[start1:start1+m1, 0:U1.shape[1]] = U1[i*m1:(i+1)*m1, :]
            start2 = i * m + m1
            U_full[start2:start2+m2, U1.shape[1]:] = U2[i*m2:(i+1)*m2, :]
            
        # V is (m*p, rank).
        V_full = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
        for j in range(p):
            start1 = 0 # V index is j*p + k? No, V index is (row of A's col, col of B's row)?
            # V is (m*p, rank). Index is j*p + k.
            # Sub-problem 1 uses m1 rows. Sub-problem 2 uses m2 rows.
            # Wait, V corresponds to B. B is m x p.
            # V rows are indexed by (j, k) where j is 0..m-1, k is 0..p-1.
            # In sub-problem 1, m is m1. So V1 rows are j*p1 + k? No, V1 is (m1*p, rank1).
            # We need to map V1 rows to V_full rows.
            # V1 row j*p + k (where j in 0..m1-1) maps to V_full row j*p + k.
            # V2 row j'*p + k (where j' in 0..m2-1) maps to V_full row (j'+m1)*p + k.
            for k in range(p):
                # Part 1
                V_full[0:m1*p, k::p] # This is wrong, V is dense.
                # Let's just do it simply.
                pass
            
        # Simpler approach for V embedding:
        V_full = np.zeros((m * p, U1.shape[1] + U2.shape[1]))
        for j in range(m1):
            for k in range(p):
                V_full[j*p+k, 0:U1.shape[1]] = V1[j*p+k, :]
        for j in range(m2):
            for k in range(p):
                V_full[(j+m1)*p+k, U1.shape[1]:] = V2[j*p+k, :]
                
        # W is (p*n, rank). Same for both, so hstack.
        W = np.hstack((W1, W2))
        
        return U_full, V_full, W

    # Split n if n > 2
    if n > 2:
        n1 = min(2, n)
        n2 = n - n1
        U1, V1, W1 = _get_factors(n1, m, p)
        U2, V2, W2 = _get_factors(n2, m, p)
        
        # U is (n*m, rank).
        U_full = np.zeros((n * m, U1.shape[1] + U2.shape[1]))
        for i in range(n1):
            for j in range(m):
                U_full[i*m+j, 0:U1.shape[1]] = U1[i*m+j, :]
        for i in range(n2):
            for j in range(m):
                U_full[(i+n1)*m+j, U1.shape[1]:] = U2[i*m+j, :]
                
        # V is (m*p, rank). Same for both, so hstack.
        V = np.hstack((V1, V2))
        
        # W is (p*n, rank).
        W_full = np.zeros((p * n, U1.shape[1] + U2.shape[1]))
        for k in range(p):
            for i in range(n1):
                W_full[k*n+i, 0:U1.shape[1]] = W1[k*n+i, :]
            for i in range(n2):
                W_full[k*n+(i+n1), U1.shape[1]:] = W2[k*n+i, :]
                
        return U_full, V, W_full

    # Fallback
    return _schoolbook_factors(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _get_factors(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
