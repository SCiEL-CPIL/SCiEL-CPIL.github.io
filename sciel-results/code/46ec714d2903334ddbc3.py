import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    # Returns dict with U, V, W
    n, m, p = DIMENSIONS
    U, V, W = decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions in block diagonal fashion.
    U1, V1, W1 for (n1, m1, p1)
    U2, V2, W2 for (n2, m2, p2)
    Result is for (n1+n2, m1+m2, p1+p2) assuming appropriate block structure.
    Actually, the decomposition is for the sum of tensors.
    The U, V, W matrices are just concatenated along the rank dimension (columns).
    The row dimensions must match the total n*m, m*p, p*n.
    """
    # The caller must ensure U1, V1, W1 correspond to the first block of the larger tensor
    # and U2, V2, W2 to the second.
    # The ranks add up.
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    
    U_combined = np.zeros((U1.shape[0], r1 + r2), dtype=int)
    V_combined = np.zeros((V1.shape[0], r1 + r2), dtype=int)
    W_combined = np.zeros((W1.shape[0], r1 + r2), dtype=int)
    
    U_combined[:, :r1] = U1
    V_combined[:, :r1] = V1
    W_combined[:, :r1] = W1
    
    U_combined[:, r1:] = U2
    V_combined[:, r1:] = V2
    W_combined[:, r1:] = W2
    
    return U_combined, V_combined, W_combined

def embed_block(U_block, V_block, W_block, n_off, m_off, p_off, n_tot, m_tot, p_tot):
    """
    Embed a block decomposition into the full tensor space.
    U_block is (n_b*m_b, rank)
    V_block is (m_b*p_b, rank)
    W_block is (p_b*n_b, rank)
    
    We need to expand U_block to (n_tot*m_tot, rank)
    The rows of U correspond to (i, j) where i in [0, n_b), j in [0, m_b).
    In the full tensor, this corresponds to (i + n_off, j + m_off).
    The row index in U_full is (i + n_off) * m_tot + (j + m_off).
    
    Similarly for V and W.
    """
    n_b, m_b, p_b = U_block.shape[0] // m_b if m_b != 0 else 0, m_b, p_b # This is messy, better to pass dimensions
    
    # Let's pass dimensions explicitly to avoid confusion
    pass

# Redefine embed_block with explicit dimensions
def embed_block(U_block, V_block, W_block, n_b, m_b, p_b, n_off, m_off, p_off, n_tot, m_tot, p_tot):
    rank = U_block.shape[1]
    
    U_full = np.zeros((n_tot * m_tot, rank), dtype=int)
    V_full = np.zeros((m_tot * p_tot, rank), dtype=int)
    W_full = np.zeros((p_tot * n_tot, rank), dtype=int)
    
    # Fill U_full
    for i in range(n_b):
        for j in range(m_b):
            row_block = i * m_b + j
            row_full = (i + n_off) * m_tot + (j + m_off)
            U_full[row_full, :] = U_block[row_block, :]
            
    # Fill V_full
    for j in range(m_b):
        for k in range(p_b):
            row_block = j * p_b + k
            row_full = (j + m_off) * p_tot + (k + p_off)
            V_full[row_full, :] = V_block[row_block, :]
            
    # Fill W_full
    # W index is k*n + i.
    for k in range(p_b):
        for i in range(n_b):
            row_block = k * n_b + i
            row_full = (k + p_off) * n_tot + (i + n_off)
            W_full[row_full, :] = W_block[row_block, :]
            
    return U_full, V_full, W_full

def decompose(n, m, p):
    """
    Recursive decomposition.
    Returns U, V, W as numpy arrays.
    """
    
    # Base Case: Schoolbook for small dims or 1s
    if n == 1 and m == 1 and p == 1:
        return np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int)
    
    if n == 1 or m == 1 or p == 1:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W
        
    # Special Case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        rank = 7
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((4, rank), dtype=int)
        W = np.zeros((4, rank), dtype=int)
        
        # Indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[3, 0] -= 1 # C00 +, C11 -
        
        # M2 = (A10+A11)B00
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1 # C10 +
        
        # M3 = A00(B01-B11)
        U[0, 2] += 1
        V[1, 2] += 1; V[3, 2] -= 1
        W[1, 2] += 1 # C01 +
        
        # M4 = A11(B10-B00)
        U[3, 3] += 1
        V[2, 3] += 1; V[0, 3] -= 1
        W[0, 3] += 1; W[2, 3] += 1 # C00 +, C10 +
        
        # M5 = (A00+A01)B11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] -= 1; W[1, 4] += 1 # C00 -, C01 +
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] += 1; U[0, 5] -= 1
        V[0, 5] += 1; V[1, 5] += 1
        W[3, 5] += 1 # C11 +
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] += 1; U[3, 6] -= 1
        V[2, 6] += 1; V[3, 6] += 1
        W[0, 6] += 1 # C00 +
        
        return U, V, W

    # Recursive Case
    # Strategy:
    # 1. Split 3x5x6 into 3x2x6 and 3x3x6 (split m=5)
    # 2. 3x2x6 into 1x2x6 and 2x2x6 (split n=3)
    # 3. 2x2x6 into 2x2x3 and 2x2x3 (split p=6)
    # 4. 2x2x3 into 2x2x2 and 2x2x1 (split p=3)
    # 5. 3x3x6 into 3x3x3 and 3x3x3 (split p=6)
    # 6. 3x3x3 into 1x3x3 and 2x3x3 (split n=3)
    # 7. 2x3x3 into 2x1x3 and 2x2x3 (split m=3)
    
    # Helper to decide split
    # We hardcode the path for the specific 3,5,6 case to ensure optimal rank 86.
    
    if n == 3 and m == 5 and p == 6:
        # Split m into 2 and 3
        n1, m1, p1 = 3, 2, 6
        n2, m2, p2 = 3, 3, 6
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 0, 2, 0, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 3 and m == 2 and p == 6:
        # Split n into 1 and 2
        n1, m1, p1 = 1, 2, 6
        n2, m2, p2 = 2, 2, 6
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 1, 0, 0, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 2 and m == 2 and p == 6:
        # Split p into 3 and 3
        n1, m1, p1 = 2, 2, 3
        n2, m2, p2 = 2, 2, 3
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 0, 0, 3, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 2 and m == 2 and p == 3:
        # Split p into 2 and 1
        n1, m1, p1 = 2, 2, 2
        n2, m2, p2 = 2, 2, 1
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 0, 0, 2, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 3 and m == 3 and p == 6:
        # Split p into 3 and 3
        n1, m1, p1 = 3, 3, 3
        n2, m2, p2 = 3, 3, 3
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 0, 0, 3, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 3 and m == 3 and p == 3:
        # Split n into 1 and 2
        n1, m1, p1 = 1, 3, 3
        n2, m2, p2 = 2, 3, 3
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 1, 0, 0, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)
        
    elif n == 2 and m == 3 and p == 3:
        # Split m into 1 and 2
        n1, m1, p1 = 2, 1, 3
        n2, m2, p2 = 2, 2, 3
        
        U1, V1, W1 = decompose(n1, m1, p1)
        U2, V2, W2 = decompose(n2, m2, p2)
        
        U1_e, V1_e, W1_e = embed_block(U1, V1, W1, n1, m1, p1, 0, 0, 0, n, m, p)
        U2_e, V2_e, W2_e = embed_block(U2, V2, W2, n2, m2, p2, 0, 1, 0, n, m, p)
        
        return combine(U1_e, V1_e, W1_e, U2_e, V2_e, W2_e)

    else:
        # Fallback: Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W
