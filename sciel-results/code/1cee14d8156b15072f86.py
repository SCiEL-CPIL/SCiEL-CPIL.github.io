import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _time_left(deadline):
    """Return remaining time before deadline."""
    return deadline - time.time()

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = delta."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    return np.allclose(recon, target, atol=tol)

def _schoolbook_block(n, m, p):
    """
    Generate schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W for this block.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Generate Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W of shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2 = (a10+a11)b00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1
    
    # M3 = a00(b01-b11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1
    
    # M4 = a11(b10-b00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[3, 3] += 1
    
    # M5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[1, 4] += 1; W[3, 4] += 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions (U, V, W) with offsets.
    This is a simplified helper. In the main function, we will build the full matrices directly.
    """
    pass

def _build_3x3x7_decomposition():
    """
    Construct the rank-60 decomposition for 3x3 * 3x7.
    Uses block decomposition:
    A = [[A00, A01], [A10, A11]] where A00 is 2x2, A01 is 2x1, A10 is 1x2, A11 is 1x1.
    B = [[B00], [B10]] where B00 is 2x7, B10 is 1x7.
    
    C = A00 B00 + A01 B10 + A10 B00 + A11 B10
    
    1. A00 B00 (2x2 * 2x7):
       Split B00 into B00_0 (2x1), B00_1 (2x2), B00_2 (2x2), B00_3 (2x2).
       - A00 B00_0: Schoolbook 2x2x1 (rank 4)
       - A00 B00_1: Strassen 2x2x2 (rank 7)
       - A00 B00_2: Strassen 2x2x2 (rank 7)
       - A00 B00_3: Strassen 2x2x2 (rank 7)
       Total: 25
    
    2. A01 B10 (2x1 * 1x7):
       Schoolbook 2x1x7 (rank 14)
    
    3. A10 B00 (1x2 * 2x7):
       Schoolbook 1x2x7 (rank 14)
    
    4. A11 B10 (1x1 * 1x7):
       Schoolbook 1x1x7 (rank 7)
    
    Total Rank: 25 + 14 + 14 + 7 = 60
    """
    n, m, p = 3, 3, 7
    total_rank = 60
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off):
        nonlocal col
        n_b, m_b = U_block.shape[0] // m_b, m_b # This logic is tricky with offsets.
        # Better to pass explicit indices
        
    # We will manually map the indices for each part.
    
    # Part 1: A00 B00
    # A00 indices: i in 0,1; j in 0,1.
    # B00 indices: j in 0,1; k in 0..6.
    
    # 1a. A00 * B00[:, 0] (2x2 * 2x1)
    # Schoolbook 2x2x1
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*3 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    
    # 1b. A00 * B00[:, 1:3] (2x2 * 2x2) -> Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map Strassen indices to global
    # Strassen U: row i*2+j. Global U: row i*3+j. (i,j in 0,1)
    # Strassen V: row j*2+k. Global V: row j*7 + (1+k). (j in 0,1, k in 0,1)
    # Strassen W: row k*2+i. Global W: row (1+k)*3 + i. (k in 0,1, i in 0,1)
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U[i*3+j, col+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V[j*7 + (1+k), col+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W[(1+k)*3 + i, col+r] += val
    col += 7
    
    # 1c. A00 * B00[:, 3:5] (2x2 * 2x2) -> Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U[i*3+j, col+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V[j*7 + (3+k), col+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W[(3+k)*3 + i, col+r] += val
    col += 7
    
    # 1d. A00 * B00[:, 5:7] (2x2 * 2x2) -> Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U[i*3+j, col+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V[j*7 + (5+k), col+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W[(5+k)*3 + i, col+r] += val
    col += 7
    
    # Part 2: A01 B10 (2x1 * 1x7)
    # A01 indices: i in 0,1; j=2.
    # B10 indices: j=2; k in 0..6.
    for i in range(2):
        for k in range(7):
            U[i*3 + 2, col] = 1
            V[2*7 + k, col] = 1
            W[k*3 + i, col] = 1
            col += 1
    
    # Part 3: A10 B00 (1x2 * 2x7)
    # A10 indices: i=2; j in 0,1.
    # B00 indices: j in 0,1; k in 0..6.
    for j in range(2):
        for k in range(7):
            U[2*3 + j, col] = 1
            V[j*7 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1
    
    # Part 4: A11 B10 (1x1 * 1x7)
    # A11 indices: i=2; j=2.
    # B10 indices: j=2; k in 0..6.
    for k in range(7):
        U[2*3 + 2, col] = 1
        V[2*7 + k, col] = 1
        W[k*3 + 2, col] = 1
        col += 1
        
    return U, V, W

def construct():
    """
    Returns a valid rank decomposition for 3x3 by 3x7 matrix multiplication.
    """
    n, m, p = 3, 3, 7
    deadline = time.time() + 100
    
    U, V, W = _build_3x3x7_decomposition()
    
    # Verify
    if _check_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if something went wrong
    rank = n * m * p
    U_s = np.zeros((n*m, rank))
    V_s = np.zeros((m*p, rank))
    W_s = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m + j, col] = 1
                V_s[j*p + k, col] = 1
                W_s[k*n + i, col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
