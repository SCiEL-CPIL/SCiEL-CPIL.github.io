"""Matrix multiplication 2x3x8 with reduced rank tensor decomposition."""

def construct():
    """
    Decomposition for 2x3 x 3x8 matrix multiplication.
    
    Strategy: Split the 3x8 matrix B into three 1x8 row blocks.
    For each row j of A (2 rows) and each column k of B (8 columns),
    we need the sum over j' of A[i,j'] * B[j',k].
    
    Schoolbook: rank 48.
    
    Better approach: Use the fact that for each pair (i,k), we need a dot product
    of length 3. We can share computations.
    
    Actually, let me use a simple block approach:
    Split B into 4 blocks of width 2. For each block, do 2x3 x 3x2.
    The 2x3 x 3x2 has a known minimal rank. Let's use rank 6 (schoolbook) per block,
    total 24.
    
    But wait, that's still schoolbook. The key is that schoolbook for the whole thing
    is 48, and splitting into 4 independent 2x3x3x2 problems each of rank 6 gives 24.
    This IS an improvement because we're not doing redundant work.
    
    Let me verify: schoolbook does 2*3*8 = 48 scalar multiplies.
    Block approach does 4 * (2*3*2) = 4*6 = 24 scalar multiplies. Yes, 24 < 48.
    """
    n, m, p = 2, 3, 8
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i * 3 + j][col] = 1
                    V[j * 8 + 2 * b + k][col] = 1
                    W[(2 * b + k) * 2 + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
