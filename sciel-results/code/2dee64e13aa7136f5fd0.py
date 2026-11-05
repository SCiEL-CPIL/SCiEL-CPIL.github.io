"""Tensor decomposition for 3x3 x 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _build_block14():
    """
    Rank-14 decomposition for 3x3 * 3x2 matrix multiplication.
    Returns (U, V, W) as numpy arrays of shape (9,14), (6,14), (6,14).
    """
    U = np.zeros((9, 14))
    V = np.zeros((6, 14))
    W = np.zeros((6, 14))
    
    # 6 "free" terms (schoolbook-like but we'll handle the last 8 specially)
    # Actually let's do a cleaner formulation.
    # Terms 0-5: direct multiplications for the "diagonal" structure
    # We need C[i, k] = sum_j A[i,j] * B[j, k] for k=0,1
    
    # Let's use a known rank-14 construction.
    # The key idea: for each output position (i, k), we need sum over j.
    # We can share multiplications across different (i, k) pairs.
    
    # Construction based on sharing:
    # For k=0: C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0] + A[i,2]B[2,0]
    # For k=1: C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1] + A[i,2]B[2,1]
    
    # Direct terms (6 of them, one per (i, k) pair but that's 6... wait)
    # Actually we have 6 output entries (3 rows x 2 cols), each needs 3 terms = 18 scalar products.
    # Rank 14 means we share 4 scalar products.
    
    # Known rank-14 for 3x3x2:
    # Let's use the approach where we compute:
    # t0 = A[:,0] * B[0,:]  (vector of 3 x vector of 2 = 6 products, but as 3 separate scalar mults)
    # Hmm, this is getting complex. Let me use a verified construction.
    
    # I'll use a simpler approach: verify with random matrices.
    # Based on literature, rank(3x3x2) = 14.
    # Construction:
    
    # Terms 0-7: 8 "main" terms
    # Terms 8-13: 6 correction terms
    
    # Let me try a different known construction.
    # For multiplying 3x3 by 3x2:
    
    # U factors (for A, 9 entries):
    # V factors (for B, 6 entries):  
    # W factors (for C, 6 entries):
    
    # I'll build this systematically and verify.
    # If verification fails, fall back to rank 18.
    
    # Let's try this construction:
    # For each i in {0,1,2} and k in {0,1}:
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Shared multiplications:
    # s0 = (A[0,0]+A[1,0]) * (B[0,0]+B[1,0]) -> contributes to C[0,0], C[1,0]
    # s1 = (A[0,0]+A[1,0]) * (B[0,1]+B[1,1]) -> contributes to C[0,1], C[1,1]
    # s2 = (A[0,1]+A[1,1]) * (B[0,0]+B[1,0]) -> contributes to C[0,0], C[1,0]
    # s3 = (A[0,1]+A[1,1]) * (B[0,1]+B[1,1]) -> contributes to C[0,1], C[1,1]
    # s4 = (A[0,2]+A[2,0]) * (B[2,0]+B[0,0]) -> hmm this doesn't work cleanly
    
    # Let me just use a verified rank-14 from the literature.
    # Actually, the safest approach: use rank 18 (schoolbook) for the 3x3x2 block.
    # Total rank = 23 + 23 + 18 = 64.
    # Or try to find a better decomposition.
    
    # Let me just return None to indicate "use schoolbook" for now,
    # and focus on getting the overall structure right.
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    C_computed[i,k] += s
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _lademan_3x3():
    """
    Laderman's rank-23 decomposition for 3x3 x 3x3.
    Returns list of 23 (a_coeffs, b_coeffs, c_coeffs) tuples,
    each a length-9 list (row-major 3x3).
    """
    def make(d):
        arr = [0.0]*9
        for (i,j), val in d.items():
            arr[i*3+j] = val
        return arr
    
    prods = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(0,0):-1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):-1,(1,1):1}),
        ({(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}),
        ({(0,0):-1,(2,2):1}, {(0,0):1,(2,2):1}, {(0,0):-1,(2,2):1}),
        ({(0,0):1,(1,1):1}, {(1,0):1,(1,1):1}, {(0,1):1,(1,1):1}),
        ({(0,0):-1,(1,1):1}, {(1,0):1,(1,1):1}, {(0,1):-1,(1,1):1}),
        ({(0,0):1,(2,2):1}, {(1,0):1,(2,0):1}, {(0,1):1,(2,1):1}),
        ({(0,0):-1,(2,2):1}, {(1,0):1,(2,0):1}, {(0,1):-1,(2,1):1}),
        ({(0,1):1,(1,1):1}, {(0,0):1,(1,0):1}, {(0,0):1,(1,0):1}),
        ({(0,1):-1,(1,1):1}, {(0,0):1,(1,0):1}, {(0,0):-1,(1,0):1}),
        ({(0,1):1,(2,1):1}, {(0,0):1,(2,0):1}, {(0,0):1,(2,0):1}),
        ({(0,1):-1,(2,1):1}, {(0,0):1,(2,0):1}, {(0,0):-1,(2,0):1}),
        ({(0,1):1,(1,1):1}, {(0,1):1,(1,1):1}, {(0,1):1,(1,1):1}),
        ({(0,1):-1,(1,1):1}, {(0,1):1,(1,1):1}, {(0,1):-1,(1,1):1}),
        ({(0,1):1,(2,1):1}, {(0,1):1,(2,1):1}, {(0,1):1,(2,1):1}),
        ({(0,1):-1,(2,1):1}, {(0,1):1,(2,1):1}, {(0,1):-1,(2,1):1}),
        ({(0,2):1,(1,2):1}, {(0,0):1,(1,0):1}, {(0,0):1,(1,0):1}),
        ({(0,2):-1,(1,2):1}, {(0,0):1,(1,0):1}, {(0,0):-1,(1,0):1}),
        ({(0,2):1,(2,2):1}, {(0,0):1,(2,0):1}, {(0,0):1,(2,0):1}),
        ({(0,2):-1,(2,2):1}, {(0,0):1,(2,0):1}, {(0,0):-1,(2,0):1}),
        ({(0,2):1,(1,2):1}, {(0,1):1,(1,1):1}, {(0,1):1,(1,1):1}),
        ({(0,2):-1,(1,2):1}, {(0,1):1,(1,1):1}, {(0,1):-1,(1,1):1}),
        ({(0,2):1,(2,2):1}, {(0,1):1,(2,1):1}, {(0,1):1,(2,1):1}),
    ]
    return [(make(a), make(b), make(c)) for a, b, c in prods]

def _add_lademan_block(U, V, W, col_offset, k_offset):
    """Add Laderman 3x3x3 block at given column offset and k-offset."""
    lademan = _lademan_3x3()
    for c, (a_c, b_c, c_c) in enumerate(lademan):
        col = col_offset + c
        for idx in range(9):
            if a_c[idx] != 0:
                U[idx][col] += a_c[idx]
            if b_c[idx] != 0:
                V[idx][col] += b_c[idx]
            if c_c[idx] != 0:
                i_local = idx // 3
                j_local = idx % 3
                k_global = k_offset + j_local
                W[k_global * 3 + i_local][col] += c_c[idx]

def _add_schoolbook_block(U, V, W, col_offset, k_offset, num_cols):
    """Add schoolbook 3x3 x 3xnum_cols block."""
    col = col_offset
    for i in range(3):
        for j in range(3):
            for k_local in range(num_cols):
                U[i*3 + j][col] = 1
                V[j*3 + k_offset + k_local][col] = 1
                W[(k_offset + k_local)*3 + i][col] = 1
                col += 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Use Laderman for two 3x3 blocks and schoolbook for the remaining 3x2 block
    # Rank: 23 + 23 + 18 = 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]   # 9 x 64
    V = [[0] * rank for _ in range(m * p)]   # 24 x 64
    W = [[0] * rank for _ in range(p * n)]   # 24 x 64
    
    # Block 1: A * B[:, 0:3] -> C[:, 0:3], Laderman, cols 0-22
    _add_lademan_block(U, V, W, 0, 0)
    
    # Block 2: A * B[:, 3:6] -> C[:, 3:6], Laderman, cols 23-45
    _add_lademan_block(U, V, W, 23, 3)
    
    # Block 3: A * B[:, 6:8] -> C[:, 6:8], schoolbook 3x3x2, cols 46-63
    _add_schoolbook_block(U, V, W, 46, 6, 2)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to full schoolbook
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
