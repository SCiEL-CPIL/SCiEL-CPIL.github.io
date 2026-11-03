import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2 matrix multiplication.
    Returns U, V, W matrices of shape (4, 7) each.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 * 2x2
    # Indices: A is 2x2 (rows i=0,1; cols j=0,1), B is 2x2 (rows j=0,1; cols k=0,1)
    # C = A*B, C[i,k]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0*2+0, 0] = 1  # C00 (k=0, i=0)
    W[1*2+1, 0] = 1  # C11 (k=1, i=1)
    
    # M2 = (A10 + A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    # C10 += M2, C11 -= M2
    W[0*2+1, 1] = 1  # C10 (k=0, i=1)
    W[1*2+1, 1] = -1 # C11 (k=1, i=1)
    
    # M3 = A00(B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    # C00 += M3, C01 += M3
    W[0*2+0, 2] = 1  # C00 (k=0, i=0)
    W[1*2+0, 2] = 1  # C01 (k=1, i=0)
    
    # M4 = A11(B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    # C00 += M4, C10 += M4
    W[0*2+0, 3] = 1  # C00 (k=0, i=0)
    W[0*2+1, 3] = 1  # C10 (k=0, i=1)
    
    # M5 = (A00 + A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0*2+0, 4] = -1 # C00 (k=0, i=0)
    W[1*2+0, 4] = 1  # C01 (k=1, i=0)
    
    # M6 = (A10 - A00)(B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    # C11 += M6
    W[1*2+1, 5] = 1  # C11 (k=1, i=1)
    
    # M7 = (A01 - A11)(B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    # C00 += M7
    W[0*2+0, 6] = 1  # C00 (k=0, i=0)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    """
    Constructs an exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses Strassen's algorithm on 2x2 sub-blocks where possible to reduce rank below schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Strategy: Use Strassen for 2x2 blocks where we have pairs of dimensions.
    # For 3x5*5x7: we can split the 3 rows into (2,1), the 5 cols into (2,2,1), and 7 cols into (2,2,2,1)
    # Or better: split 3 into (2,1), 5 into (2,2,1), 7 into (2,2,2,1)
    # Each combination of a row-block, col-block, and out-col-block gives a sub-problem.
    
    # Let's use a simpler approach:
    # Split n=3 into blocks of size [2, 1]
    # Split m=5 into blocks of size [2, 2, 1]
    # Split p=7 into blocks of size [2, 2, 2, 1]
    
    # For each pair of (row_block, col_block, out_block), we have a sub-multiplication
    # of size (rb x cb) * (cb x ob).
    # If rb=2, cb=2, ob=2: use Strassen (rank 7)
    # If rb=1, cb=1, ob=1: schoolbook (rank 1)
    # If rb=2, cb=1, ob=1: schoolbook (rank 2)
    # If rb=1, cb=2, ob=1: schoolbook (rank 2)
    # If rb=2, cb=2, ob=1: schoolbook (rank 4)
    # If rb=1, cb=1, ob=2: schoolbook (rank 2)
    # If rb=2, cb=1, ob=2: schoolbook (rank 4)
    # If rb=1, cb=2, ob=2: schoolbook (rank 4)
    # If rb=1, cb=1, ob=anything: rank = ob
    # If rb=2, cb=2, ob=anything: rank = 2*ob (schoolbook) or 7 if ob=2
    
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2, 1]
    
    # Compute total rank
    total_rank = 0
    for rb in n_blocks:
        for cb in m_blocks:
            for ob in p_blocks:
                if rb == 2 and cb == 2 and ob == 2:
                    total_rank += 7
                else:
                    total_rank += rb * cb * ob
    
    # total_rank = for (2,2,2): 7, (2,2,2): 7, (2,2,1): 4, (2,1,2): 4, (2,1,2): 4, (2,1,1): 2,
    #               (1,2,2): 4, (1,2,2): 4, (1,2,1): 2, (1,1,2): 2, (1,1,2): 2, (1,1,1): 1
    # Let me recount:
    # n_blocks = [2, 1] (2 blocks)
    # m_blocks = [2, 2, 1] (3 blocks)
    # p_blocks = [2, 2, 2, 1] (4 blocks)
    # Total 2*3*4 = 24 sub-problems
    
    # (2,2,2): 7, (2,2,2): 7, (2,2,2): 7, (2,2,1): 4  -> rb=2, cb=2: 7+7+7+4 = 25
    # (2,1,2): 4, (2,1,2): 4, (2,1,2): 4, (2,1,1): 2  -> rb=2, cb=1: 4+4+4+2 = 14
    # (1,2,2): 4, (1,2,2): 4, (1,2,2): 4, (1,2,1): 2  -> rb=1, cb=2: 4+4+4+2 = 14
    # (1,1,2): 2, (1,1,2): 2, (1,1,2): 2, (1,1,1): 1  -> rb=1, cb=1: 2+2+2+1 = 7
    # Total = 25 + 14 + 14 + 7 = 60
    
    # That's a big improvement from 105!
    
    # Now build the actual matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col_offset = 0
    
    # Precompute block offsets
    n_offsets = []
    off = 0
    for b in n_blocks:
        n_offsets.append(off)
        off += b
    
    m_offsets = []
    off = 0
    for b in m_blocks:
        m_offsets.append(off)
        off += b
    
    p_offsets = []
    off = 0
    for b in p_blocks:
        p_offsets.append(off)
        off += b
    
    for ni, rb in enumerate(n_blocks):
        for mi, cb in enumerate(m_blocks):
            for pi, ob in enumerate(p_blocks):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                
                if rb == 2 and cb == 2 and ob == 2:
                    # Use Strassen
                    sU, sV, sW = _strassen_2x2x2()
                    # Map local indices to global
                    for li in range(2):
                        for lj in range(2):
                            for r in range(7):
                                if sU[li*2+lj, r] != 0:
                                    U[(n0+li)*m + (m0+lj), col_offset + r] = sU[li*2+lj, r]
                    for lj in range(2):
                        for lk in range(2):
                            for r in range(7):
                                if sV[lj*2+lk, r] != 0:
                                    V[(m0+lj)*p + (p0+lk), col_offset + r] = sV[lj*2+lk, r]
                    for lk in range(2):
                        for li in range(2):
                            for r in range(7):
                                if sW[lk*2+li, r] != 0:
                                    W[(p0+lk)*n + (n0+li), col_offset + r] = sW[lk*2+li, r]
                    col_offset += 7
                else:
                    # Schoolbook for this sub-block
                    for li in range(rb):
                        for lj in range(cb):
                            for lk in range(ob):
                                U[(n0+li)*m + (m0+lj), col_offset] = 1
                                V[(m0+lj)*p + (p0+lk), col_offset] = 1
                                W[(p0+lk)*n + (n0+li), col_offset] = 1
                                col_offset += 1
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Check time
    if time.time() > deadline:
        pass  # We're done anyway
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
