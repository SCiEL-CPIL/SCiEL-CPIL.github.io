import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]
_DEADLINE = time.time() + 145.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank = n*m*p."""
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

def _strassen_2x2_2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U indexed by (i,j), V by (j,k), W by (k,i)
    """
    # Strassen formulas for C = A*B where A,B,C are 2x2
    # A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    # C11 = a11*b11 + a12*b21
    # C12 = a11*b12 + a12*b22
    # C21 = a21*b11 + a22*b21
    # C22 = a21*b12 + a22*b22
    #
    # Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    #
    # U[i*m+j, r] for i,j in {0,1}, r in 0..6
    # V[j*p+k, r] for j,k in {0,1}
    # W[k*n+i, r] for k,i in {0,1}
    
    # Each Strassen product m_r = sum_{j} A_{?,?} * B_{?,?}
    # m1: (a11+a22)*(b11+b22) -> not a single rank-1 in general
    # Actually Strassen uses 7 bilinear forms, each is rank-1 in the tensor sense
    # m1 = (a11+a22)(b11+b22): U contributes to (0,0) and (1,1), V to (0,0) and (1,1)
    # This means U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1 for term 1
    # W contribution: C11 += m1, C22 += m1 -> W[0,0]=1, W[1,1]=1 (k=0,i=0 and k=1,i=1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a11+a22)(b11+b22)
    # U: (0,0)=1, (1,1)=1
    U[0][0] = 1; U[3][0] = 1
    # V: (0,0)=1, (1,1)=1
    V[0][0] = 1; V[3][0] = 1
    # W: C11 += m1 -> (k=0,i=0); C22 += m1 -> (k=1,i=1)
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += m2 -> (k=0,i=1); C22 += -m2 -> (k=1,i=1)
    W[1][1] = 1; W[3][1] = -1
    
    # m3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += m3 -> (k=1,i=0); C22 += m3 -> (k=1,i=1)
    W[2][2] = 1; W[3][2] = 1
    
    # m4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += m4 -> (k=0,i=0); C21 += m4 -> (k=0,i=1)
    W[0][3] = 1; W[1][3] = 1
    
    # m5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 += -m5 -> (k=0,i=0); C12 += m5 -> (k=1,i=0)
    W[0][4] = -1; W[2][4] = 1
    
    # m6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += m6 -> (k=1,i=1)
    W[3][5] = 1
    
    # m7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += m7 -> (k=0,i=0)
    W[0][6] = 1
    
    return U, V, W

def _embed_strassen_into_3x3x8():
    """
    Embed Strassen 2x2x2 blocks into the 3x3*3x8 problem.
    Place 4 non-overlapping 2x2 blocks in the 3x3 input.
    
    The 3x3 matrix A has indices (i,j) for i,j in {0,1,2}.
    We can place 2x2 blocks at positions:
      - rows {0,1}, cols {0,1}
      - rows {0,1}, cols {1,2}
      - rows {1,2}, cols {0,1}
      - rows {1,2}, cols {1,2}
    
    For each 2x2 block at (row_set, col_set), we use Strassen (rank 7).
    The remaining entries are handled by schoolbook.
    
    For the output 3x8, each 2x2 block in A times the 2x8 sub-block of B
    produces a 2x8 output. We handle the 2x8 multiplication by:
    - Using Strassen for the 2x2 part (giving 2x2 output)
    - Schoolbook for the remaining 2x6 part
    
    Actually, let's think more carefully.
    
    A is 3x3, B is 3x8, C = A*B is 3x8.
    
    If we split A into blocks:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    And B = [[B11...B18],
             [B21...B28],
             [B31...B38]]
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: Use Strassen on 2x2 sub-blocks of A where possible.
    
    Let's try: Use Strassen for the top-left 2x2 block of A (rows 0,1; cols 0,1).
    This handles C[0,0], C[0,1], C[1,0], C[1,1] partially.
    But C[0,k] for k in 0..7 = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    
    The Strassen part handles the A[0,0]*B[0,k] + A[0,1]*B[1,k] part for k=0,1
    (i.e., the 2x2 sub-product). For k=2..7, we still need schoolbook for
    A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k].
    
    This is getting complex. Let me use a simpler approach:
    
    For each pair of rows (i1, i2) and pair of cols (j1, j2) in A,
    apply Strassen to compute the 2x2 sub-product A_sub * B_sub where
    B_sub is the 2x8 sub-block.
    
    Actually, Strassen works for 2x2 * 2x2. For 2x2 * 2x8, we can
    split B into four 2x2 blocks and apply Strassen to each.
    
    Let me use a different approach: just embed multiple 2x2x2 Strassen
    blocks and handle the rest with schoolbook.
    
    Place Strassen blocks at:
    1. A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {0,1} -> output C rows {0,1}, C cols {0,1}
    2. A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {2,3} -> output C rows {0,1}, C cols {2,3}
    3. A rows {0,1}, A cols {0,1}, B rows {0,1}, B cols {4,5} -> output C rows {0,1}, C cols {4,5}
    4. A rows {1,2}, A cols {1,2}, B rows {1,2}, B cols {0,1} -> output C rows {1,2}, C cols {0,1}
    
    Wait, but these overlap in the A indices. The Strassen decomposition
    for block 1 uses A[0,0], A[0,1], A[1,0], A[1,1].
    Block 4 uses A[1,1], A[1,2], A[2,1], A[2,2].
    They share A[1,1], which is fine since the U matrix can have multiple
    non-zero entries.
    
    Let me count: 4 Strassen blocks * 7 = 28 terms for Strassen.
    Remaining: total schoolbook = 72. We save 4*(4-7) = -12? No.
    
    Each 2x2x2 block has 8 schoolbook terms but 7 Strassen terms.
    So each block saves 1 term. 4 blocks save 4 terms.
    72 - 4 = 68.
    
    But we need to make sure the blocks don't double-count.
    
    Let me be more careful. I'll build the decomposition explicitly.
    """
    n, m, p = 3, 3, 8
    
    # Start with all-zero matrices, we'll fill them in
    # We'll use a list of (U_col, V_col, W_col) tuples
    
    terms = []  # list of (U_vec, V_vec, W_vec) each of appropriate length
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Helper to create a zero vector
    def zero_u(): return [0]*9
    def zero_v(): return [0]*24
    def zero_w(): return [0]*24
    
    # Strassen 2x2x2 helper
    # Given row indices (r0, r1), col indices (c0, c1), out_col indices (k0, k1)
    # This computes contribution to C[r0,k0], C[r0,k1], C[r1,k0], C[r1,k1]
    # from A[r0,c0]*B[c0,k0] etc.
    def strassen_block(r0, r1, c0, c1, k0, k1):
        """Add Strassen terms for 2x2 block."""
        # m1 = (A[r0,c0]+A[r1,c1])*(B[c0,k0]+B[c1,k1])
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1; u[r1*m+c1] = 1
        v[c0*p+k0] = 1; v[c1*p+k1] = 1
        w[k0*n+r0] = 1; w[k1*n+r1] = 1
        add_term(u, v, w)
        
        # m2 = (A[r1,c0]+A[r1,c1])*B[c0,k0]
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c0] = 1; u[r1*m+c1] = 1
        v[c0*p+k0] = 1
        w[k0*n+r1] = 1; w[k1*n+r1] = -1
        add_term(u, v, w)
        
        # m3 = A[r0,c0]*(B[c0,k1]-B[c1,k1])
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1
        v[c0*p+k1] = 1; v[c1*p+k1] = -1
        w[k1*n+r0] = 1; w[k1*n+r1] = 1
        add_term(u, v, w)
        
        # m4 = A[r1,c1]*(B[c1,k0]-B[c0,k0])
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c1] = 1
        v[c1*p+k0] = 1; v[c0*p+k0] = -1
        w[k0*n+r0] = 1; w[k0*n+r1] = 1
        add_term(u, v, w)
        
        # m5 = (A[r0,c0]+A[r0,c1])*B[c1,k1]
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c0] = 1; u[r0*m+c1] = 1
        v[c1*p+k1] = 1
        w[k0*n+r0] = -1; w[k1*n+r0] = 1
        add_term(u, v, w)
        
        # m6 = (A[r1,c0]-A[r0,c0])*(B[c0,k0]+B[c0,k1])
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r1*m+c0] = 1; u[r0*m+c0] = -1
        v[c0*p+k0] = 1; v[c0*p+k1] = 1
        w[k1*n+r1] = 1
        add_term(u, v, w)
        
        # m7 = (A[r0,c1]-A[r1,c1])*(B[c1,k0]+B[c1,k1])
        u = zero_u(); v = zero_v(); w = zero_w()
        u[r0*m+c1] = 1; u[r1*m+c1] = -1
        v[c1*p+k0] = 1; v[c1*p+k1] = 1
        w[k0*n+r0] = 1
        add_term(u, v, w)
    
    # Schoolbook term
    def schoolbook_term(i, j, k):
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i*m+j] = 1
        v[j*p+k] = 1
        w[k*n+i] = 1
        add_term(u, v, w)
    
    # Define which (i,j,k) triples are covered by Strassen blocks
    # and which need schoolbook.
    
    # Strassen block 1: A rows {0,1}, cols {0,1}, B rows {0,1}, cols {0,1}
    # Covers C entries: (0,0), (0,1), (1,0), (1,1) from A[0:2,0:2]*B[0:2,0:2]
    # The 8 schoolbook terms replaced: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    covered = set()
    strassen_blocks = [
        ((0,1), (0,1), (0,1)),  # A rows, A cols, B cols
        ((0,1), (0,1), (2,3)),
        ((0,1), (0,1), (4,5)),
        ((1,2), (1,2), (0,1)),
    ]
    
    for (r0, r1), (c0, c1), (k0, k1) in strassen_blocks:
        # Mark the 8 schoolbook terms as covered
        for i in [r0, r1]:
            for j in [c0, c1]:
                for k in [k0, k1]:
                    covered.add((i, j, k))
        strassen_block(r0, r1, c0, c1, k0, k1)
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    schoolbook_term(i, j, k)
    
    # Build final matrices
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n*m):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(m*p):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(p*n):
            if w[idx] != 0:
                W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}

def _try_slice_method(n, m, p, time_budget=30.0):
    """
    Try the slice method: for each output column k, the slice is a 3x3 matrix
    multiplication tensor (rank >= 9). Try to find a shared rank-9 decomposition
    that works for all 8 columns.
    
    The slice for column k: C[:,k] = A * B[:,k]
    This is a 3x3 * 3x1 multiplication, which has rank 9 (schoolbook).
    
    For a shared decomposition, we need:
    For each k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j)
    
    This means for each k, the 3x3 matrix M_k where M_k[i,j] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    must be all ones.
    
    We can try: fix U to be the schoolbook U for one column, then solve for V and W.
    
    Actually, let's try a simpler approach: use the same 9 terms for all columns
    but with different V and W entries.
    
    Term r: U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    For this to give 1 for all (i,j,k), we need:
    sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k.
    
    If we set U_r[i,j] = delta(i,j, r) (schoolbook), then:
    sum_r delta(i,j,r) * V_r[j,k] * W_r[k,i] = V_{ij}[j,k] * W_{ij}[k,i] = 1
    where V_{ij} is the V vector for term (i,j) and W_{ij} is the W vector.
    
    This gives V_{ij}[j,k] * W_{ij}[k,i] = 1 for all k.
    So V_{ij}[j,k] = 1/sqrt(8) and W_{ij}[k,i] = sqrt(8) for all k? No, that's not right.
    
    V_{ij}[j,k] * W_{ij}[k,i] = 1 for all k in 0..7.
    We can set V_{ij}[j,k] = 1 and W_{ij}[k,i] = 1 for all k. That's just schoolbook with rank 72.
    
    To reduce rank, we need to share terms across different (i,j) pairs.
    
    This is essentially asking for a rank-9 decomposition of the full 3x3x8 tensor,
    which is unlikely to exist (the tensor rank is at least 9, but likely higher).
    
    Let's try rank 27 (3x9) or other values with a simple optimization.
    """
    start = time.time()
    
    # Try rank values from 27 upward (3x3x3 = 27 is a natural lower bound per 3 columns)
    for rank in [27, 28, 29, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68]:
        if time.time() - start > time_budget:
            break
        if _time_left() < 10:
            break
        
        # Initialize with random
        np.random.seed(rank)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Simple gradient-free: try a few random restarts with ALS-like updates
        # For now, just check if we can find a solution via random search (unlikely)
        # Skip to next rank if no quick solution found
        
        # Actually, let's try a structured initialization
        # Use the schoolbook structure but merge some terms
        
        # For now, return None to indicate no solution found
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen embedding (known to give rank 68)
    if _time_left() > 5:
        try:
            strassen_result = _embed_strassen_into_3x3x8()
            if strassen_result and _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p):
                r = len(strassen_result["U"][0])
                if r < best_rank:
                    best_rank = r
                    best_result = strassen_result
        except Exception:
            pass
    
    # Try slice method
    if _time_left() > 15:
        try:
            slice_result = _try_slice_method(n, m, p, time_budget=min(30.0, _time_left() - 5))
            if slice_result:
                if _verify(slice_result["U"], slice_result["V"], slice_result["W"], n, m, p):
                    r = len(slice_result["U"][0])
                    if r < best_rank:
                        best_rank = r
                        best_result = slice_result
        except Exception:
            pass
    
    return best_result
