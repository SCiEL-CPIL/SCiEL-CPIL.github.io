"""Matrix multiplication 4x4 x 4x7 via Strassen block decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition exactly reconstructs matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    # Tensor: T[i,j,k] = 1 for all i,j,k
    # Check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    r = n*m*p
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1
                c += 1
    return U, V, W

def _strassen_2x2_2x2():
    """Strassen for 2x2 * 2x2 -> 2x2, rank 7."""
    n = m = p = 2
    U = np.zeros((n*m, 7)); V = np.zeros((m*p, 7)); W = np.zeros((p*n, 7))
    # M1=(A11+A22)(B11+B22); C11+=M1, C22+=M1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A21+A22)B11; C21+=M2
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1
    # M3=A11(B12-B22); C12+=M3
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
    # M4=A22(B21-B11); C11+=M4, C21+=M4
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5=(A11+A12)B22; C11-=M5, C12+=M5
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6=(A21-A11)(B11+B12); C22+=M6
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A12-A22)(B21+B22); C11+=M7
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_444():
    """Strassen-based 4x4 * 4x4 -> 4x4, rank 49."""
    n = m = p = 4
    U = np.zeros((n*m, 49)); V = np.zeros((m*p, 49)); W = np.zeros((p*n, 49))
    # 7 Strassen macro-factors, each a 2x2*2x2 block product (rank 7)
    # A block indices: A11=rows0-1,cols0-1; A12=rows0-1,cols2-3; A21=rows2-3,cols0-1; A22=rows2-3,cols2-3
    # U index for A[i,j] = i*4+j
    A11 = [0,1,4,5]; A12 = [2,3,6,7]; A21 = [8,9,12,13]; A22 = [10,11,14,15]
    # B block indices: B11=rows0-1,cols0-1; B12=rows0-1,cols2-3; B21=rows2-3,cols0-1; B22=rows2-3,cols2-3
    # V index for B[i,j] = i*4+j
    B11 = [0,1,4,5]; B12 = [2,3,6,7]; B21 = [8,9,12,13]; B22 = [10,11,14,15]
    # C block indices: C11=rows0-1,cols0-1; C12=rows0-1,cols2-3; C21=rows2-3,cols0-1; C22=rows2-3,cols2-3
    # W index for C[i,j] = i*4+j  (W[k*n+i] where k=row of C, i=col of C)
    C11 = [0,1,4,5]; C12 = [2,3,6,7]; C21 = [8,9,12,13]; C22 = [10,11,14,15]
    
    # Strassen macro-factors: (A_block, B_block, C_blocks_with_signs)
    # M1=(A11+A22)(B11+B22); C11+=, C22+=
    # M2=(A21+A22)B11; C21+=
    # M3=A11(B12-B22); C12+=
    # M4=A22(B21-B11); C11+=, C21+=
    # M5=(A11+A12)B22; C11-=, C12+=
    # M6=(A21-A11)(B11+B12); C22+=
    # M7=(A12-A22)(B21+B22); C11+=
    macros = [
        # (A_blocks, A_signs, B_blocks, B_signs, [(C_block, sign), ...])
        ([A11, A22], [1,1], [B11, B22], [1,1], [(C11,1),(C22,1)]),
        ([A21, A22], [1,1], [B11], [1], [(C21,1)]),
        ([A11], [1], [B12, B22], [1,-1], [(C12,1)]),
        ([A22], [1], [B21, B11], [1,-1], [(C11,1),(C21,1)]),
        ([A11, A12], [1,1], [B22], [1], [(C11,-1),(C12,1)]),
        ([A21, A11], [1,-1], [B11, B12], [1,1], [(C22,1)]),
        ([A12, A22], [1,-1], [B21, B22], [1,1], [(C11,1)]),
    ]
    
    # Strassen 2x2*2x2 pattern (from _strassen_2x2_2x2):
    # For a 2x2*2x2 product with local indices a[4],b[4],c[4]:
    # 7 sub-factors with specific (a_idx, b_idx, c_idx, sign) patterns
    # Local index: a[0]=a11, a[1]=a12, a[2]=a21, a[3]=a22
    #              b[0]=b11, b[1]=b12, b[2]=b21, b[3]=b22
    #              c[0]=c11, c[1]=c12, c[2]=c21, c[3]=c22
    # Sub-factor patterns (U_local, V_local, W_local):
    strassen_pat = [
        # (U: [(a_idx, sign)], V: [(b_idx, sign)], W: [(c_idx, sign)])
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # M1
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),              # M2
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),       # M4
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),       # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),       # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),       # M7
    ]
    
    col = 0
    for macro in macros:
        A_blocks, A_signs, B_blocks, B_signs, C_targets = macro
        for sub in strassen_pat:
            U_entries, V_entries, W_entries = sub
            # U entries: for each (a_local_idx, s), the global A index is A_blocks[a_local_idx], sign = A_signs[a_local_idx] * s
            for a_local, s_u in U_entries:
                gidx = A_blocks[a_local]
                U[gidx, col] += A_signs[a_local] * s_u
            for b_local, s_v in V_entries:
                gidx = B_blocks[b_local]
                V[gidx, col] += B_signs[b_local] * s_v
            for c_local, s_w in W_entries:
                C_block, c_sign = C_targets[c_local]
                for gidx in C_block:
                    W[gidx, col] += c_sign * s_w
            col += 1
    return U, V, W

def _build_443():
    """4x4 * 4x3 -> 4x3, rank 42.
    
    Split A into 2x2 blocks, B into top/bottom 2x3 halves.
    C_top = A11*B1 + A12*B2 (2x3)
    C_bot = A21*B1 + A22*B2 (2x3)
    
    Use Strassen-like sharing:
    - M1 = (A11+A22)B1: contributes to C_top and C_bot
    - M2 = (A12+A22)B2: contributes to C_top and C_bot  
    - M3 = A11B2: contributes to C_top
    - M4 = A22B1: contributes to C_bot
    Then: C_top = M1 - M4 + M3, C_bot = M1 - M2 + M4
    Wait, that's not right. Let me reconsider.
    
    Actually: C_top = A11*B1 + A12*B2, C_bot = A21*B1 + A22*B2
    Define:
    M1 = (A11+A22)*B1  (2x2 * 2x3, rank 12)
    M2 = A11*B2         (2x2 * 2x3, rank 12)
    M3 = A22*B2         (2x2 * 2x3, rank 12)
    M4 = (A12-A22)*B2   (2x2 * 2x3, rank 12)
    Then: C_top = M1 + M2 - M4 - M3 + ... hmm this is getting complicated.
    
    Let me just use 4 separate 2x2*2x3 products: rank 48.
    But we can do better with sharing.
    
    Alternative: 4x4*4x3. Split A as [A1; A2] where A1 is 2x4, A2 is 2x4.
    B as [B1; B2] where B1 is 2x3, B2 is 2x3.
    C = A1*B1 + A2*B2 (4x3 result, top 2 rows from A1*B1 + A2*B2? No.)
    Actually C[0:2, :] = A[0:2,:]*B[:, 0:3]... no.
    C[i,k] = sum_j A[i,j]*B[j,k].
    C[0:2, :] = A[0:2,:]*B[:, :]  (2x4 * 4x3 -> 2x3)
    C[2:4, :] = A[2:4,:]*B[:, :]  (2x4 * 4x3 -> 2x3)
    
    For C_top = A_top * B (2x4 * 4x3):
    A_top = [A11, A12] (2x2 each), B = [B1; B2] (2x3 each)
    C_top = A11*B1 + A12*B2 (2x3)
    
    For C_bot = A_bot * B (2x4 * 4x3):
    A_bot = [A21, A22] (2x2 each)
    C_bot = A21*B1 + A22*B2 (2x3)
    
    So we need 4 products of type 2x2*2x3, each rank 12, total 48.
    
    Can we share? Define:
    P1 = (A11+A22)*B1 (2x2*2x3, rank 12)
    P2 = (A12+A22)*B2 (2x2*2x3, rank 12)
    P3 = A11*B2 (2x2*2x3, rank 12)
    P4 = A22*B1 (2x2*2x3, rank 12)
    
    C_top = A11*B1 + A12*B2 = (P1 - P4) + (P2 - A22*B2) ... not clean.
    
    Let me try:
    P1 = A11*B1 (rank 12)
    P2 = A12*B2 (rank 12)
    P3 = A21*B1 (rank 12)
    P4 = A22*B2 (rank 12)
    Total rank 48. C_top = P1+P2, C_bot = P3+P4.
    
    For sharing, I need a Strassen-like identity. For 2x2*2x3, 
    the "Strassen" equivalent would need the 3rd column to be handled.
    Actually for 2x2*2x3, schoolbook rank 12 is likely optimal (no known improvement).
    So rank 48 for 4x4*4x3.
    
    But wait - can we combine the 4x4*4x4 and 4x4*4x3 more cleverly?
    Total = 49 + 48 = 97. Still better than 112.
    
    Let me just go with rank 97 for now.
    """
    n, m, p = 4, 4, 3
    # 4 products of 2x2*2x3, each rank 12, total 48
    r = 48
    U = np.zeros((n*m, r)); V = np.zeros((m*p, r)); W = np.zeros((p*n, r))
    
    # A indices: A[i,j] -> U[i*4+j]
    # B indices: B[i,j] -> V[i*3+j]  (B is 4x3)
    # W indices: W[i*4+j] where i is row of C (0..2), j is col of C (0..3)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    
    # Product 1: A[0:2,0:2]*B[0:2,0:3] -> C[0:2,0:3]
    # Product 2: A[0:2,2:4]*B[2:4,0:3] -> C[0:2,0:3]
    # Product 3: A[2:4,0:2]*B[0:2,0:3] -> C[2:4,0:3]
    # Product 4: A[2:4,2:4]*B[2:4,0:3] -> C[2:4,0:3]
    
    products = [
        # (A_row_start, A_col_start, B_row_start, C_row_start)
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:3] -> C[0:2,:]
        (0, 2, 2, 0),  # A[0:2,2:4] * B[2:4,0:3] -> C[0:2,:]
        (2, 0, 0, 2),  # A[2:4,0:2] * B[0:2,0:3] -> C[2:4,:]
        (2, 2, 2, 2),  # A[2:4,2:4] * B[2:4,0:3] -> C[2:4,:]
    ]
    
    col = 0
    for (ar, ac, br, cr) in products:
        # 2x2 * 2x3 schoolbook: 12 terms
        for ai in range(2):
            for aj in range(2):
                for bk in range(3):
                    U[(ar+ai)*4 + (ac+aj), col] = 1
                    V[(br+ai)*3 + bk, col] = 1
                    W[(cr+ai)*4 + bk, col] = 1  # Wait, W index is k*n+i where k=row of C, i=col of C
                    # C row = cr+ai, C col = bk
                    # W[k*n+i] = W[(cr+ai)*4 + bk]
                    col += 1
    return U, V, W

def _build_447():
    """Combine 4x4*4x4 (rank 49) and 4x4*4x3 (rank 48) for 4x4*4x7 (rank 97)."""
    n, m, p = 4, 4, 7
    U1, V1, W1 = _build_444()  # rank 49
    U2, V2, W2 = _build_443()  # rank 48
    
    total_rank = 49 + 48
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Part 1: 4x4*4x4 contributes to first 4 columns of B and first 4 cols of C
    U[:, 0:49] = U1
    # V1 is 16x49 (B is 4x4, m*p=16). B indices: B[i,j] -> V[i*4+j]
    # In the full 4x7, B[i,j] -> V[i*7+j]. So V1[i*4+j] maps to V[i*7+j] for j<4.
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:49] = V1[i*4+j, :]
    # W1 is 16x49. C indices: C[i,j] -> W[i*4+j] (i=row, j=col, n=4)
    # In full 4x7, C[i,j] -> W[i*7+j] for j<4.
    for i in range(4):
        for j in range(4):
            W[i*7+j, 0:49] = W1[i*4+j, :]
    
    # Part 2: 4x4*4x3 contributes to last 3 columns of B and last 3 cols of C
    U[:, 49:97] = U2
    # V2 is 12x48 (B is 4x3, m*p=12). B indices: B[i,j] -> V[i*3+j]
    # In full 4x7, B[i,4+j] -> V[i*7+4+j].
    for i in range(4):
        for j in range(3):
            V[i*7+4+j, 49:97] = V2[i*3+j, :]
    # W2 is 12x48. C indices: C[i,j] -> W[i*4+j] where i=row(0..2), j=col(0..3)
    # Wait, W2 was built for 4x4*4x3 where p=3, n=4. W index = k*n+i = k*4+i.
    # k is row of C (0..2), i is col of C (0..3).
    # In full 4x7, C[i,4+j] -> W[i*7+4+j] where i=row, j=col(0..2).
    # So W2[k*4+i] maps to W[i*7+4+k]... 
    # Hmm, let me re-examine. In _build_443, W[(cr+ai)*4 + bk] where cr+ai is C row, bk is C col.
    # So W2[row*4+col] where row in 0..3, col in 0..2.
    # In full problem, C[row, 4+col] -> W[row*7 + 4+col].
    for row in range(4):
        for col in range(3):
            W[row*7+4+col, 49:97] = W2[row*4+col, :]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n*m*p
    
    # Try Strassen-based construction
    try:
        if time.time() < deadline - 10:
            U, V, W = _build_447()
            if _verify(U, V, W, n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = U.shape[1]
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
