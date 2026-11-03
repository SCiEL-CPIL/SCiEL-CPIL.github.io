import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_decomp():
    """
    Returns the 7-term rank decomposition for 2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u is a vector of length 4 (for A, flattened row-major: a00,a01,a10,a11)
      v is a vector of length 4 (for B, flattened row-major: b00,b01,b10,b11)
      w is a vector of length 4 (for C, flattened row-major: c00,c01,c10,c11)
    
    The decomposition satisfies:
      sum_t u_t[i] * v_t[j] * w_t[k] = delta_{i,j,k} for the multiplication tensor
    where i indexes A, j indexes B, k indexes C.
    
    Strassen's 7 multiplications:
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11)b00
      M3 = a00(b01-b11)
      M4 = a11(b10-b00)
      M5 = (a00+a01)b11
      M6 = (a10-a00)(b00+b01)
      M7 = (a11-a01)(b10+b11)
    
    C00 = M1 + M4 - M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M6 + M7
    """
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    
    terms = []
    
    # M1 = (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    # M1 appears in C00 with +1, C11 with +1
    w1 = [1, 0, 0, 1]
    terms.append((u1, v1, w1))
    
    # M2 = (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    # M2 appears in C10 with +1
    w2 = [0, 0, 1, 0]
    terms.append((u2, v2, w2))
    
    # M3 = a00(b01-b11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    # M3 appears in C01 with +1, C11 with -1
    w3 = [0, 1, 0, -1]
    terms.append((u3, v3, w3))
    
    # M4 = a11(b10-b00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    # M4 appears in C00 with +1, C10 with +1
    w4 = [1, 0, 1, 0]
    terms.append((u4, v4, w4))
    
    # M5 = (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    # M5 appears in C00 with -1, C01 with +1
    w5 = [-1, 1, 0, 0]
    terms.append((u5, v5, w5))
    
    # M6 = (a10-a00)(b00+b01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    # M6 appears in C11 with +1
    w6 = [0, 0, 0, 1]
    terms.append((u6, v6, w6))
    
    # M7 = (a11-a01)(b10+b11)
    u7 = [0, -1, 0, 1]
    v7 = [0, 0, 1, 1]
    # M7 appears in C00 with -1, C11 with +1
    w7 = [-1, 0, 0, 1]
    terms.append((u7, v7, w7))
    
    return terms

def _embed_2x2_to_4x4(terms):
    """
    Embed a 2x2 Strassen decomposition into a 4x4 decomposition.
    A 4x4 matrix multiplication can be viewed as four 2x2 block multiplications.
    If we partition A (4x4) into four 2x2 blocks A00, A01, A10, A11,
    and B (4x4) into four 2x2 blocks B00, B01, B10, B11,
    then C = A*B has blocks C00=A00*B00+A01*B10, etc.
    
    Actually, this doesn't directly give a rank-49 decomposition because
    the blocks share the same B columns. Let me think differently.
    
    For 4x4x4, we can use a block approach:
    Split A into 2x2 blocks of size 2x2: A = [[A00, A01], [A10, A11]]
    Split B into 2x2 blocks of size 2x2: B = [[B00, B01], [B10, B11]]
    
    Then C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each Cij is a sum of two 2x2 products. If we use Strassen for each product,
    that's 8 products * 7 = 56 terms. But we can share some.
    
    Actually, a better approach: use the fact that we can do 4x4x4 with
    rank 23 (known result). But let me try a simpler block approach.
    
    Alternative: Use 2x2 Strassen in a divide-and-conquer way.
    4x4 = 2x2 blocks of 2x2.
    
    Let me use a different approach: directly construct a rank-49 decomposition
    by using 7 Strassen terms for each of 7 block positions... no, that's not right either.
    
    Let me just use the simple approach: for 4x4x4, use schoolbook for now (rank 64)
    and see if we can improve.
    
    Actually, let me try a proper block decomposition.
    """
    # For 4x4x4, I'll use a 2x2 block structure with Strassen on each block.
    # This gives rank 7*4 = 28 for the "diagonal" blocks if we could share,
    # but actually it's more complex.
    
    # Let me just return None and use a different approach in construct().
    return None

def _strassen_4x4_decomp():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication
    using a 2x2 block decomposition with Strassen's algorithm.
    
    We view 4x4 matrices as 2x2 blocks of 2x2 submatrices.
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]] where each Bji is 2x2
    
    C = A*B where:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    For each 2x2 product, we use Strassen (7 terms).
    Total: 8 products * 7 terms = 56 terms.
    
    But we can be smarter. Let me use the standard result that
    4x4x4 has rank at most 49 using a specific construction.
    
    Actually, the simplest correct approach: just use 8 independent
    2x2 Strassen decompositions, giving rank 56.
    """
    n, m, p = 4, 4, 4
    # A is 4x4, flattened as a[0..15] where a[i*4+j] = A[i,j]
    # B is 4x4, flattened as b[0..15] where b[j*4+k] = B[j,k]
    # C is 4x4, flattened as c[0..15] where c[i*4+k] = C[i,k]
    
    # Block indices:
    # A block (bi, bj) where bi,bj in {0,1}: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # B block (bj, bk) where bj,bk in {0,1}: rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
    # C block (bi, bk) where bi,bk in {0,1}: rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    
    terms = _strassen_2x2_decomp()
    
    U = []
    V = []
    W = []
    
    for bi in range(2):
        for bk in range(2):
            # C block (bi, bk) = sum over bj of A block (bi, bj) * B block (bj, bk)
            for bj in range(2):
                for (u2, v2, w2) in terms:
                    u4 = [0] * 16
                    v4 = [0] * 16
                    w4 = [0] * 16
                    
                    # A block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
                    for a_idx in range(4):
                        ai, aj = a_idx // 2, a_idx % 2  # local 2x2 index
                        global_i = 2 * bi + ai
                        global_j = 2 * bj + aj
                        u4[global_i * 4 + global_j] = u2[a_idx]
                    
                    # B block (bj, bk): rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
                    for b_idx in range(4):
                        bi_local, bk_local = b_idx // 2, b_idx % 2
                        global_j = 2 * bj + bi_local
                        global_k = 2 * bk + bk_local
                        v4[global_j * 4 + global_k] = v2[b_idx]
                    
                    # C block (bi, bk): rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
                    for c_idx in range(4):
                        ci, ck = c_idx // 2, c_idx % 2
                        global_i = 2 * bi + ci
                        global_k = 2 * bk + ck
                        w4[global_i * 4 + global_k] = w2[c_idx]
                    
                    U.append(u4)
                    V.append(v4)
                    W.append(w4)
    
    return U, V, W

def _schoolbook_447():
    """
    Schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    Rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
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

def _hybrid_447_decomp():
    """
    Hybrid decomposition for 4x4 * 4x7:
    - Use Strassen-based decomposition for the first 4 columns of B (4x4x4 part)
    - Use schoolbook for the last 3 columns of B (4x4x3 part)
    
    Total rank = 56 + 48 = 104 (using the block Strassen approach)
    """
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4 * 4x4 (first 4 columns)
    U1, V1, W1 = _strassen_4x4_decomp()
    rank1 = len(U1)
    
    # Part 2: 4x4 * 4x3 (last 3 columns), schoolbook
    n2, m2, p2 = 4, 4, 3
    rank2 = n2 * m2 * p2
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    # Combine: U has shape (16, rank1+rank2)
    # For part 1: U1[i][c] for c in 0..rank1-1
    # For part 2: U2[i][c] for c in 0..rank2-1
    # 
    # V has shape (28, rank1+rank2)
    # For part 1: V1[j*4+k][c] for k in 0..3 (first 4 cols of B)
    # For part 2: V2[j*3+k][c] for k in 0..2, mapped to global col k+4
    #
    # W has shape (28, rank1+rank2)
    # For part 1: W1[k*4+i][c] for k in 0..3
    # For part 2: W2[k*4+i][c] for k in 0..2, mapped to global row (k+4)*4+i
    
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Fill part 1
    for c in range(rank1):
        for i in range(16):
            U[i][c] = U1[i][c]
        for idx in range(16):
            j, k = idx // 4, idx % 4
            V[idx][c] = V1[idx][c]
        for idx in range(16):
            k, i = idx // 4, idx % 4
            W[idx][c] = W1[idx][c]
    
    # Fill part 2 (columns 4,5,6 of B)
    for c in range(rank2):
        for i in range(16):
            U[i][rank1 + c] = U2[i][c]
        # V2 has shape (12, rank2), indexing: row = j*3 + k_local
        # Global B index: row = j*7 + (k_local + 4)
        for j in range(4):
            for k_local in range(3):
                v2_idx = j * 3 + k_local
                v_global_idx = j * 7 + (k_local + 4)
                V[v_global_idx][rank1 + c] = V2[v2_idx][c]
        # W2 has shape (12, rank2), indexing: row = k_local*4 + i
        # Global W index: row = (k_local+4)*4 + i
        for k_local in range(3):
            for i in range(4):
                w2_idx = k_local * 4 + i
                w_global_idx = (k_local + 4) * 4 + i
                W[w_global_idx][rank1 + c] = W2[w2_idx][c]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook (rank 112)
    schoolbook = _schoolbook_447()
    
    # Try hybrid approach
    hybrid = _hybrid_447_decomp()
    
    # Verify hybrid
    if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
        result = hybrid
    else:
        result = schoolbook
    
    # Verify final result
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        result = schoolbook
    
    return result
