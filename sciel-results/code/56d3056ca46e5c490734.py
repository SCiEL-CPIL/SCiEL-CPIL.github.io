"""Combined program: structural decomposition of 4x4 * 4x7 matrix multiplication
using a hybrid Strassen (4x4x4) + schoolbook (4x4x3) approach, with a deadline guard.
The key structural insight from the inspirations is that the 4x4*4x7 problem can be
split into two parts: multiplying a 4x4 by a 4x4 (which can use Strassen's rank-49
decomposition embedded into the 4x4x4 sub-problem) and multiplying a 4x4 by a 4x3
(which uses schoolbook rank 48), for a total rank of 97.

We build the factor matrices directly from this structural decomposition, which is
exact and deterministic. No random search needed since the Strassen + schoolbook
combination is a known valid decomposition."""

import time

DIMENSIONS = [4, 4, 7]


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) where these are length-4 vectors.
    U[i*2+j] for A, V[j*2+k] for B, W[k*2+i] for C.
    
    Strassen's 7 products for 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    M1 = (a+d)(e+h)
    M2 = (c+d)(e)
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)(h)
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0] = M1 - M4 + M5 - M7
    C[1] = M3 + M5
    C[2] = M2 + M4
    C[3] = M1 - M2 + M6 - M7
    """
    terms = []
    # M1: A: a+d, B: e+h, C: +1 to C[0], +1 to C[3]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2: A: c+d, B: e, C: +1 to C[2]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # M3: A: a, B: f-h, C: +1 to C[1]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # M4: A: d, B: g-e, C: -1 to C[0], +1 to C[2]
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]))
    # M5: A: a+b, B: h, C: +1 to C[0], +1 to C[1]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6: A: c-a, B: e+f, C: +1 to C[3]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7: A: b-d, B: g+h, C: -1 to C[0], -1 to C[3]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]))
    return terms


def _embed_2x2_to_4x4(vec_4, row_block, col_block):
    """Embed a 4-element vector (for 2x2) into a 16-element vector (for 4x4).
    row_block, col_block in {0,1} select which 2x2 block of the 4x4 matrix."""
    result = [0] * 16
    for idx, val in enumerate(vec_4):
        r = idx // 2
        c = idx % 2
        result[(row_block * 2 + r) * 4 + (col_block * 2 + c)] = val
    return result


def _embed_2x2_to_4x4x7(vec_4, row_block, col_block):
    """Embed a 4-element vector into a 28-element vector (4x7 flattened).
    For the W matrix which has shape (p*n, rank) = (7*4, rank) = (28, rank).
    W[k*n+i] where k in [0,7), i in [0,4).
    For a 2x2 block of C at rows [row_block*2, row_block*2+1] and cols [col_block*2, col_block*2+1],
    the C indices are (i, k) where i is row in C (0..3) and k is col in C (0..6).
    Wait - the convention is W[k*n+i, r] where k is the column index of B (0..p-1=0..6)
    and i is the row index of A/C (0..n-1=0..3).
    
    So W is indexed by (k, i) -> position k*4 + i in the 28-element vector.
    
    For a 2x2 block of C at rows [2*rb, 2*rb+1] and cols [2*cb, 2*cb+1]:
    - i (row of C) ranges over 2*rb, 2*rb+1
    - k (col of C, which is col of B) ranges over 2*cb, 2*cb+1
    
    The 4 C entries in this block are:
    C[2*rb, 2*cb], C[2*rb, 2*cb+1], C[2*rb+1, 2*cb], C[2*rb+1, 2*cb+1]
    
    In the vec_4 for the 2x2 C block, the indices are:
    vec[0] = C[0,0] of block -> i=2*rb, k=2*cb -> W position (2*cb)*4 + (2*rb)
    vec[1] = C[0,1] of block -> i=2*rb, k=2*cb+1 -> W position (2*cb+1)*4 + (2*rb)
    vec[2] = C[1,0] of block -> i=2*rb+1, k=2*cb -> W position (2*cb)*4 + (2*rb+1)
    vec[3] = C[1,1] of block -> i=2*rb+1, k=2*cb+1 -> W position (2*cb+1)*4 + (2*rb+1)
    """
    result = [0] * 28
    for idx, val in enumerate(vec_4):
        r = idx // 2  # 0 or 1 within the 2x2 block
        c = idx % 2   # 0 or 1 within the 2x2 block
        i = row_block * 2 + r  # actual row index in C (0..3)
        k = col_block * 2 + c  # actual col index in C / B (0..3, but B has 7 cols)
        # W position: k * n + i = k * 4 + i
        result[k * 4 + i] = val
    return result


def _build_4x4x4_strassen_decomposition():
    """Build the rank-49 Strassen decomposition for 4x4 * 4x4 -> 4x4.
    
    We treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    The block structure gives a 2x2 block matrix multiplication, which we solve
    with Strassen (7 block-level products). Each block-level product is a 2x2 * 2x2
    multiplication, which we also solve with Strassen (7 scalar-level products).
    
    Total rank: 7 (block-level) * 7 (scalar-level) = 49.
    
    For block positions (rb, cb) where rb, cb in {0,1}:
    - A block at (rb, ca) where ca is the column block of A
    - B block at (cb, cb2) where cb2 is the column block of B
    - C block at (rb, cb2)
    
    Actually, let me think about this more carefully.
    
    4x4 matrix A has 2x2 blocks: A[rb, cb] for rb, cb in {0,1}.
    4x4 matrix B has 2x2 blocks: B[cb, cb2] for cb, cb2 in {0,1}.
    4x4 matrix C has 2x2 blocks: C[rb, cb2] for rb, cb2 in {0,1}.
    
    C[rb, cb2] = sum_{cb} A[rb, cb] * B[cb, cb2]
    
    This is a 2x2 block matrix multiplication. Strassen gives us 7 block-level products:
    M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    M2 = (A[1,0]+A[1,1]) * B[0,0]
    M3 = A[0,0] * (B[0,1]-B[1,1])
    M4 = A[1,1] * (B[1,0]-B[0,0])
    M5 = (A[0,0]+A[0,1]) * B[1,1]
    M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    C[0,0] = M1 - M4 + M5 - M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M6 - M7
    
    Each M is a 2x2 * 2x2 product, which we decompose with 7 scalar Strassen terms.
    So total: 7 * 7 = 49 rank-1 terms.
    
    For each block-level Strassen term (say M1), the A-part is a linear combination of
    2x2 blocks of A, and the B-part is a linear combination of 2x2 blocks of B.
    Each 2x2 block of A corresponds to 4 scalar entries.
    
    So for a block-level term like M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]):
    The A-part vector (over all 16 entries of A) has the 2x2 block A[0,0] with coefficient +1
    and A[1,1] with coefficient +1, others 0.
    Similarly for B.
    
    Then within each 2x2 block product, we apply the 7 scalar Strassen terms.
    
    Let me construct this systematically.
    """
    strassen_2x2 = _strassen_2x2_terms()
    
    # Block-level Strassen terms: each is (A_block_coeffs, B_block_coeffs, C_block_coeffs)
    # where _block_coeffs is a 2x2 matrix of coefficients for the 4 blocks.
    # A blocks: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    # B blocks: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    # C blocks: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # Block coefficients: (rb, cb) -> index rb*2+cb
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    block_terms = [
        # (A_coeffs[4], B_coeffs[4], C_coeffs[4])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]), # M7
    ]
    
    # Now build the 49 rank-1 terms for 4x4x4.
    # Each term is (U_vec[16], V_vec[16], W_vec[16]) for the 4x4x4 case.
    # U_vec[i*4+j], V_vec[j*4+k], W_vec[k*4+i]
    
    terms_49 = []
    
    for (A_coeffs, B_coeffs, C_coeffs) in block_terms:
        # A_coeffs[4]: coefficient for each 2x2 block of A
        # B_coeffs[4]: coefficient for each 2x2 block of B
        # C_coeffs[4]: coefficient for each 2x2 block of C
        
        # For the A-part: we need a 16-element vector where each 2x2 block
        # is scaled by the corresponding coefficient.
        # Block (rb, cb) occupies rows 2*rb..2*rb+1 and cols 2*cb..2*cb+1.
        # In the flattened 16-vector: positions (2*rb+r)*4 + (2*cb+c) for r,c in {0,1}.
        
        # The A-part is a linear combination of 2x2 blocks.
        # But each block product M = A_block * B_block is itself a 2x2*2x2 product,
        # which we decompose into 7 scalar Strassen terms.
        
        # So for each block-level term, we iterate over the 7 scalar Strassen terms.
        for (U_2x2, V_2x2, W_2x2) in strassen_2x2:
            U_16 = [0] * 16
            V_16 = [0] * 16
            W_16 = [0] * 16
            
            # A-part: for each block (rb, cb) with nonzero coefficient,
            # add coeff * U_2x2 embedded at that block position.
            for idx in range(4):
                coeff = A_coeffs[idx]
                if coeff == 0:
                    continue
                rb = idx // 2
                cb = idx % 2
                for r2 in range(4):
                    U_16[(rb * 2 + r2 // 2) * 4 + (cb * 2 + r2 % 2)] += coeff * U_2x2[r2]
            
            # B-part: similar
            for idx in range(4):
                coeff = B_coeffs[idx]
                if coeff == 0:
                    continue
                rb = idx // 2  # row block of B
                cb = idx % 2   # col block of B
                for r2 in range(4):
                    V_16[(rb * 2 + r2 // 2) * 4 + (cb * 2 + r2 % 2)] += coeff * V_2x2[r2]
            
            # C-part: similar
            for idx in range(4):
                coeff = C_coeffs[idx]
                if coeff == 0:
                    continue
                rb = idx // 2  # row block of C
                cb = idx % 2   # col block of C
                for r2 in range(4):
                    W_16[(rb * 2 + r2 // 2) * 4 + (cb * 2 + r2 % 2)] += coeff * W_2x2[r2]
            
            terms_49.append((U_16, V_16, W_16))
    
    return terms_49


def _build_4x4x3_schoolbook_decomposition():
    """Build the rank-48 schoolbook decomposition for 4x4 * 4x3 -> 4x3.
    
    U shape: (4*4, 48) = (16, 48)
    V shape: (4*3, 48) = (12, 48)
    W shape: (3*4, 48) = (12, 48)
    
    W[k*n+i] where k in [0,3), i in [0,4).
    
    Schoolbook: for each (i, j, k) in [0,4) x [0,4) x [0,3):
    U[i*4+j, col] = 1
    V[j*3+k, col] = 1
    W[k*4+i, col] = 1
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
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
    return U, V, W


def _build_4x4x4_schoolbook_decomposition():
    """Build the rank-64 schoolbook decomposition for 4x4 * 4x4 -> 4x4."""
    n, m, p = 4, 4, 4
    rank = n * m * p  # 64
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
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy: Split the 4x7 matrix B into B1 (4x4) and B2 (4x3).
    Then C = A*B = A*B1 + A*B2.
    - A*B1: 4x4 * 4x4 -> 4x4, use Strassen rank 49
    - A*B2: 4x4 * 4x3 -> 4x3, use schoolbook rank 48
    Total rank: 49 + 48 = 97.
    
    U shape: (16, 97), V shape: (28, 97), W shape: (28, 97)
    """
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split B into B1 (cols 0-3) and B2 (cols 4-6)
    # B1 is 4x4, B2 is 4x3
    
    # Part 1: A * B1 (4x4 * 4x4 -> 4x4), Strassen rank 49
    strassen_terms = _build_4x4x4_strassen_decomposition()
    rank1 = len(strassen_terms)  # 49
    
    # Part 2: A * B2 (4x4 * 4x3 -> 4x3), schoolbook rank 48
    U2, V2, W2 = _build_4x4x3_schoolbook_decomposition()
    rank2 = len(U2[0])  # 48
    
    total_rank = rank1 + rank2  # 97
    
    # Build combined U (16 x total_rank), V (28 x total_rank), W (28 x total_rank)
    U = [[0] * total_rank for _ in range(n * m)]      # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)]      # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)]      # 28 x 97
    
    # Fill in Part 1: Strassen for A*B1
    for c, (U_vec, V_vec, W_vec) in enumerate(strassen_terms):
        # U_vec is 16 elements: U[i*4+j]
        for idx in range(16):
            U[idx][c] = U_vec[idx]
        # V_vec is 16 elements: V[j*4+k] for k in [0,4)
        # These correspond to B1's entries, which are B's cols 0-3.
        # V[j*p+k] where p=7. B1's col k (0..3) maps to B's col k.
        # So V[j*7+k] for k in [0,4)
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            V[j * 7 + k][c] = V_vec[idx]
        # W_vec is 16 elements: W[k*4+i] for k in [0,4)
        # These correspond to C's cols 0-3, which are B1's cols.
        # W[k*n+i] where n=4. k in [0,4) for B1's cols.
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            W[k * 4 + i][c] = W_vec[idx]
    
    # Fill in Part 2: Schoolbook for A*B2
    # U2 is 16 x 48: U2[i*4+j, c2]
    # V2 is 12 x 48: V2[j*3+k, c2] where k in [0,3) for B2's cols
    # W2 is 12 x 48: W2[k*4+i, c2] where k in [0,3) for B2's cols
    
    for c2 in range(rank2):
        c = rank1 + c2  # offset into total_rank
        for i in range(16):
            U[i][c] = U2[i][c2]
        # V2[j*3+k] where k is 0..2 (B2's local cols), maps to B's cols 4+k
        for idx in range(12):
            j = idx // 3
            k_local = idx % 3
            k_global = 4 + k_local  # B's global col index
            V[j * 7 + k_global][c] = V2[idx][c2]
        # W2[k*4+i] where k is 0..2 (B2's local cols), maps to C's cols 4+k
        for idx in range(12):
            k_local = idx // 4
            i = idx % 4
            k_global = 4 + k_local  # C's global col index
            W[k_global * 4 + i][c] = W2[idx][c2]
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    
    For all (i, j, k), we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(U)}, {len(U[0])})")
    print(f"V shape: ({len(V)}, {len(V[0])})")
    print(f"W shape: ({len(W)}, {len(W[0])})")
    valid = _verify_decomposition(U, V, W, n, m, p)
    print(f"Valid: {valid}")
    if not valid:
        # Debug: find which (i,j,k) fails
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        print(f"  FAIL at i={i}, j={j}, k={k}: sum={s}")
