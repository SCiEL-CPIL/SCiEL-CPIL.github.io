"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np
from numpy.linalg import norm, matrix_rank

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS

# Build the target tensor T of shape (n*m, m*p, p*n) with trace convention W[k*n+i, r]
def _build_target_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_idx = i * m + j
                V_idx = j * p + k
                W_idx = k * n + i
                T[U_idx, V_idx, W_idx] = 1
    return T

# Verify exact decomposition: sum over rank of U[r,a]*V[r,a]*W[r,a] == T[i,j,k]
def _verify(U, V, W, T):
    r = U.shape[1]
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=object)
    for a in range(r):
        for i in range(n * m):
            if U[i, a] == 0:
                continue
            for j in range(m * p):
                if V[j, a] == 0:
                    continue
                for k in range(p * n):
                    if W[k, a] == 0:
                        continue
                    T_decomp[i, j, k] += U[i, a] * V[j, a] * W[k, a]
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                if T_decomp[i, j, k] != T[i, j, k]:
                    return False
    return True

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 terms for 2x2 * 2x2 matrix multiplication.
    Each term is (u_pattern, v_pattern, w_pattern) where each pattern is a dict
    mapping (row, col) indices to coefficients.
    
    For A (2x2) * B (2x2) = C (2x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For each of the 7 multiplications, define:
    # u: which elements of A are multiplied (with coefficients)
    # v: which elements of B are multiplied (with coefficients)
    # w: which elements of C receive the result (with coefficients)
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00(+1), C11(+1)
    terms.append((
        {(0,0): 1, (1,1): 1},   # u: A elements
        {(0,0): 1, (0,1): 1},   # v: B elements (B[j,k], j=0,k=0 and j=0,k=1? No...)
        {(0,0): 1, (1,1): 1}    # w: C elements
    ))
    # Wait, I need to be more careful. Let me re-derive.
    # A is 2x2: A[i,j] for i,j in {0,1}
    # B is 2x2: B[j,k] for j,k in {0,1}
    # C is 2x2: C[i,k] for i,k in {0,1}
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[0,1])
    # This doesn't look right. Let me use the standard formulation.
    
    # Standard Strassen for C = A*B where all are 2x2:
    # M1 = (a+d)(e+h)       where a=A00,b=A01,c=A10,d=A11, e=B00,f=B01,g=B10,h=B11
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M, the A-side and B-side are products of sums.
    # M1: A-side = A00 + A11, B-side = B00 + B11
    #     C-side: C00 += 1*M1, C11 += 1*M1
    # M2: A-side = A10 + A11, B-side = B00
    #     C-side: C10 += 1*M2, C11 += -1*M2
    # M3: A-side = A00, B-side = B01 - B11
    #     C-side: C01 += 1*M3, C11 += 1*M3
    # M4: A-side = A11, B-side = B10 - B00
    #     C-side: C00 += 1*M4, C10 += 1*M4
    # M5: A-side = A00 + A01, B-side = B11
    #     C-side: C00 += -1*M5, C01 += 1*M5
    # M6: A-side = A10 - A00, B-side = B00 + B01
    #     C-side: C11 += 1*M6
    # M7: A-side = A01 - A11, B-side = B10 + B11
    #     C-side: C00 += 1*M7
    
    # Now express as rank-1 tensor terms:
    # Each term is a product of (sum of A elements) * (sum of B elements) * (sum of C elements)
    # But actually each M is a scalar, and it contributes to multiple C entries.
    # So the rank-1 decomposition has:
    # For each M, we need one term per C entry it contributes to? No.
    # Actually, each M is computed as one multiplication, and then added to C entries.
    # The tensor decomposition is:
    # T[i,j,k] = sum_a U[i,a] * V[j,a] * W[k,a]
    # where i indexes A, j indexes B, k indexes C.
    
    # For M1 = (A00+A11)(B00+B11):
    # This contributes +1 to C00 and +1 to C11.
    # As a tensor term, we need:
    #   U part: A00 and A11 have coefficient 1
    #   V part: B00 and B11 have coefficient 1  
    #   W part: C00 and C11 have coefficient 1
    # But this is ONE rank-1 term only if the W part is a single element.
    # Since W has two non-zero entries, this is actually TWO rank-1 terms!
    # Wait no. A rank-1 tensor term has U[i]*V[j]*W[k] for all i,j,k.
    # If W has two non-zero entries, say W[0]=1 and W[3]=1, then the term
    # contributes U[i]*V[j]*1 to T[i,j,0] and U[i]*V[j]*1 to T[i,j,3].
    # That's fine - it's still one rank-1 term in the decomposition.
    
    # So the 7 Strassen terms give us 7 rank-1 terms for 2x2*2x2.
    # But wait - each M is a product of two sums, so it's actually:
    # M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is 4 rank-1 terms if expanded! But Strassen says we only need 7 multiplications total.
    # The trick is that we compute M1 as ONE multiplication, then use it in multiple C entries.
    # In the tensor decomposition, M1 contributes to both C00 and C11.
    # So the rank-1 term for M1 has:
    #   U: A00=1, A11=1 (others 0)
    #   V: B00=1, B11=1 (others 0)
    #   W: C00=1, C11=1 (others 0)
    # And this single term accounts for all 4 products in M1 contributing to both C00 and C11.
    
    # Let me verify: T[i,j,k] = sum_a U[i,a]*V[j,a]*W[k,a]
    # For the M1 term:
    # U[0,0]=1, U[3,0]=1 (A00 and A11, using row-major: A00=index0, A01=index1, A10=index2, A11=index3)
    # V[0,0]=1, V[3,0]=1 (B00=index0, B01=index1, B10=index2, B11=index3)
    # W[0,0]=1, W[3,0]=1 (C00=index0, C01=index1, C10=index2, C11=index3)
    # 
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1  (A00*B00 contributes to C00) ✓
    # T[0,0,3] += U[0,0]*V[0,0]*W[3,0] = 1*1*1 = 1  (A00*B00 contributes to C11) ✓
    # T[0,3,0] += U[0,0]*V[3,0]*W[0,0] = 1*1*1 = 1  (A00*B11 contributes to C00) ✓
    # T[0,3,3] += U[0,0]*V[3,0]*W[3,0] = 1*1*1 = 1  (A00*B11 contributes to C11) ✓
    # T[3,0,0] += U[3,0]*V[0,0]*W[0,0] = 1*1*1 = 1  (A11*B00 contributes to C00) ✓
    # T[3,0,3] += U[3,0]*V[0,0]*W[3,0] = 1*1*1 = 1  (A11*B00 contributes to C11) ✓
    # T[3,3,0] += U[3,0]*V[3,0]*W[0,0] = 1*1*1 = 1  (A11*B11 contributes to C00) ✓
    # T[3,3,3] += U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1  (A11*B11 contributes to C11) ✓
    # 
    # But we also need to check that we're NOT adding to wrong places:
    # T[1,0,0] += U[1,0]*V[0,0]*W[0,0] = 0*1*1 = 0 ✓
    # T[0,1,0] += U[0,0]*V[1,0]*W[0,0] = 1*0*1 = 0 ✓
    # Good!
    
    # Now let me define all 7 terms properly.
    # Indexing: A[i,j] -> index i*2+j, B[j,k] -> index j*2+k, C[i,k] -> index i*2+k
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is a dict: {index: coeff}
        # M1 = (A00+A11)*(B00+B11), contributes +1 to C00, +1 to C11
        (
            {0: 1, 3: 1},    # A00, A11
            {0: 1, 3: 1},    # B00, B11
            {0: 1, 3: 1}     # C00, C11
        ),
        # M2 = (A10+A11)*B00, contributes +1 to C10, -1 to C11
        (
            {2: 1, 3: 1},    # A10, A11
            {0: 1},          # B00
            {2: 1, 3: -1}    # C10, C11
        ),
        # M3 = A00*(B01-B11), contributes +1 to C01, +1 to C11
        (
            {0: 1},          # A00
            {1: 1, 3: -1},   # B01, -B11
            {1: 1, 3: 1}     # C01, C11
        ),
        # M4 = A11*(B10-B00), contributes +1 to C00, +1 to C10
        (
            {3: 1},          # A11
            {2: 1, 0: -1},   # B10, -B00
            {0: 1, 2: 1}     # C00, C10
        ),
        # M5 = (A00+A01)*B11, contributes -1 to C00, +1 to C01
        (
            {0: 1, 1: 1},    # A00, A01
            {3: 1},          # B11
            {0: -1, 1: 1}    # C00, C01
        ),
        # M6 = (A10-A00)*(B00+B01), contributes +1 to C11
        (
            {2: 1, 0: -1},   # A10, -A00
            {0: 1, 1: 1},    # B00, B01
            {3: 1}           # C11
        ),
        # M7 = (A01-A11)*(B10+B11), contributes +1 to C00
        (
            {1: 1, 3: -1},   # A01, -A11
            {2: 1, 3: 1},    # B10, B11
            {0: 1}           # C00
        ),
    ]
    return strassen_terms

def _build_strassen_4x4x4():
    """
    Build Strassen decomposition for 4x4 * 4x4 = 4x4.
    Partition into 2x2 blocks of size 2x2.
    Apply Strassen at the block level, then recursively within each block.
    This gives rank 7^2 = 49.
    
    Block structure:
    A = [[A00, A01], [A10, A11]]  each 2x2
    B = [[B00, B01], [B10, B11]]  each 2x2
    C = [[C00, C01], [C10, C11]]  each 2x2
    
    Strassen at block level gives 7 block multiplications.
    Each block multiplication is 2x2 * 2x2, which we decompose with Strassen (7 terms).
    Total: 7 * 7 = 49 rank-1 terms.
    """
    base_terms = _strassen_2x2_factors()
    
    # For 4x4 * 4x4:
    # A is 4x4, indexed as A[i,j] with i,j in {0,1,2,3}
    # Block (bi,bj) of A is A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # Within block, element (ri,rj) has global index (2*bi+ri)*4 + (2*bj+rj)
    
    # B is 4x4, indexed as B[j,k] with j,k in {0,1,2,3}
    # Block (bj,bk) of B is B[2*bj:2*bj+2, 2*bk:2*bk+2]
    # Within block, element (rj,rk) has global index (2*bj+rj)*4 + (2*bk+rk)
    
    # C is 4x4, indexed as C[i,k] with i,k in {0,1,2,3}
    # Block (bi,bk) of C is C[2*bi:2*bi+2, 2*bk:2*bk+2]
    # Within block, element (ri,rk) has global index (2*bi+ri)*4 + (2*bk+rk)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Outer Strassen: 7 block-level terms
    for outer_idx, (u_block, v_block, w_block) in enumerate(base_terms):
        # u_block: dict {block_idx: coeff} for A blocks
        # v_block: dict {block_idx: coeff} for B blocks
        # w_block: dict {block_idx: coeff} for C blocks
        
        # For each combination of A block, B block, C block:
        for a_block, a_coeff in u_block.items():
            for b_block, b_coeff in v_block.items():
                # The block multiplication is: (sum of A blocks) * (sum of B blocks)
                # But we need to decompose each 2x2 block multiplication.
                # Actually, the outer Strassen term has:
                #   U part: a_coeff for A block a_block
                #   V part: b_coeff for B block b_block
                #   W part: sum over c_block of w_coeff for C block c_block
                # 
                # The actual computation is:
                #   result_block = (a_coeff * A[a_block]) * (b_coeff * B[b_block])
                #   This is a 2x2 * 2x2 multiplication.
                #   We decompose it with inner Strassen (7 terms).
                #   The result contributes to C blocks according to w_block.
                
                # Inner Strassen decomposition of the 2x2 * 2x2:
                for inner_idx, (u_inner, v_inner, w_inner) in enumerate(base_terms):
                    # u_inner: dict {local_idx: coeff} for elements within A block
                    # v_inner: dict {local_idx: coeff} for elements within B block
                    # w_inner: dict {local_idx: coeff} for elements within C block
                    
                    U_row = {}
                    V_row = {}
                    W_row = {}
                    
                    # Fill U: elements in A block a_block
                    for local_idx, local_coeff in u_inner.items():
                        # local_idx is 0..3 for 2x2 block
                        # local row = local_idx // 2, local col = local_idx % 2
                        li = local_idx // 2
                        lj = local_idx % 2
                        # Global index in A (4x4):
                        # A is 4x4, block a_block = (bi, bj) where bi = a_block//2, bj = a_block%2
                        bi = a_block // 2
                        bj = a_block % 2
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        global_idx = gi * 4 + gj
                        coeff = a_coeff * local_coeff
                        U_row[global_idx] = U_row.get(global_idx, 0) + coeff
                    
                    # Fill V: elements in B block b_block
                    for local_idx, local_coeff in v_inner.items():
                        li = local_idx // 2  # this is the row of B (which is col of A)
                        lj = local_idx % 2  # this is the col of B
                        bj = b_block // 2
                        bk = b_block % 2
                        gi = 2 * bj + li
                        gj = 2 * bk + lj
                        global_idx = gi * 4 + gj
                        coeff = b_coeff * local_coeff
                        V_row[global_idx] = V_row.get(global_idx, 0) + coeff
                    
                    # Fill W: elements in C blocks
                    for c_block, c_coeff in w_block.items():
                        for local_idx, local_coeff in w_inner.items():
                            li = local_idx // 2
                            lj = local_idx % 2
                            ci = c_block // 2
                            cj = c_block % 2
                            gi = 2 * ci + li
                            gj = 2 * cj + lj
                            global_idx = gi * 4 + gj
                            coeff = c_coeff * local_coeff
                            W_row[global_idx] = W_row.get(global_idx, 0) + coeff
                    
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)
    
    # Convert to arrays
    rank = len(U_list)
    U = np.zeros((16, rank), dtype=object)
    V = np.zeros((16, rank), dtype=object)
    W = np.zeros((16, rank), dtype=object)
    
    for a in range(rank):
        for idx, coeff in U_list[a].items():
            if coeff != 0:
                U[idx, a] = coeff
        for idx, coeff in V_list[a].items():
            if coeff != 0:
                V[idx, a] = coeff
        for idx, coeff in W_list[a].items():
            if coeff != 0:
                W[idx, a] = coeff
    
    return U, V, W

def _build_rank1_term_for_col(U, V, W, col_offset, a_row, b_col, c_row):
    """
    Add a rank-1 term for the single-column multiplication:
    A[a_row, :] * B[:, b_col] contributes to C[c_row, b_col]
    
    Wait, I need to think about this more carefully.
    For 4x4 * 4x5:
    A is 4x4: A[i,j], i,j in 0..3
    B is 4x5: B[j,k], j in 0..3, k in 0..4
    C is 4x5: C[i,k], i in 0..3, k in 0..4
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For the first 4 columns (k=0..3), we use Strassen 4x4*4x4.
    For the 5th column (k=4), we need:
    C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    
    This is 4 separate dot products. Each is a rank-1 term:
    Term for C[i,4]: U has A[i,j]=1 for all j, V has B[j,4]=1 for all j, W has C[i,4]=1
    But that's 4 terms total (one per row i).
    
    Actually, each term is:
    U[i*4+j, a] = 1 for j=0..3  (entire row i of A)
    V[j*5+4, a] = 1 for j=0..3  (column 4 of B)
    W[4*i+4, a] = 1  (element C[i,4])
    
    Wait, the indexing:
    U index: i*m + j = i*4 + j (for A[i,j])
    V index: j*p + k = j*5 + k (for B[j,k])
    W index: k*n + i = k*4 + i (for C[i,k])
    
    So for the 5th column term for row i:
    U[i*4 + j, a] = 1 for j=0..3
    V[j*5 + 4, a] = 1 for j=0..3
    W[4*4 + i, a] = W[16+i, a] = 1
    """
    pass

def _build_strassen_4x4x5():
    """
    Build decomposition for 4x4 * 4x5.
    Use Strassen for the 4x4 * 4x4 part (columns 0-3 of B),
    plus 4 rank-1 terms for the 5th column.
    Total rank: 49 + 4 = 53.
    """
    # Get the 4x4 * 4x4 Strassen decomposition
    U_ss, V_ss, W_ss = _build_strassen_4x4x4()
    
    rank_ss = U_ss.shape[1]  # should be 49
    
    # Now add 4 terms for the 5th column
    # For each row i of the result (i=0..3):
    # C[i,4] = sum_{j=0}^{3} A[i,j] * B[j,4]
    # U index for A[i,j]: i*4 + j
    # V index for B[j,4]: j*5 + 4
    # W index for C[i,4]: 4*4 + i = 16 + i
    
    extra_rank = 4
    total_rank = rank_ss + extra_rank
    
    U = np.zeros((16, total_rank), dtype=object)
    V = np.zeros((20, total_rank), dtype=object)  # m*p = 4*5 = 20
    W = np.zeros((20, total_rank), dtype=object)  # p*n = 5*4 = 20
    
    # Copy Strassen part
    # The Strassen part handles C[i,k] for k=0..3
    # W index for C[i,k]: k*4 + i
    # For k=0..3, this is 0..15
    # U and V are the same size as in the 4x4*4x4 case for the first 4 columns
    
    for a in range(rank_ss):
        for i in range(16):
            if U_ss[i, a] != 0:
                U[i, a] = U_ss[i, a]
        for j in range(16):  # B has 4*4=16 elements for first 4 columns
            if V_ss[j, a] != 0:
                V[j, a] = V_ss[j, a]
        for k in range(16):  # C has 4*4=16 elements for first 4 columns
            if W_ss[k, a] != 0:
                W[k, a] = W_ss[k, a]
    
    # Add 4 terms for the 5th column
    for i in range(4):
        a = rank_ss + i
        for j in range(4):
            U[i * 4 + j, a] = 1
            V[j * 5 + 4, a] = 1
        W[4 * 4 + i, a] = 1  # W index for C[i,4] = 4*4 + i = 16 + i
    
    return U, V, W

def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    T = _build_target_tensor()
    
    # Try Strassen-based decomposition
    best_rank = n * m * p  # schoolbook rank as fallback
    best_U, best_V, best_W = None, None, None
    
    if time.time() < deadline:
        try:
            U_ss, V_ss, W_ss = _build_strassen_4x4x5()
            if _verify(U_ss, V_ss, W_ss, T):
                best_rank = U_ss.shape[1]
                best_U, best_V, best_W = U_ss, V_ss, W_ss
        except Exception as e:
            pass
    
    # Fallback to schoolbook
    if best_U is None:
        U, V, W = _schoolbook()
        best_rank = U.shape[1]
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                if val == 0:
                    r.append(0)
                elif isinstance(val, (int, np.integer)):
                    r.append(int(val))
                elif isinstance(val, float):
                    if val == int(val):
                        r.append(int(val))
                    else:
                        r.append(val)
                elif isinstance(val, Fraction):
                    r.append(val)
                else:
                    try:
                        r.append(int(val))
                    except:
                        try:
                            r.append(float(val))
                        except:
                            r.append(val)
            result.append(r)
        return result
    
    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
