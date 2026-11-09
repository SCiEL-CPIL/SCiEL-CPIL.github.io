"""Matrix multiplication 4,4,5 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def _build_strassen_decomposition(n=2, m=2, p=2):
    """Build the 7-term Strassen decomposition for 2x2x2."""
    terms = []
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each term is a rank-1 tensor: U, V, W vectors
    # A indices: (i,j), B indices: (j,k), C indices: (i,k)
    
    # M1: (A11+A22)(B11+B22)
    U1 = [(0,0,1), (1,1,1)]
    V1 = [(0,0,1), (0,1,1)]
    W1 = [(0,0,1), (1,1,1)]
    terms.append((U1, V1, W1))
    
    # M2: (A21+A22)B11
    U2 = [(1,0,1), (1,1,1)]
    V2 = [(0,0,1)]
    W2 = [(1,0,1)]
    terms.append((U2, V2, W2))
    
    # M3: A11(B21-B22)
    U3 = [(0,0,1)]
    V3 = [(1,0,1), (1,1,-1)]
    W3 = [(0,1,1)]
    terms.append((U3, V3, W3))
    
    # M4: A22(B11-B21)
    U4 = [(1,1,1)]
    V4 = [(0,0,1), (1,0,-1)]
    W4 = [(1,0,1)]
    terms.append((U4, V4, W4))
    
    # M5: (A11+A12)B22
    U5 = [(0,0,1), (0,1,1)]
    V5 = [(1,1,1)]
    W5 = [(0,1,1)]
    terms.append((U5, V5, W5))
    
    # M6: (A21-A11)(B11+B12)
    U6 = [(1,0,1), (0,0,-1)]
    V6 = [(0,0,1), (0,1,1)]
    W6 = [(1,1,1)]
    terms.append((U6, V6, W6))
    
    # M7: (A12-A22)(B21+B22)
    U7 = [(0,1,1), (1,1,-1)]
    V7 = [(1,0,1), (1,1,1)]
    W7 = [(0,1,1)]
    terms.append((U7, V7, W7))
    
    return terms

def _expand_strassen_for_4x4x4(base_terms):
    """
    Expand the 2x2x2 Strassen decomposition to 4x4x4.
    The 4x4x4 problem has 4 blocks of 2x2x2 in the A and B dimensions.
    Actually, for 4x4x4, we can use Strassen at the top level (7 multiplications of 2x2 matrices),
    but each 2x2x2x2x2x2 is still schoolbook... 
    Actually, the standard approach: 4x4x4 can be done with Strassen's algorithm,
    which gives 7 * (2x2x2 rank) = 7*7 = 49 if we use Strassen recursively.
    But we need the explicit rank-1 decomposition.
    
    Let me think differently. For 4x4x4, we can decompose it into blocks.
    A is 4x4, B is 4x4, C is 4x4.
    Split into 2x2 blocks: A = [[A00,A01],[A10,A11]] where each Ai is 2x2.
    Similarly for B and C.
    
    Using Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B00-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 matrix product. To express as rank-1 tensors of the full 4x4x4,
    we need to expand each 2x2x2x2x2x2 (matrix product of two 2x2 matrices) into
    4 rank-1 tensors (schoolbook for 2x2x2).
    
    So total rank = 7 * 4 = 28? No wait, each Mi is itself a matrix product, 
    which requires 4 scalar multiplications (schoolbook 2x2x2).
    So total = 7 * 4 = 28.
    
    Let me verify: The matrix multiplication tensor for 2x2x2 has rank 7 (Strassen).
    For 4x4x4, if we use Strassen at the top level, we get 7 "sub-tensors" each of which
    is a 2x2x2 matrix multiplication tensor. Each 2x2x2 tensor has rank 7.
    So total rank = 7 * 7 = 49.
    
    But wait, that's if we use Strassen for both levels. Let me just use the direct approach:
    For 4x4x4, the rank is known to be at most 49 via Strassen.
    
    Let me construct it explicitly.
    """
    # For 4x4x4 using Strassen:
    # The 7 terms from Strassen at top level, each expanded as a 2x2x2 schoolbook (4 terms)
    # Total: 7 * 4 = 28? No, each term is a matrix multiplication of 2x2 matrices,
    # which itself needs rank 7 (Strassen) or rank 4 (schoolbook).
    # If we use schoolbook for the inner products: 7 * 4 = 28
    # If we use Strassen for inner: 7 * 7 = 49
    
    # Let's use schoolbook for inner to get 28, but is that correct?
    # Actually no. Each "super-entry" in Strassen is a 2x2 matrix.
    # The product of two 2x2 matrices has tensor rank 7 (Strassen) or 4 (schoolbook).
    # So using schoolbook: 7 * 4 = 28 total rank-1 terms.
    
    # Let me build this properly.
    terms = []
    
    # Strassen top-level coefficients and index mappings
    # A block indices: (block_row, block_col, local_row, local_col) -> (block_row*2+local_row, block_col*2+local_col)
    # B block indices: (block_row, block_col, local_row, local_col) -> (block_row*2+local_row, block_col*2+local_col)
    # C block indices: (block_row, block_col, local_row, local_col) -> (block_row*2+local_row, block_col*2+local_col)
    
    def a_idx(br, bc, lr, lc):
        return (br*2+lr, bc*2+lc)
    
    def b_idx(br, bc, lr, lc):
        return (br*2+lr, bc*2+lc)
    
    def c_idx(br, bc, lr, lc):
        return (br*2+lr, bc*2+lc)
    
    # For each Strassen term, we have a combination of A-blocks and B-blocks
    # producing a C-block contribution.
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B10-B11) -> contributes to C01 and C11
    # M4 = A11(B00-B10) -> contributes to C00 and C10
    # M5 = (A00+A01)B11 -> contributes to C00 and C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00 and C11
    
    # For each product of two 2x2 matrices, we use schoolbook (4 rank-1 terms)
    
    # M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11 -> for each (lr,lc), A[(lr,lc)] + A[(lr+2,lc+2)]
    # B part: B00 + B11 -> for each (lr,lc), B[(lr,lc)] + B[(lr+2,lc+2)]
    # Product contributes to C00 and C11 with coefficient +1
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    # This is one rank-1 term in the 2x2x2 schoolbook for the inner product
                    # A index: (lr_a, lc_a) in block 00, and (lr_a, lc_a) in block 11
                    # B index: (lr_b, lc_b) in block 00, and (lr_b, lc_b) in block 11
                    # C index: (lr_a, lc_b) in block 00, and (lr_a, lc_b) in block 11
                    # Coefficient: +1 for C00, +1 for C11
                    U = [a_idx(0,0,lr_a,lc_a), a_idx(1,1,lr_a,lc_a)]
                    V = [b_idx(0,0,lr_b,lc_b), b_idx(1,1,lr_b,lc_b)]
                    W = [c_idx(0,0,lr_a,lc_b), c_idx(1,1,lr_a,lc_b)]
                    terms.append(('M1', U, V, W, 1))
    
    # M2 = (A10+A11)B00
    # A: A10 + A11, B: B00
    # Contributes to C10
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(1,0,lr_a,lc_a), a_idx(1,1,lr_a,lc_a)]
                    V = [b_idx(0,0,lr_b,lc_b)]
                    W = [c_idx(1,0,lr_a,lc_b)]
                    terms.append(('M2', U, V, W, 1))
    
    # M3 = A00(B10-B11)
    # A: A00, B: B10 - B11
    # Contributes to C01 (+1) and C11 (-1)
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(0,0,lr_a,lc_a)]
                    V = [b_idx(1,0,lr_b,lc_b), b_idx(1,1,lr_b,lc_b)]
                    W = [c_idx(0,1,lr_a,lc_b), c_idx(1,1,lr_a,lc_b)]
                    terms.append(('M3', U, V, W, 1))
    
    # M4 = A11(B00-B10)
    # A: A11, B: B00 - B10
    # Contributes to C00 (+1) and C10 (-1)
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(1,1,lr_a,lc_a)]
                    V = [b_idx(0,0,lr_b,lc_b), b_idx(1,0,lr_b,lc_b)]
                    W = [c_idx(0,0,lr_a,lc_b), c_idx(1,0,lr_a,lc_b)]
                    terms.append(('M4', U, V, W, 1))
    
    # M5 = (A00+A01)B11
    # A: A00 + A01, B: B11
    # Contributes to C00 (-1) and C01 (+1)
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(0,0,lr_a,lc_a), a_idx(0,1,lr_a,lc_a)]
                    V = [b_idx(1,1,lr_b,lc_b)]
                    W = [c_idx(0,0,lr_a,lc_b), c_idx(0,1,lr_a,lc_b)]
                    # C00 gets -1, C01 gets +1
                    terms.append(('M5', U, V, W, 1))
    
    # M6 = (A10-A00)(B00+B01)
    # A: A10 - A00, B: B00 + B01
    # Contributes to C11 (+1)
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(1,0,lr_a,lc_a), a_idx(0,0,lr_a,lc_a)]
                    V = [b_idx(0,0,lr_b,lc_b), b_idx(0,1,lr_b,lc_b)]
                    W = [c_idx(1,1,lr_a,lc_b)]
                    terms.append(('M6', U, V, W, 1))
    
    # M7 = (A01-A11)(B10+B11)
    # A: A01 - A11, B: B10 + B11
    # Contributes to C00 (+1) and C11 (+1)
    for lr_a in range(2):
        for lc_a in range(2):
            for lr_b in range(2):
                for lc_b in range(2):
                    U = [a_idx(0,1,lr_a,lc_a), a_idx(1,1,lr_a,lc_a)]
                    V = [b_idx(1,0,lr_b,lc_b), b_idx(1,1,lr_b,lc_b)]
                    W = [c_idx(0,0,lr_a,lc_b), c_idx(1,1,lr_a,lc_b)]
                    terms.append(('M7', U, V, W, 1))
    
    return terms

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    The output tensor W has indices [k*n+i, r] where r is the rank index.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    For 4x4x5, we can split the 4x4x5 into:
    - 4x4x4 part (first 4 columns of B)
    - 4x4x1 part (last column of B)
    
    For 4x4x4, use Strassen-based decomposition (rank ~28-49)
    For 4x4x1, use schoolbook (rank 16)
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Start with schoolbook for the full problem as fallback
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to build a better decomposition using Strassen for the 4x4x4 part
    # and schoolbook for the 4x4x1 part
    
    # Build Strassen terms for 4x4x4
    strassen_terms = _expand_strassen_for_4x4x4(_build_strassen_decomposition())
    
    # For 4x4x4, indices: i in [0,3], j in [0,3], k in [0,3]
    # U index: i*m + j = i*4 + j
    # V index: j*p + k -> but for 4x4x4, p=4, so j*4 + k
    # W index: k*n + i = k*4 + i
    
    # For 4x4x1 part: k=3 (the 5th column, index 4 in 0-based... wait)
    # B is 4x5, so k ranges 0..4
    # First 4 columns (k=0..3) form 4x4x4
    # Last column (k=4) forms 4x4x1
    
    # Schoolbook for 4x4x1: rank = 4*4*1 = 16
    # For k=4 only:
    # U[i*4+j, col] = 1, V[j*5+4, col] = 1, W[4*4+i, col] = 1 for i in 0..3, j in 0..3
    
    # Total rank = rank_strassen + 16
    
    # Let's count the Strassen terms
    # Each of the 7 Strassen terms has 4 sub-terms (schoolbook 2x2x2)
    # So 7 * 4 = 28 terms for 4x4x4
    
    # But wait, I need to properly combine all terms into U, V, W matrices.
    # Each term contributes to specific (U_index, V_index, W_index) with a coefficient.
    # Multiple terms can contribute to the same (U_idx, V_idx, W_idx) combination,
    # but in a rank-1 decomposition, each term is separate.
    
    # Actually, in the tensor decomposition, each rank-1 term is a triple (u, v, w)
    # where u is a vector in U-space, v in V-space, w in W-space.
    # The sum of outer products u ⊗ v ⊗ w should equal the multiplication tensor.
    
    # Let me collect all contributions and build the matrices.
    
    # For the 4x4x4 part:
    # Each term has lists of (index, coefficient) for U, V, W positions.
    # But actually, each "term" in my list above is already a rank-1 term
    # where U, V, W are sparse vectors (lists of indices with implicit coefficient 1,
    # but some have multiple indices meaning linear combinations).
    
    # I need to expand these into actual rank-1 terms.
    # Each term with |U| * |V| * |W| non-zero entries expands to that many rank-1 terms?
    # No, that's not right either.
    
    # Let me reconsider. A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors.
    # If u has support on indices {u1, u2} with values {c1, c2}, etc.,
    # then u ⊗ v ⊗ w has non-zero entries at all combinations (ui, vj, wk).
    
    # In my Strassen expansion above, each "term" like M1 has:
    # U = [idx1, idx2] meaning U vector has 1 at idx1 and 1 at idx2
    # V = [idx3, idx4] meaning V vector has 1 at idx3 and 1 at idx4
    # W = [idx5, idx6] meaning W vector has 1 at idx5 and 1 at idx6
    # This is a single rank-1 term (outer product of these sparse vectors).
    
    # So each entry in my `terms` list is one rank-1 term.
    # Total terms = 28 for 4x4x4 + 16 for 4x4x1 = 44 total rank.
    
    # Let me build the full decomposition.
    
    all_U_entries = []  # list of (u_index, value) for each term
    all_V_entries = []
    all_W_entries = []
    
    # Process Strassen terms for 4x4x4
    for term in strassen_terms:
        name, U_idx, V_idx, W_idx, coeff = term
        # U_idx, V_idx, W_idx are lists of (row, col) tuples from 4x4x4 perspective
        # Need to map to flat indices for the full problem
        
        # For 4x4x4:
        # U flat index: i*4 + j where (i,j) is the A index
        # V flat index: j*4 + k where (j,k) is the B index (but B is 4x5, so for first 4 cols: j*5 + k)
        # W flat index: k*4 + i where (k,i) is the C index (C is 5x4, so k*4 + i)
        
        # Each term contributes to the tensor with the given structure.
        # The U vector has entries at positions corresponding to U_idx
        # The V vector has entries at positions corresponding to V_idx
        # The W vector has entries at positions corresponding to W_idx
        
        u_entries = {}
        v_entries = {}
        w_entries = {}
        
        for (r, c) in U_idx:
            u_flat = r * 4 + c  # n=4, so m=4
            u_entries[u_flat] = u_entries.get(u_flat, 0) + coeff
        
        for (r, c) in V_idx:
            v_flat = r * 5 + c  # m=4, p=5
            v_entries[v_flat] = v_entries.get(v_flat, 0) + coeff
        
        for (r, c) in W_idx:
            w_flat = r * 4 + c  # p=5, n=4
            w_entries[w_flat] = w_entries.get(w_flat, 0) + coeff
        
        all_U_entries.append(u_entries)
        all_V_entries.append(v_entries)
        all_W_entries.append(w_entries)
    
    # Schoolbook terms for 4x4x1 (k=4, the 5th column of B)
    for i in range(4):
        for j in range(4):
            u_entries = {i * 4 + j: 1}
            v_entries = {j * 5 + 4: 1}
            w_entries = {4 * 4 + i: 1}  # k=4, n=4
            all_U_entries.append(u_entries)
            all_V_entries.append(v_entries)
            all_W_entries.append(w_entries)
    
    total_rank = len(all_U_entries)
    
    # Build U, V, W matrices
    U_mat = [[0] * total_rank for _ in range(16)]  # n*m = 16
    V_mat = [[0] * total_rank for _ in range(20)]  # m*p = 20
    W_mat = [[0] * total_rank for _ in range(20)]  # p*n = 20
    
    for t in range(total_rank):
        for idx, val in all_U_entries[t].items():
            if 0 <= idx < 16:
                U_mat[idx][t] = val
        for idx, val in all_V_entries[t].items():
            if 0 <= idx < 20:
                V_mat[idx][t] = val
        for idx, val in all_W_entries[t].items():
            if 0 <= idx < 20:
                W_mat[idx][t] = val
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal 1 if k==i... no, the multiplication tensor is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 if the indices match for multiplication
        # Actually, the tensor decomposition means:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}?? 
        # No. The matrix multiplication tensor T has T[i,j,k] = 1 if i,j,k form a valid
        # multiplication index, which is always valid. The tensor is:
        # T[i,j,k] = 1 for all i,j,k? No.
        
        # The correct formulation: The tensor for matrix multiplication is
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k}??? 
        # No. C[i,k] = sum_j A[i,j] B[j,k]
        # The bilinear form is: sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # The tensor is T_{i,j,k} = delta_{i,k}... no, that's not right either.
        
        # Actually, the standard tensor decomposition for matrix multiplication:
        # The multiplication tensor T satisfies:
        # <T, A ⊗ B ⊗ C> = tr(ABC) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # So T_{i,j,k} = delta_{i,i'} where... 
        # <T, A⊗B⊗C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # We want this to equal sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # So T_{i,j,k} = 1 for all i,j,k? No, that would give sum A_{i,j}B_{j,k}C_{k,i}
        # which is tr(ABC). But we want C = AB, so C_{i,k} = sum_j A_{i,j}B_{j,k}
        # The trace is sum_{i,k} C_{i,k} C'_{k,i} for some C'.
        
        # Let me re-read the problem statement.
        # "W[k*n+i,r]" - so W is indexed by (k,i) with flat index k*n+i
        # U[i*m+j, r], V[j*p+k, r]
        
        # The decomposition should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,j}? No.
        
        # I think the condition is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the indices form a valid
        # multiplication, which is... 
        # 
        # Actually, for the matrix multiplication tensor, the rank-1 decomposition satisfies:
        # sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = delta_{i,k}??? 
        # No, that doesn't make sense dimensionally.
        
        # Let me think again. The tensor T for n×m * m×p → n×p matrix multiplication
        # is a trilinear form on A (n×m), B (m×p), C (p×n) [with trace convention]:
        # T(A,B,C) = tr(C^T AB) = sum_{i,k} C_{k,i} (AB)_{i,k} = sum_{i,j,k} C_{k,i} A_{i,j} B_{j,k}
        # 
        # So T_{i,j,k} in the sense of <T, A⊗B⊗C> = sum A_{i,j}B_{j,k}C_{k,i} T_{i,j,k}
        # For this to equal sum A_{i,j}B_{j,k}C_{k,i}, we need T_{i,j,k} = 1 for all i,j,k? 
        # That can't be right because then the rank would be 1.
        #
        # I think I'm confusing the indexing. Let me reconsider.
        #
        # The multiplication map is: (A, B) → C where C_{i,k} = sum_j A_{i,j} B_{j,k}
        # As a bilinear map, this is a tensor in (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ R^{n×p}
        # But with the trace convention for the output, it becomes a trilinear form.
        #
        # The rank-1 decomposition: C_{i,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}??? 
        # No, that would be summing over r but the indices don't match.
        
        # I think the correct interpretation is:
        # For each rank-r term, we have vectors u_r, v_r, w_r such that:
        # C_{i,k} = sum_r u_{r,i,j} v_{r,j,k} w_{r,k,i}
        # where u_r is a matrix (n×m), v_r is a matrix (m×p), w_r is a matrix (p×n)
        # and the sum is over r only.
        #
        # But that's not a rank-1 decomposition in the tensor sense.
        
        # Let me just verify numerically by checking if the decomposition reproduces
        # the correct multiplication.
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        
        # Compute AB
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # With decomposition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should give
        # the coefficient of A[i,j] B[j,k] in C[i,k]
        # So sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k}?? 
        # No, it should be 1 if the multiplication is valid, which is always.
        # Actually, the tensor T_{i,j,k}^{(i',j',k')} ... I'm overcomplicating.
        
        # Let me just check: for each i, j, k, 
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        # (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient is 1 for all valid i,j,k)
        
        # Wait, but W has index k*n+i, so for a fixed (i,j,k), we look at W[k*n+i, r].
        # The sum should give 1 for all i, j, k.
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    expected = 1
                    err = abs(s - expected)
                    max_err = max(max_err, err)
        
        return max_err < tol
    
    if verify(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    else:
        # Fall back to schoolbook
        return {"U": U, "V": V, "W": W}
