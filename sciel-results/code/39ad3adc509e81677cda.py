"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses block decomposition 5=2+2+1 with Strassen for 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # Block partition: sizes [2, 2, 1]
    block_sizes = [2, 2, 1]
    num_blocks = len(block_sizes)
    
    # Precompute block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Collect all terms: (u_indices, v_indices, w_indices, coeffs_u, coeffs_v, coeffs_w)
    # Each term is a rank-1 tensor contribution
    terms = []
    
    # Strassen 2x2x2 decomposition
    # For 2x2 matrices A, B -> C
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    strassen_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # M1: A00+A11, B00+B11, contributes to C00(+1), C11(+1)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2: A10+A11, B00, contributes to C10(+1), C11(-1)
        ([0,1,1,0], [1,0,0,0], [0,0,1,-1]),
        # M3: A00, B10-B11, contributes to C01(+1), C11(+1)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        # M4: A11, B01-B10, contributes to C00(+1), C10(+1)
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),
        # M5: A00+A10, B11, contributes to C00(-1), C01(+1)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M6: A10-A00, B00+B10, contributes to C11(+1)
        ([-1,1,0,0], [1,1,0,0], [0,0,0,1]),
        # M7: A01-A11, B01+B11, contributes to C00(+1)
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    # Schoolbook 2x1x1: A(2x1) * B(1x1) -> C(2x1)
    # A0*B0 -> C0, A1*B0 -> C1
    schoolbook_2x1x1 = [
        ([1,0], [1], [1,0]),
        ([0,1], [1], [0,1]),
    ]
    
    # Schoolbook 2x2x1: A(2x2) * B(2x1) -> C(2x1)
    # A00*B00 -> C0, A01*B10 -> C0, A10*B00 -> C1, A11*B10 -> C1
    schoolbook_2x2x1 = [
        ([1,0,0,0], [1,0], [1,0]),
        ([0,1,0,0], [0,1], [1,0]),
        ([0,0,1,0], [1,0], [0,1]),
        ([0,0,0,1], [0,1], [0,1]),
    ]
    
    # Schoolbook 2x1x2: A(2x1) * B(1x2) -> C(2x2)
    # A0*B00 -> C00, A0*B01 -> C01, A1*B00 -> C10, A1*B01 -> C11
    schoolbook_2x1x2 = [
        ([1,0], [1,0], [1,0,0,0]),
        ([1,0], [0,1], [0,1,0,0]),
        ([0,1], [1,0], [0,0,1,0]),
        ([0,1], [0,1], [0,0,0,1]),
    ]
    
    # Schoolbook 1x2x2: A(1x2) * B(2x2) -> C(1x2)
    # A00*B00 -> C0, A00*B10 -> C0, A01*B01 -> C1, A01*B11 -> C1
    schoolbook_1x2x2 = [
        ([1,0], [1,0,0,0], [1,0]),
        ([1,0], [0,1,0,0], [1,0]),
        ([0,1], [0,0,1,0], [0,1]),
        ([0,1], [0,0,0,1], [0,1]),
    ]
    
    # Schoolbook 1x2x1: A(1x2) * B(2x1) -> C(1x1)
    # A00*B00 -> C0, A01*B10 -> C0
    schoolbook_1x2x1 = [
        ([1,0], [1,0], [1]),
        ([0,1], [0,1], [1]),
    ]
    
    # Schoolbook 1x1x2: A(1x1) * B(1x2) -> C(1x2)
    schoolbook_1x1x2 = [
        ([1], [1,0], [1,0]),
        ([1], [0,1], [0,1]),
    ]
    
    # Schoolbook 1x1x1: A(1x1) * B(1x1) -> C(1x1)
    schoolbook_1x1x1 = [
        ([1], [1], [1]),
    ]
    
    # For each block triple (bi, bj, bk), determine the sub-multiplication type
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    sub_terms = strassen_terms
                elif si == 2 and sj == 2 and sk == 1:
                    sub_terms = schoolbook_2x2x1
                elif si == 2 and sj == 1 and sk == 2:
                    sub_terms = schoolbook_2x1x2
                elif si == 2 and sj == 1 and sk == 1:
                    sub_terms = schoolbook_2x1x1
                elif si == 1 and sj == 2 and sk == 2:
                    sub_terms = schoolbook_1x2x2
                elif si == 1 and sj == 2 and sk == 1:
                    sub_terms = schoolbook_1x2x1
                elif si == 1 and sj == 1 and sk == 2:
                    sub_terms = schoolbook_1x1x2
                elif si == 1 and sj == 1 and sk == 1:
                    sub_terms = schoolbook_1x1x1
                else:
                    continue
                
                for a_coeffs, b_coeffs, c_coeffs in sub_terms:
                    terms.append((bi, bj, bk, a_coeffs, b_coeffs, c_coeffs))
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (bi, bj, bk, a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
        ai_off = offsets[bi]
        aj_off = offsets[bj]
        ak_off = offsets[bk]
        
        # A is n×m, indexed as A[i,j] -> U[i*m+j]
        # For block (bi, bj), the local indices range over [ai_off, ai_off+si) × [aj_off, aj_off+sj)
        # a_coeffs is indexed as [local_i * sj + local_j] for A
        for local_i in range(block_sizes[bi]):
            for local_j in range(block_sizes[bj]):
                global_i = ai_off + local_i
                global_j = aj_off + local_j
                coeff = a_coeffs[local_i * block_sizes[bj] + local_j]
                if coeff != 0:
                    U[global_i * m + global_j][col] = coeff
        
        # B is m×p, indexed as B[j,k] -> V[j*p+k]
        # For block (bj, bk), local indices [aj_off, aj_off+sj) × [ak_off, ak_off+sk)
        for local_j in range(block_sizes[bj]):
            for local_k in range(block_sizes[bk]):
                global_j = aj_off + local_j
                global_k = ak_off + local_k
                coeff = b_coeffs[local_j * block_sizes[bk] + local_k]
                if coeff != 0:
                    V[global_j * p + global_k][col] = coeff
        
        # C is p×n... wait, C is n×p typically, but the trace convention is W[k*n+i, r]
        # So C[i,k] is stored at W[k*n + i]
        # For block (bi, bk), local indices [ai_off, ai_off+si) × [ak_off, ak_off+sk)
        for local_i in range(block_sizes[bi]):
            for local_k in range(block_sizes[bk]):
                global_i = ai_off + local_i
                global_k = ak_off + local_k
                coeff = c_coeffs[local_i * block_sizes[bk] + local_k]
                if coeff != 0:
                    W[global_k * n + global_i][col] = coeff
    
    if time.time() > deadline - 5:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
