"""Matrix multiplication 4x4x4 - Strassen rank 49 decomposition"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _strassen_2x2_decomposition():
    """Return U, V, W for 2x2 matrix multiplication with rank 7 (Strassen).
    
    Convention: A is 2x2, B is 2x2, C is 2x2.
    A indices: A[0,0], A[0,1], A[1,0], A[1,1] -> flat index i*m+j with m=2
    B indices: B[0,0], B[0,1], B[1,0], B[1,1] -> flat index j*p+k with p=2
    C indices: C[0,0], C[0,1], C[1,0], C[1,1] -> flat index k*n+i with n=2
    
    The tensor T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    should equal A[i,j] * B[j,k] for the (i,k) entry of C.
    """
    # Strassen 7 multiplications for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, c] where i,j in {0,1}, c in {0..6}
    # V[j*2+k, c] where j,k in {0,1}, c in {0..6}
    # W[k*2+i, c] where k,i in {0,1}, c in {0..6}
    
    # U rows: (i,j) -> row index i*2+j: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V rows: (j,k) -> row index j*2+k: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W rows: (k,i) -> row index k*2+i: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00->+1, A11->+1
    U[0][0] = 1; U[3][0] = 1
    # V: B00->+1, B11->+1
    V[0][0] = 1; V[3][0] = 1
    # W: C00->+1, C11->+1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)(B00)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # W: C10->+1
    W[2][1] = 1
    
    # M3 = (A00)(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # W: C01->+1
    W[1][2] = 1
    
    # M4 = (A11)(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # W: C00->+1, C10->+1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)(B11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # W: C00->-1, C01->+1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # W: C11->+1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # W: C00->+1
    W[0][6] = 1
    
    return U, V, W

def _combine_strassen_4x4():
    """Combine two levels of Strassen to get 4x4x4 decomposition with rank 49.
    
    Level 1: Split 4x4 into 2x2 blocks of 2x2 submatrices.
    Use 7 "virtual" 2x2 matrix multiplications.
    Level 2: Each virtual 2x2 multiplication uses Strassen rank 7.
    Total rank: 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # For 4x4, we have:
    # A is 4x4, split into A11, A12, A21, A22 each 2x2
    # B is 4x4, split into B11, B12, B21, B22 each 2x2
    # C is 4x4, split into C11, C12, C21, C22 each 2x2
    #
    # Strassen at block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix product, decomposed with Strassen rank 7.
    #
    # U4 has shape (16, 49): rows indexed by (i,j) in 4x4, i.e., (block_i, sub_i, sub_j)
    # Actually, let's think in terms of the full 4x4 indices.
    #
    # Let's use a cleaner construction. 
    # The 4x4 multiplication tensor can be seen as:
    # T[a,b,c] where a,c in {0,1,2,3} (output indices), b in {0,1,2,3} (inner)
    # 
    # Block decomposition: a = 2*ai + ai', c = 2*ci + ci', b = 2*bi + bi'
    # where ai,ci,bi in {0,1} and ai',ci',bi' in {0,1}
    #
    # The Strassen block-level coefficients give us how to combine the 7 block products.
    # Each block product (say M1 = (A11+A22)(B11+B22)) is itself a 2x2 matmul.
    
    # Let me build this more carefully using the tensor structure.
    # 
    # For 4x4: n=m=p=4
    # U: shape (16, rank), U[i*4+j, c]
    # V: shape (16, rank), V[j*4+k, c]  
    # W: shape (16, rank), W[k*4+i, c]
    #
    # The tensor element T[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k]
    # with W[k*4+i, c] factor.
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Block-level Strassen coefficients:
    # For each of the 7 block products, we have:
    # - Which A blocks contribute and with what signs
    # - Which B blocks contribute and with what signs  
    # - Which C blocks the result contributes to and with what signs
    
    # Block indices: 00, 01, 10, 11 (row_block, col_block)
    # A blocks: A[br, bc] where br, bc in {0,1}
    # B blocks: B[br, bc]
    # C blocks: C[br, bc]
    
    # The 7 Strassen products at block level:
    # prod_idx: (A_blocks: list of (block, coeff), B_blocks: list of (block, coeff), 
    #            C_blocks: list of (block, coeff))
    
    strassen_blocks = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        # M2 = (A10+A11)(B00) -> C10 += 1
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1)]),
        # M3 = (A00)(B01-B11) -> C01 += 1
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1)]),
        # M4 = (A11)(B10-B00) -> C00 += 1, C10 += 1
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        # M5 = (A00+A01)(B11) -> C00 -= 1, C01 += 1
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1)]),
    ]
    
    # Now for each block product (prod_idx), and for each of the 7 Strassen 
    # scalar multiplications within that block product, we add contributions.
    
    # For a block product with A_blocks, B_blocks, C_blocks:
    # The A block (ar, ac) has elements A[2*ar+ai, 2*ac+aj] for ai,aj in {0,1}
    # The B block (br, bc) has elements B[2*br+bj, 2*bc+bk] for bj,bk in {0,1}
    # The C block (cr, cc) has elements C[2*cr+ci, 2*cc+ck] for ci,ck in {0,1}
    #
    # The inner Strassen decomposition (U2,V2,W2) handles the 2x2 structure.
    # U2[ai*2+aj, s] for the A side of the 2x2 block
    # V2[bj*2+bk, s] for the B side
    # W2[ck*2+ci, s] for the C side (note: W2[k*2+i, c] convention)
    
    for prod_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_blocks):
        for s in range(7):  # 7 Strassen scalar mults within each block product
            col = prod_idx * 7 + s
            
            # A side: sum over A blocks
            for (ar, ac), a_coeff in a_blocks:
                for ai in range(2):
                    for aj in range(2):
                        # A element at global index (2*ar+ai, 2*ac+aj)
                        # U row index: i*4+j = (2*ar+ai)*4 + (2*ac+aj)
                        i_global = 2*ar + ai
                        j_global = 2*ac + aj
                        u_row = i_global * 4 + j_global
                        # Inner Strassen: U2[ai*2+aj, s]
                        u2_val = U2[ai*2 + aj][s]
                        if u2_val != 0:
                            U[u_row][col] += a_coeff * u2_val
            
            # B side: sum over B blocks
            for (br, bc), b_coeff in b_blocks:
                for bj in range(2):
                    for bk in range(2):
                        # B element at global index (2*br+bj, 2*bc+bk)
                        # V row index: j*4+k = (2*br+bj)*4 + (2*bc+bk)
                        j_global = 2*br + bj
                        k_global = 2*bc + bk
                        v_row = j_global * 4 + k_global
                        # Inner Strassen: V2[bj*2+bk, s]
                        v2_val = V2[bj*2 + bk][s]
                        if v2_val != 0:
                            V[v_row][col] += b_coeff * v2_val
            
            # C side: sum over C blocks
            for (cr, cc), c_coeff in c_blocks:
                for ci in range(2):
                    for ck in range(2):
                        # C element at global index (2*cr+ci, 2*cc+ck)
                        # W row index: k*4+i = (2*cc+ck)*4 + (2*cr+ci)
                        k_global = 2*cc + ck
                        i_global = 2*cr + ci
                        w_row = k_global * 4 + i_global
                        # Inner Strassen: W2[ck*2+ci, s]
                        w2_val = W2[ck*2 + ci][s]
                        if w2_val != 0:
                            W[w_row][col] += c_coeff * w2_val
    
    return U, V, W

def construct():
    """Return rank-49 decomposition of 4x4 matrix multiplication using Strassen."""
    U, V, W = _combine_strassen_4x4()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
