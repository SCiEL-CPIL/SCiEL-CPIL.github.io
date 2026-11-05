import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition for 4x4x8 using Strassen (rank 49) for first 4 cols 
    # and Schoolbook (rank 16) for last 4 cols.
    # Then project to 3x3x8.
    
    # 1. Define 2x2x2 Strassen decomposition
    # Returns lists of (u_vec, v_vec, w_vec) for 2x2x2
    # u is 4x1, v is 4x1, w is 4x1
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    
    def strassen_2x2_2x2():
        terms = []
        # M1 = (A11+A22)(B11+B22)
        # C11 += M1, C22 += M1
        # u: A11, A22 are 1, others 0. v: B11, B22 are 1. w: C11, C22 are 1.
        # u indices: 0 (11), 3 (22)
        # v indices: 0 (11), 3 (22)
        # w indices: 0 (11), 3 (22)
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[3]=1
        terms.append((u, v, w))
        
        # M2 = (A21+A22)B11
        # C21 += M2
        # u: A21 (2), A22 (3)
        # v: B11 (0)
        # w: C21 (2)
        u = [0]*4; u[2]=1; u[3]=1
        v = [0]*4; v[0]=1
        w = [0]*4; w[2]=1
        terms.append((u, v, w))
        
        # M3 = A11(B12-B22)
        # C12 += M3
        # u: A11 (0)
        # v: B12 (1), B22 (3) with -1
        # w: C12 (1)
        u = [0]*4; u[0]=1
        v = [0]*4; v[1]=1; v[3]=-1
        w = [0]*4; w[1]=1
        terms.append((u, v, w))
        
        # M4 = A22(B21-B11)
        # C11 += M4, C21 += M4
        # u: A22 (3)
        # v: B21 (2), B11 (0) with -1
        # w: C11 (0), C21 (2)
        u = [0]*4; u[3]=1
        v = [0]*4; v[2]=1; v[0]=-1
        w = [0]*4; w[0]=1; w[2]=1
        terms.append((u, v, w))
        
        # M5 = (A11+A12)B22
        # C11 -= M5, C12 += M5
        # u: A11 (0), A12 (1)
        # v: B22 (3)
        # w: C11 (0) -1, C12 (1) 1
        u = [0]*4; u[0]=1; u[1]=1
        v = [0]*4; v[3]=1
        w = [0]*4; w[0]=-1; w[1]=1
        terms.append((u, v, w))
        
        # M6 = (A21-A11)(B11+B12)
        # C22 += M6
        # u: A21 (2), A11 (0) with -1
        # v: B11 (0), B12 (1)
        # w: C22 (3)
        u = [0]*4; u[2]=1; u[0]=-1
        v = [0]*4; v[0]=1; v[1]=1
        w = [0]*4; w[3]=1
        terms.append((u, v, w))
        
        # M7 = (A12-A22)(B21+B22)
        # C11 += M7
        # u: A12 (1), A22 (3) with -1
        # v: B21 (2), B22 (3)
        # w: C11 (0)
        u = [0]*4; u[1]=1; u[3]=-1
        v = [0]*4; v[2]=1; v[3]=1
        w = [0]*4; w[0]=1
        terms.append((u, v, w))
        
        return terms

    def schoolbook_2x2_2x2():
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = [0]*4; u[i*2+j] = 1
                    v = [0]*4; v[j*2+k] = 1
                    w = [0]*4; w[k*2+i] = 1
                    terms.append((u, v, w))
        return terms

    # 2. Construct 4x4x4 decomposition
    # Strassen 4x4x4 uses 7 multiplications of 2x2x2 blocks.
    # We need to map the 2x2 indices to 4x4 indices.
    # Block (bi, bj) where bi, bj in {0,1} corresponds to rows/cols 2*bi..2*bi+1.
    
    def expand_2x2_to_4x4(u_2x2, v_2x2, w_2x2, block_i, block_j, block_k):
        # block_i: row block of A (0 or 1)
        # block_j: col block of A / row block of B (0 or 1)
        # block_k: col block of B (0 or 1)
        
        u = [0]*16
        v = [0]*16
        w = [0]*16
        
        for idx_u, val in enumerate(u_2x2):
            if val != 0:
                i2, j2 = divmod(idx_u, 2)
                i4 = 2*block_i + i2
                j4 = 2*block_j + j2
                u[i4*4 + j4] = val
                
        for idx_v, val in enumerate(v_2x2):
            if val != 0:
                j2, k2 = divmod(idx_v, 2)
                j4 = 2*block_j + j2
                k4 = 2*block_k + k2
                v[j4*4 + k4] = val
                
        for idx_w, val in enumerate(w_2x2):
            if val != 0:
                k2, i2 = divmod(idx_w, 2)
                k4 = 2*block_k + k2
                i4 = 2*block_i + i2
                w[k4*4 + i4] = val
                
        return u, v, w

    def strassen_4x4_4x4():
        terms_2x2 = strassen_2x2_2x2()
        terms_4x4 = []
        
        # Strassen formulas for 4x4x4:
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # Here A11 is block (0,0), A22 is block (1,1), etc.
        # We sum the contributions.
        
        # M1
        u1 = [0]*16; v1 = [0]*16; w1 = [0]*16
        # A11+A22: blocks (0,0) and (1,1)
        # B11+B22: blocks (0,0) and (1,1)
        # This requires combining the 2x2 terms appropriately.
        # Actually, it's easier to just use the 7 2x2 decompositions on the 7 2x2 matrix multiplications 
        # defined by Strassen's formulas for the 4x4 case.
        # Strassen 4x4x4 is defined by 7 multiplications of 2x2 matrices.
        # Let's define the 7 2x2 multiplications:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # For each Pi, we determine which 2x2 blocks of A and B are multiplied, 
        # and how the result 2x2 matrix is added to C.
        
        # P1: A blocks (0,0) and (1,1). B blocks (0,0) and (1,1).
        # This is a sum of two 2x2 multiplications? No, it's a single 2x2 multiplication 
        # of the summed matrices.
        # So we take the 2x2 Strassen decomposition for the matrix product 
        # of (A11+A22) and (B11+B22).
        # The "A" matrix for this 2x2 product has entries that are 2x2 blocks.
        # This is getting complicated to code generically.
        
        # Alternative: Use the known fact that R(4,4,4) = 49.
        # We can generate the 49 terms by iterating over the 7 Strassen steps 
        # and for each step, using the 7 terms of the 2x2x2 Strassen decomposition.
        # Total 7 * 7 = 49 terms.
        
        # Let's define the 7 Strassen steps for 4x4x4 explicitly in terms of 2x2 blocks.
        # Each step is a 2x2x2 multiplication.
        # We need to specify:
        # 1. Which 2x2 blocks of A are used (and their signs/sums)
        # 2. Which 2x2 blocks of B are used (and their signs/sums)
        # 3. Which 2x2 blocks of C receive the result (and their signs)
        
        # This is effectively the same as the 2x2x2 Strassen, but the "scalars" are 2x2 matrices.
        # So we can just reuse the 2x2x2 Strassen terms, but expand each "scalar" 
        # into a 2x2 block.
        
        # The 7 terms from strassen_2x2_2x2() give us the structure.
        # For each term (u2, v2, w2) in the 2x2x2 Strassen:
        # u2 indicates which 2x2 blocks of the "meta-A" are used.
        # But the "meta-A" blocks are themselves sums of 4x4 blocks?
        # No, the 4x4 matrix is divided into four 2x2 blocks.
        # Let A = [[A00, A01], [A10, A11]] where Ai are 2x2.
        # Strassen formulas operate on Ai.
        # So, for each of the 7 Strassen "scalar" multiplications, 
        # we perform a 2x2x2 multiplication of the actual 2x2 matrices.
        # Wait, no. The 7 Strassen multiplications are of 2x2 matrices.
        # So for each of the 7 steps, we use the 7-rank decomposition of 2x2x2.
        # Total 49 terms.
        
        # We need to map the "meta" indices to the 4x4 indices.
        # For a given Strassen step, we know which meta-blocks (00, 01, 10, 11) are involved.
        # Let's list the 7 steps and their block indices.
        
        steps = [
            # (A_blocks, B_blocks, C_blocks)
            # A_blocks: list of (block_idx, sign)
            # B_blocks: list of (block_idx, sign)
            # C_blocks: list of (block_idx, sign)
            # block_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            
            # M1 = (A00+A11)(B00+B11)
            # C00 += M1, C11 += M1
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            
            # M2 = (A10+A11)B00
            # C10 += M2
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
            
            # M3 = A00(B01-B11)
            # C01 += M3
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
            
            # M4 = A11(B10-B00)
            # C00 += M4, C10 += M4
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            
            # M5 = (A00+A01)B11
            # C00 -= M5, C01 += M5
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            
            # M6 = (A10-A00)(B00+B01)
            # C11 += M6
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            
            # M7 = (A01-A11)(B10+B11)
            # C00 += M7
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        terms_2x2 = strassen_2x2_2x2()
        
        for a_blocks, b_blocks, c_blocks in steps:
            # For each of the 7 terms in the 2x2x2 decomposition:
            for u2, v2, w2 in terms_2x2:
                u4 = [0]*16
                v4 = [0]*16
                w4 = [0]*16
                
                # Expand u2
                # u2 has 4 entries corresponding to meta-blocks 0,1,2,3
                for meta_idx, val in enumerate(u2):
                    if val != 0:
                        # Find which block(s) in a_blocks correspond to this meta_idx?
                        # No, the meta_idx in u2 corresponds to the position in the 2x2 meta-matrix.
                        # The meta-matrix is formed by summing the actual blocks.
                        # This logic is flawed. The 2x2x2 decomposition assumes the inputs are scalars.
                        # Here the inputs are 2x2 matrices.
                        # So we just multiply the 2x2 matrices.
                        # The 7 terms of the 2x2x2 decomposition give us the linear combinations 
                        # of the elements of the 2x2 matrices.
                        
                        # Actually, it's simpler:
                        # The 4x4x4 decomposition is the Kronecker product of the 2x2x2 structure 
                        # and the 2x2x2 decomposition?
                        # No.
                        
                        # Let's just use the 49 terms generated by:
                        # For each of the 7 Strassen steps, we perform a 2x2x2 multiplication.
                        # The 2x2x2 multiplication has rank 7.
                        # So we generate 7 terms for each step.
                        # Total 49 terms.
                        
                        # We need to map the 2x2 indices of the step to 4x4 indices.
                        # The "scalar" value in the 2x2x2 decomposition is actually a 2x2 matrix product.
                        # So we expand each scalar term into 4 scalar terms (one for each element of the 2x2 result).
                        # This would give 7 * 4 = 28 terms per step? No.
                        
                        # Correct approach:
                        # The 2x2x2 Strassen decomposition has 7 terms.
                        # Each term is a rank-1 tensor.
                        # When we replace scalars with 2x2 matrices, each rank-1 tensor 
                        # becomes a set of rank-1 tensors for the 4x4 case.
                        # Specifically, if u2[i] * v2[j] * w2[k] is a term,
                        # and u2[i] represents a sum of 2x2 blocks, etc.
                        
                        # This is too complex to derive on the fly.
                        # Let's use the Schoolbook for 4x4x4 as well?
                        # Rank 16.
                        # Total rank for 4x4x8: 16 + 16 = 32.
                        # Wait, 4x4x4 schoolbook is 16.
                        # So 4x4x8 schoolbook is 32.
                        # Projected to 3x3x8, it will be less.
                        # 3x3x8 schoolbook is 72.
                        # 4x4x8 schoolbook is 32.
                        # This is a HUGE improvement.
                        
                        # Why did I think Strassen was needed?
                        # 4x4x4 schoolbook rank is 16.
                        # 3x3x3 schoolbook rank is 27.
                        # 4x4x4 is more efficient in terms of rank vs size?
                        # 16 / 64 = 0.25
                        # 27 / 27 = 1.0
                        # Yes, padding to 4x4 and using schoolbook is much better than 3x3 schoolbook.
                        
                        # So, use Schoolbook for both 4x4x4 parts.
                        # Total rank 32.
                        
                        pass # Skip Strassen implementation for now, use Schoolbook 4x4x4
                
        # Let's just use Schoolbook for 4x4x4.
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u = [0]*16; u[i*4+j] = 1
                    v = [0]*16; v[j*4+k] = 1
                    w = [0]*16; w[k*4+i] = 1
                    terms.append((u, v, w))
        return terms

    # 3. Construct 4x4x8 decomposition
    # Part 1: 4x4x4 (cols 0-3)
    # Part 2: 4x4x4 (cols 4-7)
    
    terms_4x4_4 = schoolbook_4x4_4x4()
    
    final_terms = []
    
    # Part 1: cols 0-3
    for u, v, w in terms_4x4_4:
        # v is 16 (4x4). We need 32 (4x8).
        # Map v[j*4+k] to v'[j*8+k] for k in 0..3
        v_full = [0]*32
        for idx, val in enumerate(v):
            if val != 0:
                j, k = divmod(idx, 4)
                v_full[j*8 + k] = val
        final_terms.append((u, v_full, w))
        
    # Part 2: cols 4-7
    for u, v, w in terms_4x4_4:
        # v is 16 (4x4). Map v[j*4+k] to v'[j*8+(k+4)]
        v_full = [0]*32
        for idx, val in enumerate(v):
            if val != 0:
                j, k = divmod(idx, 4)
                v_full[j*8 + (k+4)] = val
        final_terms.append((u, v_full, w))
        
    # 4. Project to 3x3x8
    U = []
    V = []
    W = []
    
    for u, v, w in final_terms:
        # u is 16 (4x4). Keep 3x3 part.
        u_proj = [0]*9
        for idx, val in enumerate(u):
            if val != 0:
                i, j = divmod(idx, 4)
                if i < 3 and j < 3:
                    u_proj[i*3 + j] = val
                    
        # v is 32 (4x8). Keep 3x8 part (rows 0-2).
        v_proj = [0]*24
        for idx, val in enumerate(v):
            if val != 0:
                j, k = divmod(idx, 8)
                if j < 3:
                    v_proj[j*8 + k] = val
                    
        # w is 32 (8x4). Keep 8x3 part (cols 0-2).
        w_proj = [0]*24
        for idx, val in enumerate(w):
            if val != 0:
                k, i = divmod(idx, 4)
                if i < 3:
                    w_proj[k*3 + i] = val
                    
        # Check if all zero
        if any(u_proj) and any(v_proj) and any(w_proj):
            U.append(u_proj)
            V.append(v_proj)
            W.append(w_proj)
            
    rank = len(U)
    
    # Convert to lists of lists
    U_mat = [[0]*rank for _ in range(9)]
    V_mat = [[0]*rank for _ in range(24)]
    W_mat = [[0]*rank for _ in range(24)]
    
    for r in range(rank):
        for i in range(9):
            U_mat[i][r] = U[r][i]
        for i in range(24):
            V_mat[i][r] = V[r][i]
        for i in range(24):
            W_mat[i][r] = W[r][i]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
