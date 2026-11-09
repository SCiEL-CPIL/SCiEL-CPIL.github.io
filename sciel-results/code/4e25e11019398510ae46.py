import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition using recursive Strassen
    rank = 49
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block indices
    # A[i, j] where i,j in 0..3
    # Block I, J in 0..1
    # Local i', j' in 0..1
    # Global i = 2*I + i', j = 2*J + j'
    
    # We will generate the 49 terms.
    # Term index r = t * 7 + s, where t is outer step (0-6), s is inner step (0-6)
    
    # Outer Strassen Linear Forms for Blocks
    # Blocks are 2x2 matrices.
    # We represent a block linear form as a 4x4 matrix of coefficients for A[i,j]
    # Or simply, for each outer step t, we define L_t and R_t as 2x2 matrices of linear forms.
    # Each linear form is a vector of length 16 (flattened 4x4).
    
    # Let's define the block linear forms explicitly.
    # A_blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    def get_block_coeffs(block_i, block_j):
        """Returns a 16-dim vector with 1s at positions corresponding to the block."""
        vec = [0] * 16
        for i in range(2):
            for j in range(2):
                gi = 2 * block_i + i
                gj = 2 * block_j + j
                vec[gi * 4 + gj] = 1
        return vec
    
    # Linear combination of blocks
    def combine_blocks(coeffs_list):
        """coeffs_list is list of (block_i, block_j, coeff). Returns 16-dim vector."""
        vec = [0] * 16
        for bi, bj, c in coeffs_list:
            block_vec = get_block_coeffs(bi, bj)
            for idx in range(16):
                vec[idx] += c * block_vec[idx]
        return vec

    # Outer Strassen Steps (t=0..6)
    # L_t: 2x2 matrix of linear forms (each is 16-dim vector for A)
    # R_t: 2x2 matrix of linear forms (each is 16-dim vector for B)
    
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B01-B11)
    # t=3: M4 = A11(B10-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    outer_L = []
    outer_R = []
    
    # t=0
    L0 = [
        [combine_blocks([(0,0,1), (1,1,1)]), [0]*16], # L00 = A00+A11, L01=0
        [[0]*16, [0]*16]
    ]
    # Wait, Strassen formulas are for the whole 2x2 matrix.
    # M1 = (A00+A11)(B00+B11). This is a block product.
    # The "L" part is the matrix A00+A11? No.
    # Strassen computes C = AB.
    # M1 = (A00+A11)(B00+B11).
    # This M1 is a 2x2 block matrix.
    # The multiplication is (Matrix A_part) * (Matrix B_part).
    # A_part = [[A00, A00], [A11, A11]]? No.
    # A_part = A00 + A11 is a 2x2 matrix.
    # B_part = B00 + B11 is a 2x2 matrix.
    # So L_t is a 2x2 matrix where each entry is a linear form.
    # For M1, L is [[A00+A11, A00+A11], [A00+A11, A00+A11]]? No.
    # The product is (A00+A11) * (B00+B11).
    # So L_t = [[A00+A11, 0], [0, A00+A11]]? No.
    # The matrix being multiplied is the 2x2 block matrix A.
    # In Strassen, we form linear combinations of the *blocks*.
    # For M1, the left factor is the block matrix A00+A11? No, A00 and A11 are 2x2 blocks.
    # A00+A11 is a 2x2 matrix.
    # So the left factor for the block multiplication is the 2x2 matrix $L = A_{00} + A_{11}$.
    # Wait, the Strassen algorithm operates on 2x2 matrices of *scalars*.
    # Here we operate on 2x2 matrices of *blocks*.
    # So $L$ is a 2x2 matrix where each entry is a block (2x2 matrix).
    # For M1: Left Factor $L$ has entries:
    # L[0,0] = A00 + A11
    # L[0,1] = A00 + A11
    # L[1,0] = A00 + A11
    # L[1,1] = A00 + A11
    # This is correct because $(A00+A11) * (B00+B11)$ is the product of two 2x2 matrices.
    
    # Let's define the 4 linear forms for each L_t (rows 0,1 and cols 0,1 of the 2x2 block matrix).
    
    def get_outer_L(t):
        if t == 0:
            # M1: (A00+A11)(B00+B11)
            val = combine_blocks([(0,0,1), (1,1,1)])
            return [[val, val], [val, val]]
        elif t == 1:
            # M2: (A10+A11)B00
            val = combine_blocks([(1,0,1), (1,1,1)])
            return [[val, val], [val, val]]
        elif t == 2:
            # M3: A00(B01-B11)
            val = combine_blocks([(0,0,1)])
            return [[val, val], [val, val]]
        elif t == 3:
            # M4: A11(B10-B00)
            val = combine_blocks([(1,1,1)])
            return [[val, val], [val, val]]
        elif t == 4:
            # M5: (A00+A01)B11
            val = combine_blocks([(0,0,1), (0,1,1)])
            return [[val, val], [val, val]]
        elif t == 5:
            # M6: (A10-A00)(B00+B01)
            val = combine_blocks([(1,0,1), (0,0,-1)])
            return [[val, val], [val, val]]
        elif t == 6:
            # M7: (A01-A11)(B10+B11)
            val = combine_blocks([(0,1,1), (1,1,-1)])
            return [[val, val], [val, val]]
        return None

    def get_outer_R(t):
        if t == 0:
            val = combine_blocks([(0,0,1), (1,1,1)])
            return [[val, val], [val, val]]
        elif t == 1:
            val = combine_blocks([(0,0,1)])
            return [[val, val], [val, val]]
        elif t == 2:
            val = combine_blocks([(0,1,1), (1,1,-1)])
            return [[val, val], [val, val]]
        elif t == 3:
            val = combine_blocks([(1,0,1), (0,0,-1)])
            return [[val, val], [val, val]]
        elif t == 4:
            val = combine_blocks([(1,1,1)])
            return [[val, val], [val, val]]
        elif t == 5:
            val = combine_blocks([(0,0,1), (0,1,1)])
            return [[val, val], [val, val]]
        elif t == 6:
            val = combine_blocks([(1,0,1), (1,1,1)])
            return [[val, val], [val, val]]
        return None

    # Inner Strassen Steps
    # For each t, we have L_t (2x2 matrix of 16-dim vectors) and R_t (2x2 matrix of 16-dim vectors).
    # We perform Strassen on these 2x2 matrices of linear forms.
    # The result is 7 products S_{t,0}..S_{t,6}.
    # Each S is a product of two linear forms (sums of A and B terms).
    
    # We need to extract the scalar coefficients for U and V.
    # U[i,j,r] is the coefficient of A[i,j] in the first factor.
    # V[j,k,r] is the coefficient of B[j,k] in the second factor.
    
    # The inner Strassen formulas for 2x2 matrices P and Q:
    # S0 = (P00+P11)(Q00+Q11)
    # S1 = (P10+P11)Q00
    # S2 = P00(Q01-Q11)
    # S3 = P11(Q10-Q00)
    # S4 = (P00+P01)Q11
    # S5 = (P10-P00)(Q00+Q01)
    # S6 = (P01-P11)(Q10+Q11)
    
    # Output accumulation:
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    
    # Note: C here is the 2x2 block result for step t.
    # The global C is assembled from these blocks.
    
    # We need to map the local C indices (0,0), (0,1), (1,0), (1,1) to global C indices.
    # Global C[i,k].
    # Block I, K. Local i', k'.
    # i = 2*I + i', k = 2*K + k'.
    
    # Outer Assembly:
    # C00 (Global Block 0,0) = M1 + M4 - M5 + M7
    # C01 (Global Block 0,1) = M3 + M5
    # C10 (Global Block 1,0) = M2 + M4
    # C11 (Global Block 1,1) = M1 - M2 + M3 + M6
    
    # We will iterate t=0..6, s=0..6.
    # For each (t,s), we determine the linear forms for U and V.
    # Then we determine which global C entries are affected and by what weight.
    
    for t in range(7):
        L = get_outer_L(t)
        R = get_outer_R(t)
        
        # L is 2x2 matrix of vectors. L[i][j] is vector for A.
        # R is 2x2 matrix of vectors. R[i][j] is vector for B.
        
        # Compute the 7 inner Strassen products
        for s in range(7):
            r = t * 7 + s
            
            # Determine U_vec and V_vec
            if s == 0:
                # (L00+L11)(R00+R11)
                U_vec = [L[0][0][idx] + L[1][1][idx] for idx in range(16)]
                V_vec = [R[0][0][idx] + R[1][1][idx] for idx in range(16)]
            elif s == 1:
                # (L10+L11)R00
                U_vec = [L[1][0][idx] + L[1][1][idx] for idx in range(16)]
                V_vec = R[0][0]
            elif s == 2:
                # L00(R01-R11)
                U_vec = L[0][0]
                V_vec = [R[0][1][idx] - R[1][1][idx] for idx in range(16)]
            elif s == 3:
                # L11(R10-R00)
                U_vec = L[1][1]
                V_vec = [R[1][0][idx] - R[0][0][idx] for idx in range(16)]
            elif s == 4:
                # (L00+L01)R11
                U_vec = [L[0][0][idx] + L[0][1][idx] for idx in range(16)]
                V_vec = R[1][1]
            elif s == 5:
                # (L10-L00)(R00+R01)
                U_vec = [L[1][0][idx] - L[0][0][idx] for idx in range(16)]
                V_vec = [R[0][0][idx] + R[0][1][idx] for idx in range(16)]
            elif s == 6:
                # (L01-L11)(R10+R11)
                U_vec = [L[0][1][idx] - L[1][1][idx] for idx in range(16)]
                V_vec = [R[1][0][idx] + R[1][1][idx] for idx in range(16)]
            
            # Fill U and V
            for idx in range(16):
                if U_vec[idx] != 0:
                    i = idx // 4
                    j = idx % 4
                    U[i * 4 + j][r] += U_vec[idx]
                if V_vec[idx] != 0:
                    j = idx // 4
                    k = idx % 4
                    V[j * 4 + k][r] += V_vec[idx]
            
            # Determine W contributions
            # We need to know which global C blocks this inner product S_{t,s} contributes to.
            # The inner product S_{t,s} contributes to the local C block of M_t.
            # Let C_local[ci, ck] be the local 2x2 block of M_t.
            # C_local[0,0] gets S0 + S3 - S4 + S6
            # C_local[0,1] gets S2 + S4
            # C_local[1,0] gets S1 + S3
            # C_local[1,1] gets S0 - S1 + S2 + S5
            
            local_contribs = {}
            if s == 0:
                local_contribs = {(0,0): 1, (1,1): 1}
            elif s == 1:
                local_contribs = {(1,0): 1, (1,1): -1}
            elif s == 2:
                local_contribs = {(0,1): 1, (1,1): 1}
            elif s == 3:
                local_contribs = {(0,0): 1, (1,0): 1}
            elif s == 4:
                local_contribs = {(0,0): -1, (0,1): 1}
            elif s == 5:
                local_contribs = {(1,1): 1}
            elif s == 6:
                local_contribs = {(0,0): 1}
            
            # Now map local block (li, lk) of M_t to Global Block (I, K)
            # And then to Global C indices.
            
            # Outer Assembly Weights for M_t:
            # M0 (t=0): C00+=1, C11+=1
            # M1 (t=1): C10+=1, C11+=-1
            # M2 (t=2): C01+=1, C11+=1
            # M3 (t=3): C00+=1, C10+=1
            # M4 (t=4): C00+=-1, C01+=1
            # M5 (t=5): C11+=1
            # M6 (t=6): C00+=1
            
            outer_weights = {
                0: {(0,0): 1, (1,1): 1},
                1: {(1,0): 1, (1,1): -1},
                2: {(0,1): 1, (1,1): 1},
                3: {(0,0): 1, (1,0): 1},
                4: {(0,0): -1, (0,1): 1},
                5: {(1,1): 1},
                6: {(0,0): 1}
            }
            
            for (li, lk), w_local in local_contribs.items():
                # li, lk are local indices within the 2x2 block matrix M_t.
                # The block M_t itself is placed in the global C according to outer_weights.
                # Wait. M_t is a 2x2 block matrix.
                # The outer formula says C_Global_Block[GI, GK] += weight * M_t.
                # So for each (GI, GK) in outer_weights[t], we add w_local * weight to C_Global[GI, GK][li, lk].
                
                for (GI, GK), w_outer in outer_weights[t].items():
                    total_w = w_local * w_outer
                    if total_w == 0: continue
                    
                    # Global C indices:
                    # Block GI, GK.
                    # Local indices li, lk.
                    # Global i = 2*GI + li
                    # Global k = 2*GK + lk
                    
                    gi = 2 * GI + li
                    gk = 2 * GK + lk
                    
                    # W index is k*n + i = gk*4 + gi
                    W[gk * 4 + gi][r] += total_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
