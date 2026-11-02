"""Tensor decomposition for 4x4 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Try to find a low-rank decomposition using rank-2 block Winograd
    # This is the 2x2 Winograd scheme applied to 4x4 blocks
    # Actually, let's use the known rank-56 decomposition for 4x4
    
    # The 2x2 Winograd algorithm for matrix multiplication:
    # For C = A*B where A, B are 2x2:
    # m1 = (a11+a21)(b11-b21)
    # m2 = (a11+a21)(b21+b22)
    # m3 = (a11-a21+a22)(b11)
    # m4 = (a11-a21+a22)(b22)
    # m5 = (a22-a21)(b11+b21)
    # m6 = (a22-a21)(b12+b22)
    # m7 = (a11+a22)(b12-b22)
    # m8 = (a12+a22)(b12+b22)
    # Then:
    # c11 = m1 + m2 + m3 + m4 - m5 - m7
    # c12 = m2 - m4 + m6 + m7
    # c21 = -m1 + m3 - m5 + m8
    # c22 = -m1 - m3 + m4 + m6 - m8
    
    # For 4x4, we can partition into 2x2 blocks. But the blocks interact.
    # Actually, a simpler approach: use the rank-56 decomposition from 
    # Strassen's generalization or the 2x2 block Winograd.
    
    # Let me try a different approach: use the known fact that the rank of 
    # 4x4 matrix multiplication is at most 56 (from block Winograd/Strassen).
    # 
    # Actually, let me just implement the rank-56 decomposition properly.
    # The 4x4 matrix multiplication can be done with 56 multiplications 
    # using a 2x2 block Strassen approach where each 2x2 block multiplication
    # uses 7 multiplications (Winograd/Strassen for 2x2).
    #
    # For 4x4 = [A11 A12; A21 A22] * [B11 B12; B21 B22]
    # where each block is 2x2:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Using Strassen for each 2x2 block product (7 mults each), but we need 
    # to combine them. Actually, this gives 4 * 7 = 28 mults for the block 
    # products, but we need additions. The total rank would be higher.
    #
    # Let me instead use a verified approach: the rank-64 schoolbook is safe.
    # But the task asks for rank-minimal. Let me try rank 56.
    
    # Actually, I recall that the best known upper bound for 4x4 is 49 from 
    # Coppersmith, but implementing that correctly is complex.
    
    # Let me use a safer intermediate: rank 56 via 2x2 block Winograd.
    # Each 2x2 multiplication uses 7 products. For 4x4, we have 4 blocks in A
    # and 4 blocks in B, but the structure allows sharing.
    
    # Actually, the simplest correct improvement is to use the rank-56 
    # decomposition from the 2x2 Winograd applied to 2x2 sub-blocks.
    
    # Let me construct it explicitly for 4x4.
    # Index 4x4 matrix as M[i,j] where i,j in {0,1,2,3}
    # Partition: A11 = M[0:2,0:2], A12 = M[0:2,2:4], A21 = M[2:4,0:2], A22 = M[2:4,2:4]
    
    # Winograd 2x2: for X*Y where X,Y are 2x2:
    # p1 = (x00+x10)*(y00-y10)
    # p2 = (x00+x10)*(y10+y11)
    # p3 = (x00-x10+x11)*(y00)
    # p4 = (x00-x10+x11)*(y11)
    # p5 = (x11-x10)*(y00+y10)
    # p6 = (x11-x10)*(y01+y11)
    # p7 = (x00+x11)*(y01-y11)
    # p8 = (x01+x11)*(y01+y11)
    # Result:
    # r00 = p1+p2+p3+p4-p5-p7
    # r01 = p2-p4+p6+p7
    # r10 = -p1+p3-p5+p8
    # r11 = -p1-p3+p4+p6-p8
    
    # Wait, that's 8 products, not 7. Let me recheck Winograd 2x2.
    # The standard Winograd 2x2 uses 7 products:
    # m1 = (a00+a10)*(b00-b10)
    # m2 = (a00+a10)*(b10+b11)
    # m3 = (a00-a10+a11)*(b00)
    # m4 = (a00-a10+a11)*(b11)
    # m5 = (a11-a10)*(b00+b10)
    # m6 = (a11-a10)*(b01+b11)
    # m7 = (a00+a11)*(b01-b11)
    # Then:
    # c00 = m1 + m2 + m3 + m4 - m5 - m7
    # c01 = m2 - m4 + m6 + m7
    # c10 = -m1 + m3 - m5 + (a01+a11)*(b01+b11)  # wait, this needs an 8th
    # 
    # Actually I think the standard 2x2 Strassen uses 7 mults:
    # m1 = (a00+a10)(b00+b10)
    # m2 = (a10+a11)(b00)
    # m3 = (a00+a01)(b10)
    # m4 = (a11-a10-a01)(b11)
    # m5 = (a00+a11)(b01-b11)
    # m6 = (a01+a11)(b00-b01)
    # m7 = (a00+a10)(b10-b00)
    # c00 = m1 - m2 + m4 + m5
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m6 - m7
    
    # Let me verify: this is the standard Strassen 2x2. 7 multiplications.
    
    # For 4x4, partition into 2x2 blocks:
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    #
    # Each block product (e.g., A11*B11) can be done with 7 mults via Strassen.
    # But the sums like A11*B11 + A12*B21 require combining results.
    #
    # Total: 8 block products * 7 = 56 multiplications.
    # This gives rank 56.
    
    # Let me construct the U, V, W matrices for rank 56.
    # Each of the 8 block products has 7 "scalar" multiplications.
    # Total rank = 56.
    
    # Block indices: (bi, bj) where bi, bj in {0,1} for the 2x2 block grid
    # Within each 2x2 block, use Strassen's 7 formulas.
    
    # Strassen 2x2 formulas (inputs are 2x2 blocks, each element is a scalar):
    # For X = [x00 x01; x10 x11], Y = [y00 y01; y10 y11]:
    # m1 = (x00+x10)(y00+y10)
    # m2 = (x10+x11)(y00)
    # m3 = (x00+x01)(y10)
    # m4 = (x11-x10-x01)(y11)
    # m5 = (x00+x11)(y01-y11)
    # m6 = (x01+x11)(y00-y01)
    # m7 = (x00+x10)(y10-y00)
    # Result Z:
    # z00 = m1 - m2 + m4 + m5
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m3 + m6 - m7
    
    # Now for 4x4:
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), 
    #            A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #            B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    
    # C blocks:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each block product, use Strassen. Let me enumerate all 8 block products.
    
    # Block product (ai, bi) -> contributes to C block (ci)
    # A11*B11 -> C11
    # A12*B21 -> C11
    # A11*B12 -> C12
    # A12*B22 -> C12
    # A21*B11 -> C21
    # A22*B21 -> C21
    # A21*B12 -> C22
    # A22*B22 -> C22
    
    # Let's define the 8 block products with their (A_block, B_block) pairs:
    # bp0: A11 * B11
    # bp1: A12 * B21
    # bp2: A11 * B12
    # bp3: A12 * B22
    # bp4: A21 * B11
    # bp5: A22 * B21
    # bp6: A21 * B12
    # bp7: A22 * B22
    
    # For each block product, Strassen uses 7 mults.
    # Total: 8 * 7 = 56 mults.
    
    # Now I need to build U, V, W.
    # U has shape (n*m, rank) = (16, 56)
    # V has shape (m*p, rank) = (16, 56)
    # W has shape (p*n, rank) = (16, 56)
    
    # The contraction is: C[k*n+i, j*m+l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So the bilinear form is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And this should equal C[i,j,k,l] = (A*B)_{i,l} where A is (i,j), B is (j,k), result is (i,k)
    # Wait, matrix mult: C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k,l] where... hmm.
    
    # Actually the standard matrix multiplication tensor:
    # T[a, b, c, d] = delta_{b,c} * delta_{a,d} ... no.
    # The tensor M such that C_{i,k} = sum_{j,l} A_{i,j} B_{j,k} ... 
    # Actually, the matrix multiplication tensor is defined as:
    # (A \boxtimes B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a 4-tensor: M[i, j, k, l] = delta_{j,k} * delta_{i,l}? No.
    # Let me think again. The bilinear form is:
    # <M, A \otimes B> = trace(A^T B C) ... no.
    # 
    # The matrix multiplication tensor T of order 4: T[i,j,k,l] = 1 if j=k and i... 
    # Actually: C_{i,l} = sum_j A_{i,j} B_{j,l}
    # So the tensor is: T[i, j, j, l] = 1 (summing over j)
    # As a 4-index tensor: T[i, j, k, l] = delta_{j,k} (for the "middle" indices)
    # But wait, that's not right either. Let me be precise.
    
    # The matrix multiplication map: (A, B) -> A*B
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So as a multilinear map: M(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T[i, j, k, l] where A is indexed (i,j), B is indexed (j,k), 
    # and the result C is indexed (i,k). So:
    # T[i, j, k, l] = delta_{j,k} * delta_{i,l}? 
    # Let's check: sum_{j,k} T[i,j,k,l] A[i,j] B[k,l] ... 
    # If T[i,j,k,l] = delta_{j,k}, then sum_{j,k} delta_{j,k} A[i,j] B[k,l] = sum_j A[i,j] B[j,l] = C[i,l]. 
    # Yes! So T[i,j,k,l] = delta_{j,k}.
    
    # The decomposition: T[i,j,k,l] = sum_r U[i,j,r] V[k,l,r] W[?,r]
    # Wait, the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And the convention is W[k*n+i, r].
    # 
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U is indexed by (i,j) -> i*m+j, V by (j,k) -> j*p+k, W by (k,i) -> k*n+i.
    # The bilinear form: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal T[i,j,k,?] ... 
    # 
    # Actually, I think the convention is that the tensor element is:
    # T[i,j,k,l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[l*n+i, c]
    # where l is the row of the output... 
    # 
    # Let me just match the schoolbook: in the schoolbook, 
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # for each (i,j,k). So the bilinear form gives:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if (i,j,k) matches, else 0.
    # And this equals delta_{j,j} ... it's just 1 for each (i,j,k).
    # But T[i,j,k,l] should be delta_{j,k} (with l being the output column).
    # 
    # I think the W index k*n+i means: W is indexed by (output_row=i, something=k) -> k*n+i.
    # So the full expression is: for output (i, l), 
    # C[i,l] = sum_{j,k} T[i,j,k,l] A[i,j] B[k,l] ... 
    # 
    # Let me just trust the schoolbook convention and mirror it.
    # In schoolbook: for each (i,j,k), we have one term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The "tensor element" this contributes to is at position... 
    # It contributes to the (i, j, k, i) position? Or (i, j, k, ?)
    # 
    # I think the convention is: the r-th rank-1 term contributes U[r] ⊗ V[r] ⊗ W[r] 
    # to the 4-tensor, where U is a vector in R^{n*m}, V in R^{m*p}, W in R^{p*n}.
    # The 4-tensor T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[l*p+n... 
    # 
    # OK this is getting confusing. Let me just directly verify with numpy.
    
    # Let me build the 56-rank decomposition and verify it.
    
    rank = 56
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Strassen 2x2 coefficients
    # For X*Y -> Z:
    # m1 = (x00+x10)(y00+y10)
    # m2 = (x10+x11)(y00)
    # m3 = (x00+x01)(y10)
    # m4 = (x11-x10-x01)(y11)
    # m5 = (x00+x11)(y01-y11)
    # m6 = (x01+x11)(y00-y01)
    # m7 = (x00+x10)(y10-y00)
    # z00 = m1 - m2 + m4 + m5
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m3 + m6 - m7
    
    # Strassen multipliers:
    # (u_factor, v_factor, w_contribs)
    # u_factor: 2x2 matrix for X (the left block)
    # v_factor: 2x2 matrix for Y (the right block)
    # w_contribs: dict mapping output position (z00,z01,z10,z11) to coefficient
    
    # Let me encode each of the 7 Strassen multiplications:
    # For m1: X = [1,0; 1,0] * [1,0; 1,0] ... 
    # m1 = (x00+x10)*(y00+y10)
    # X factor: x00 + x10 -> vector [1, 0, 1, 0] (in order x00,x01,x10,x11)
    # Y factor: y00 + y10 -> vector [1, 0, 1, 0] (in order y00,y01,y10,y11)
    # Contribution to z00: +1
    
    # Let me define the 2x2 block element ordering: (0,0), (0,1), (1,0), (1,1)
    # So for a 2x2 block X: x[0]=X00, x[1]=X01, x[2]=X10, x[3]=X11
    
    strassen = [
        # (X_coef, Y_coef, Z_contribs)
        # X_coef: [x00, x01, x10, x11]
        # Y_coef: [y00, y01, y10, y11]
        # Z_contribs: {z00: c, z01: c, z10: c, z11: c}
        ([1,0,1,0], [1,0,1,0], {0: 1, 3: 1}),      # m1
        ([0,0,1,1], [1,0,0,0], {2: 1}),              # m2
        ([1,1,0,0], [0,0,1,0], {1: 1}),              # m3
        ([-1,-1,0,1], [0,0,0,1], {0: 1, 2: 1}),      # m4
        ([1,0,0,1], [0,1,0,-1], {0: 1, 1: 1}),       # m5
        ([0,1,0,1], [1,-1,0,0], {3: 1}),             # m6
        ([1,0,1,0], [0,0,1,-1], {3: -1}),            # m7
    ]
    
    # Now, for 4x4, the 8 block products:
    # Each block product uses a 2x2 sub-block of A and a 2x2 sub-block of B.
    # 
    # A block (ai, aj) where ai, aj in {0,1}:
    # ai=0,aj=0: rows 0-1, cols 0-1 -> A elements at (0,0),(0,1),(1,0),(1,1)
    # ai=0,aj=1: rows 0-1, cols 2-3 -> A elements at (0,2),(0,3),(1,2),(1,3)
    # ai=1,aj=0: rows 2-3, cols 0-1 -> A elements at (2,0),(2,1),(3,0),(3,1)
    # ai=1,aj=1: rows 2-3, cols 2-3 -> A elements at (2,2),(2,3),(3,2),(3,3)
    
    # B block (bi, bj) where bi, bj in {0,1}:
    # bi=0,bj=0: rows 0-1, cols 0-1 -> B elements at (0,0),(0,1),(1,0),(1,1)
    # bi=0,bj=1: rows 0-1, cols 2-3 -> B elements at (0,2),(0,3),(1,2),(1,3)
    # bi=1,bj=0: rows 2-3, cols 0-1 -> B elements at (2,0),(2,1),(3,0),(3,1)
    # bi=1,bj=1: rows 2-3, cols 2-3 -> B elements at (2,2),(2,3),(3,2),(3,3)
    
    # C block (ci, cj) where ci, cj in {0,1}:
    # ci=0,cj=0: rows 0-1, cols 0-1 -> C elements at (0,0),(0,1),(1,0),(1,1)
    # ci=0,cj=1: rows 0-1, cols 2-3 -> C elements at (0,2),(0,3),(1,2),(1,3)
    # ci=1,cj=0: rows 2-3, cols 0-1 -> C elements at (2,0),(2,1),(3,0),(3,1)
    # ci=1,cj=1: rows 2-3, cols 2-3 -> C elements at (2,2),(2,3),(3,2),(3,3)
    
    # The 8 block products and their target C blocks:
    # bp(ai,aj,bi,bj) -> C block (ci,cj)
    # bp(0,0, 0,0) -> C(0,0): A11*B11
    # bp(0,1, 1,0) -> C(0,0): A12*B21
    # bp(0,0, 0,1) -> C(0,1): A11*B12
    # bp(0,1, 1,1) -> C(0,1): A12*B22
    # bp(1,0, 0,0) -> C(1,0): A21*B11
    # bp(1,1, 1,0) -> C(1,0): A22*B21
    # bp(1,0, 0,1) -> C(1,1): A21*B12
    # bp(1,1, 1,1) -> C(1,1): A22*B22
    
    block_products = [
        ((0,0), (0,0), (0,0)),  # A11*B11 -> C11
        ((0,1), (1,0), (0,0)),  # A12*B21 -> C11
        ((0,0), (0,1), (0,1)),  # A11*B12 -> C12
        ((0,1), (1,1), (0,1)),  # A12*B22 -> C12
        ((1,0), (0,0), (1,0)),  # A21*B11 -> C21
        ((1,1), (1,0), (1,0)),  # A22*B21 -> C21
        ((1,0), (0,1), (1,1)),  # A21*B12 -> C22
        ((1,1), (1,1), (1,1)),  # A22*B22 -> C22
    ]
    
    col = 0
    for (ai, aj), (bi, bj), (ci, cj) in block_products:
        for s_idx in range(7):
            X_coef, Y_coef, Z_contribs = strassen[s_idx]
            
            # X is A block (ai, aj): rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
            # X elements: X00=A[2*ai, 2*aj], X01=A[2*ai, 2*aj+1], X10=A[2*ai+1, 2*aj], X11=A[2*ai+1, 2*aj+1]
            for dx in range(2):
                for dy in range(2):
                    x_idx = dx * 2 + dy  # 0,1,2,3
                    a_row = 2*ai + dx
                    a_col = 2*aj + dy
                    U[a_row * 4 + a_col, col] += X_coef[x_idx]
            
            # Y is B block (bi, bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
            # Y elements: Y00=B[2*bi, 2*bj], Y01=B[2*bi, 2*bj+1], Y10=B[2*bi+1, 2*bj], Y11=B[2*bi+1, 2*bj+1]
            for dx in range(2):
                for dy in range(2):
                    y_idx = dx * 2 + dy
                    b_row = 2*bi + dx
                    b_col = 2*bj + dy
                    V[b_row * 4 + b_col, col] += Y_coef[y_idx]
            
            # Z contributes to C block (ci, cj): 
            # z00 -> C[2*ci, 2*cj], z01 -> C[2*ci, 2*cj+1], z10 -> C[2*ci+1, 2*cj], z11 -> C[2*ci+1, 2*cj+1]
            for z_pos, coef in Z_contribs.items():
                dz_row = z_pos // 2
                dz_col = z_pos % 2
                c_row = 2*ci + dz_row
                c_col = 2*cj + dz_col
                # W is indexed by (k, i) -> k*n+i where k is output col? 
                # In schoolbook: W[k*n+i] where i is row, k is col.
                # So W index = c_col * 4 + c_row
                W[c_col * 4 + c_row, col] += coef
            
            col += 1
    
    # Verify the decomposition
    # Build the tensor and check
    A_test = np.random.rand(4, 4)
    B_test = np.random.rand(4, 4)
    C_expected = A_test @ B_test
    
    # The bilinear form: C[i,l] = sum_{j,k} sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # Wait, V is indexed by (j*p+k) = (j*4+k) where j is A's col and k is B's row.
    # But in matrix mult, A[i,j] * B[j,k], so j is shared.
    # U[i*m+j] = U[i*4+j], V[j*p+k] = V[j*4+k], W[k*n+i] = W[k*4+i]
    # So: C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # But this sums over j AND r, and k is the output column.
    # Wait, that doesn't look right. Let me re-examine.
    
    # In the schoolbook: for each (i,j,k), U[i*4+j]=1, V[j*4+k]=1, W[k*4+i]=1.
    # So the bilinear form for output (i, k): sum_j sum_r U[i*4+j,r] V[j*4+k,r] W[k*4+i,r]
    # = sum_j (1 if the (i,j,k) term exists) = sum_j 1 = ... 
    # For fixed (i,k), the sum over j gives sum_j A[i,j] B[j,k] = C[i,k]. Correct!
    
    C_computed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            s = 0
            for j in range(4):
                for r in range(rank):
                    s += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
            C_computed[i, k] = s
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fall back to schoolbook rank 64
        rank = 64
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
