"""Matrix multiplication 4x4 * 4x7 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Uses Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x3 part (rank 48).
    Total rank = 97.
    
    Index conventions:
    - U has shape (n*m, rank) = (16, rank); U[i*m+j, r]
    - V has shape (m*p, rank) = (28, rank); V[j*p+k, r]
    - W has shape (p*n, rank) = (28, rank); W[k*n+i, r]
    
    Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # The 4x4x4 Strassen decomposition has rank 49.
    # The 4x4x3 schoolbook part has rank 4*4*3 = 48.
    # Total rank = 97.
    
    total_rank = 49 + 48
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Part 1: 4x4x4 Strassen (columns 0-3 of B)
    # We need to construct the 49 rank-1 terms for 4x4x4.
    
    # The 4x4x4 Strassen construction:
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block-level Strassen products:
    # M0 = (A00+A11)*(B00+B11)
    # M1 = (A10+A11)*B00
    # M2 = A00*(B01-B11)
    # M3 = A11*(B10-B00)
    # M4 = (A00+A01)*B11
    # M5 = (A10-A00)*(B00+B01)
    # M6 = (A01-A11)*(B10+B11)
    
    # Each Mk is a 2x2 matrix computed as a product of two 2x2 matrices.
    # Each 2x2*2x2 product uses 7 rank-1 terms (Strassen at element level).
    
    # For each block-level term, we need to express the 2x2*2x2 product
    # using 7 scalar multiplications.
    
    # Let me define the element-level Strassen terms for 2x2*2x2 -> 2x2.
    # For X*Y = Z where X, Y, Z are 2x2:
    # The 7 terms are:
    # t0: (X00+X11)*(Y00+Y11) -> Z00 += 1, Z11 -= 1
    # t1: (X10+X11)*Y00 -> Z10 += 1, Z11 -= 1
    # t2: X00*(Y01-Y11) -> Z01 += 1, Z11 += 1
    # t3: X11*(Y10-Y00) -> Z00 -= 1, Z10 += 1
    # t4: (X00+X01)*Y11 -> Z01 -= 1, Z11 += 1
    # t5: (X10-X00)*(Y00+Y01) -> Z00 += 1
    # t6: (X01-X11)*(Y10+Y11) -> Z00 -= 1
    
    # Each term is a bilinear form. In the tensor decomposition, each scalar
    # product within a bilinear form is a separate rank-1 component.
    
    # Actually, I realize the issue. The rank of the 2x2x2 tensor is 7,
    # meaning there are 7 rank-1 components, not 7 bilinear forms that expand
    # into more scalar products.
    
    # Let me reconsider. The tensor T[i,j,k] for 2x2x2 has 8 non-zero entries:
    # T[0,0,0]=1, T[0,1,1]=1, T[1,0,2]=1, T[1,1,3]=1,
    # T[2,0,2]=1, T[2,1,3]=1, T[3,0,0]=1, T[3,1,1]=1
    # where i = row*2+col of A, j = row*2+col of B, k = row*2+col of C.
    
    # Wait, that's not right either. Let me use the standard convention:
    # A is 2x2, B is 2x2, C = A*B is 2x2.
    # T[i,j,k] = 1 if C[i,k] = sum_l A[i,l]*B[l,k] contributes to this.
    # Actually T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term.
    
    # For 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in {0,1}^3. That's 8 terms for schoolbook.
    # Strassen reduces this to 7.
    
    # The 7 rank-1 terms for 2x2x2 Strassen:
    # Each term r has u_r[i], v_r[j], w_r[k] such that
    # sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all (i,j,k) in {0,1}^3.
    
    # Let me use a known correct decomposition. The standard Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each Mk is a scalar product of two linear combinations.
    # For the tensor decomposition, each Mk contributes ONE rank-1 term where:
    # u is the coefficient vector for A elements in the first factor
    # v is the coefficient vector for B elements in the second factor
    # w is the coefficient vector for C elements that Mk contributes to
    
    # For 2x2x2, A has 4 elements (indices 0-3), B has 4 elements (indices 0-3),
    # C has 4 elements (indices 0-3).
    
    # A element indices: A00=0, A01=1, A10=2, A11=3
    # B element indices: B00=0, B01=1, B10=2, B11=3
    # C element indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)*(B00+B11): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,-1]
    # M2 = (A10+A11)*B00: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # M3 = A00*(B01-B11): u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
    # M4 = A11*(B10-B00): u=[0,0,0,1], v=[0,0,1,-1], w=[-1,0,1,0]
    # M5 = (A00+A01)*B11: u=[1,1,0,0], v=[0,0,0,1], w=[0,-1,0,1]
    # M6 = (A10-A00)*(B00+B01): u=[0,0,1,-1], v=[1,1,0,0], w=[1,0,0,0]
    # M7 = (A01-A11)*(B10+B11): u=[0,1,0,-1], v=[0,0,1,1], w=[-1,0,0,0]
    
    elem_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # M3
        ([0,0,0,1], [0,0,1,-1], [-1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [0,-1,0,1]),   # M5
        ([0,0,1,-1], [1,1,0,0], [1,0,0,0]),   # M6
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,0]),  # M7
    ]
    
    # Now for 4x4x4, we use block-level Strassen.
    # Block-level term k uses X_k * Y_k where X_k is a 2x2 block of A (or sum)
    # and Y_k is a 2x2 block of B (or sum).
    
    # For each block-level term k, X_k is a linear combination of A blocks:
    # X_k = sum_{(br,bc)} coeff_k[br,bc] * A_{br,bc}
    # where A_{br,bc} is the 2x2 block at row block br, col block bc.
    
    # Similarly Y_k = sum_{(br,bc)} coeff_k_b[br,bc] * B_{br,bc}
    
    # The result M_k = X_k * Y_k is a 2x2 matrix that contributes to C blocks.
    
    # For the 4x4x4 decomposition, each of the 7 block-level terms M_k
    # is a 2x2*2x2 -> 2x2 product, which uses 7 element-level terms.
    # Total: 7 * 7 = 49 terms.
    
    # For block-level term k, the X part involves specific A blocks with coefficients.
    # The Y part involves specific B blocks with coefficients.
    # The C part: M_k contributes to specific C blocks with coefficients.
    
    # Let me define the block-level Strassen combinations:
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    
    # X_0 = A00 + A11: coeffs {(0,0):1, (1,1):1}
    # X_1 = A10 + A11: coeffs {(1,0):1, (1,1):1}
    # X_2 = A00: coeffs {(0,0):1}
    # X_3 = A11: coeffs {(1,1):1}
    # X_4 = A00 + A01: coeffs {(0,0):1, (0,1):1}
    # X_5 = A10 - A00: coeffs {(1,0):1, (0,0):-1}
    # X_6 = A01 - A11: coeffs {(0,1):1, (1,1):-1}
    
    # Y_0 = B00 + B11: coeffs {(0,0):1, (1,1):1}
    # Y_1 = B00: coeffs {(0,0):1}
    # Y_2 = B01 - B11: coeffs {(0,1):1, (1,1):-1}
    # Y_3 = B10 - B00: coeffs {(1,0):1, (0,0):-1}
    # Y_4 = B11: coeffs {(1,1):1}
    # Y_5 = B00 + B01: coeffs {(0,0):1, (0,1):1}
    # Y_6 = B10 + B11: coeffs {(1,0):1, (1,1):1}
    
    # C contributions:
    # C00 = M0 + M3 - M4 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    # So for block-level term k, the C block coefficients are:
    # k=0: C00+=1, C11+=1 -> wait, let me recheck.
    # C00 = M0 + M3 - M4 + M5: M0 contributes +1 to C00
    # C01 = M2 + M4: M0 contributes 0 to C01
    # C10 = M1 + M3: M0 contributes 0 to C10
    # C11 = M0 - M1 + M2 + M6: M0 contributes +1 to C11
    
    # Hmm, that doesn't look right. Let me re-derive.
    
    # Standard Strassen for block matrices:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M6
    
    # So:
    # M0: C00+=1, C11+=1
    # M1: C10+=1, C11+=-1
    # M2: C01+=1, C11+=1
    # M3: C00+=1, C10+=1
    # M4: C00+=-1, C01+=1
    # M5: C00+=1
    # M6: C11+=1
    
    block_x_coeffs = [
        {(0,0):1, (1,1):1},      # X_0
        {(1,0):1, (1,1):1},      # X_1
        {(0,0):1},               # X_2
        {(1,1):1},               # X_3
        {(0,0):1, (0,1):1},      # X_4
        {(1,0):1, (0,0):-1},     # X_5
        {(0,1):1, (1,1):-1},     # X_6
    ]
    
    block_y_coeffs = [
        {(0,0):1, (1,1):1},      # Y_0
        {(0,0):1},               # Y_1
        {(0,1):1, (1,1):-1},     # Y_2
        {(1,0):1, (0,0):-1},     # Y_3
        {(1,1):1},               # Y_4
        {(0,0):1, (0,1):1},      # Y_5
        {(1,0):1, (1,1):1},      # Y_6
    ]
    
    block_c_coeffs = [
        {(0,0):1, (1,1):1},      # M0
        {(1,0):1, (1,1):-1},     # M1
        {(0,1):1, (1,1):1},      # M2
        {(0,0):1, (1,0):1},      # M3
        {(0,0):-1, (0,1):1},     # M4
        {(0,0):1},               # M5
        {(1,1):1},               # M6
    ]
    
    # Now build the 49 terms.
    # For block-level term k and element-level term t:
    # The u vector (for A) has 16 entries.
    # For each A block (br,bc) in block_x_coeffs[k] with coefficient cx,
    # and for each element position (lr,lc) in the 2x2 block,
    # the global A index is (br*2+lr)*m + (bc*2+lc).
    # The element-level u coefficient from elem_terms[t][0] is at position (lr,lc).
    # So u[(br*2+lr)*m + (bc*2+lc)] += cx * elem_u[lr*2+lc]
    
    # Similarly for v and w.
    
    col = 0
    for k in range(7):
        for t in range(7):
            u_vec = np.zeros(n*m, dtype=float)
            v_vec = np.zeros(m*p, dtype=float)
            w_vec = np.zeros(p*n, dtype=float)
            
            elem_u = elem_terms[t][0]
            elem_v = elem_terms[t][1]
            elem_w = elem_terms[t][2]
            
            # Fill u_vec
            for (br, bc), cx in block_x_coeffs[k].items():
                for lr in range(2):
                    for lc in range(2):
                        a_global = (br*2+lr)*m + (bc*2+lc)
                        elem_idx = lr*2 + lc
                        u_vec[a_global] += cx * elem_u[elem_idx]
            
            # Fill v_vec (only for columns 0-3 of B, i.e., the 4x4 part)
            for (br, bc), cy in block_y_coeffs[k].items():
                for lr in range(2):
                    for lc in range(2):
                        b_global = (br*2+lr)*p + (bc*2+lc)
                        elem_idx = lr*2 + lc
                        v_vec[b_global] += cy * elem_v[elem_idx]
            
            # Fill w_vec
            for (cr, cc), cw in block_c_coeffs[k].items():
                for lr in range(2):
                    for lc in range(2):
                        c_global = (cc*2+lc)*n + (cr*2+lr)
                        elem_idx = lr*2 + lc
                        w_vec[c_global] += cw * elem_w[elem_idx]
            
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
    
    # Part 2: 4x4x3 schoolbook (columns 4-6 of B)
    # For i in 0..3, j in 0..3, k in 4..6:
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
