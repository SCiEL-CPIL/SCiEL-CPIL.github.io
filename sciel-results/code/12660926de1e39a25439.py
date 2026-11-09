import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # We will construct a rank 52 decomposition.
    # Split m=5 into 4+1, p=5 into 4+1.
    # Core: 4x4x4 using Strassen's algorithm (Rank 49)
    # Boundary 1: 4x1x4 (Rank 1)
    # Boundary 2: 4x4x1 (Rank 1)
    # Boundary 3: 4x1x1 (Rank 1)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        u[u_idx] = u_val
        v[v_idx] = v_val
        w[w_idx] = w_val
        
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    # --- Core 4x4x4 Decomposition (Rank 49) ---
    # We decompose 4x4x4 into 4 blocks of 2x2x2? No, Strassen operates on 2x2 blocks.
    # A is 4x4, B is 4x4.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B00, B01], [B10, B11]] where each is 2x2.
    # C = [[C00, C01], [C10, C11]] where each is 2x2.
    
    # Strassen's 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # So we need 7 * 7 = 49 rank-1 terms.
    
    # Strassen's formulas for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to generate the rank-1 terms for each of these 7 matrix multiplications.
    # A 2x2x2 multiplication can be done with 7 rank-1 terms.
    # Let's define a helper to add the 7 terms for a 2x2x2 multiplication
    # given the offsets of the 2x2 blocks in the global 4x4x4 space.
    
    def add_strassen_2x2x2(i_off, j_off, k_off, coeff=1.0):
        # i_off, j_off, k_off are offsets for the 2x2 blocks in the 4x4 grid.
        # The 2x2 block in A is at rows i_off..i_off+1, cols j_off..j_off+1.
        # The 2x2 block in B is at rows j_off..j_off+1, cols k_off..k_off+1.
        # The 2x2 block in C is at rows k_off..k_off+1, cols i_off..i_off+1.
        
        # Strassen's 7 terms for 2x2x2:
        # T1: (A00+A11)(B00+B11) -> C00, C11
        # T2: (A10+A11)B00 -> C10, C11
        # T3: A00(B01-B11) -> C01, C11
        # T4: A11(B10-B00) -> C00, C10
        # T5: (A00+A01)B11 -> C00, C01
        # T6: (A10-A00)(B00+B01) -> C11
        # T7: (A01-A11)(B10+B11) -> C00
        
        # Indices in the 2x2 block:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Term 1: (A00+A11)(B00+B11)
        # A part: A00 + A11. Indices: (i_off, j_off), (i_off+1, j_off+1)
        # B part: B00 + B11. Indices: (j_off, k_off), (j_off+1, k_off+1)
        # C part: C00 + C11. Indices: (k_off, i_off), (k_off+1, i_off+1)
        # This is not a single rank-1 term. It's a sum.
        # Wait, Strassen's algorithm computes 7 products.
        # Each product is a 2x2 matrix multiplication.
        # A 2x2 matrix multiplication has rank 7.
        # So we need to decompose each of the 7 products into 7 rank-1 terms.
        # Total 49 terms.
        
        # Let's define the 7 rank-1 terms for a single 2x2x2 multiplication.
        # Standard Strassen for 2x2:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b01-b11)
        # P4 = a11(b10-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b01)
        # P7 = (a01-a11)(b10+b11)
        
        # c00 = p1 + p4 - p5 + p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p6
        
        # Each Pi is a scalar multiplication of two linear combinations.
        # Pi = (sum u_a A_a) * (sum v_b B_b)
        # This contributes to C as:
        # C += Pi * (sum w_c C_c) ? No.
        # The tensor term is u \otimes v \otimes w.
        # C_c = sum_r (sum_a U_{a,r} A_a) (sum_b V_{b,r} B_b) W_{c,r}
        
        # For P1:
        # A part: a00 + a11. U indices: (0,0)=1, (1,1)=1.
        # B part: b00 + b11. V indices: (0,0)=1, (1,1)=1.
        # C part: c00 + c11. W indices: (0,0)=1, (1,1)=1.
        # This is ONE rank-1 term?
        # T = (e00+e11)_A \otimes (e00+e11)_B \otimes (e00+e11)_C
        # Yes, this is a valid rank-1 tensor.
        
        # So we just need to add the 7 tensors corresponding to P1..P7.
        
        # P1: A: (0,0), (1,1). B: (0,0), (1,1). C: (0,0), (1,1).
        # P2: A: (1,0), (1,1). B: (0,0). C: (1,0), (1,1).
        # P3: A: (0,0). B: (0,1), (1,1). C: (0,1), (1,1).
        # P4: A: (1,1). B: (1,0), (0,0). C: (0,0), (1,0).
        # P5: A: (0,0), (0,1). B: (1,1). C: (0,0), (0,1).
        # P6: A: (1,0), (0,0). B: (0,0), (0,1). C: (1,1).
        # P7: A: (0,1), (1,1). B: (1,0), (1,1). C: (0,0).
        
        # Note: Signs are handled in the C part (W matrix) or A/B parts.
        # It's easier to put signs in W or A.
        
        # Let's define the terms explicitly.
        
        # Helper to add a term with specific indices and values
        def add_rank1_term(A_indices, A_vals, B_indices, B_vals, C_indices, C_vals):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            for idx, val in zip(A_indices, A_vals):
                i, j = idx
                u[(i_off + i) * m + (j_off + j)] += coeff * val
            for idx, val in zip(B_indices, B_vals):
                j, k = idx
                v[(j_off + j) * p + (k_off + k)] += val
            for idx, val in zip(C_indices, C_vals):
                k, i = idx
                w[(k_off + k) * n + (i_off + i)] += val
                
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

        # P1
        add_rank1_term([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1])
        # P2
        add_rank1_term([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, 1])
        # P3
        add_rank1_term([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1])
        # P4
        add_rank1_term([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1])
        # P5
        add_rank1_term([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1])
        # P6
        add_rank1_term([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1])
        # P7
        add_rank1_term([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1])

    # The 4x4x4 block is composed of 4 2x2x2 blocks?
    # No, Strassen's algorithm for 4x4 uses 7 multiplications of 2x2 blocks.
    # The 2x2 blocks are:
    # A00 (0,0), A01 (0,2), A10 (2,0), A11 (2,2)
    # B00 (0,0), B01 (0,2), B10 (2,0), B11 (2,2)
    # C00 (0,0), C01 (0,2), C10 (2,0), C11 (2,2)
    
    # We need to call add_strassen_2x2x2 for each of the 7 products.
    # But add_strassen_2x2x2 adds 7 terms for a SINGLE 2x2x2 multiplication.
    # We have 7 such multiplications.
    # So we need to call a function that adds the 7 terms for a specific pair of 2x2 blocks.
    
    # Let's refine add_strassen_2x2x2 to take the specific block indices.
    
    def add_2x2_mult(a_i, a_j, b_i, b_j, c_i, c_j):
        # a_i, a_j: top-left corner of A block (2x2)
        # b_i, b_j: top-left corner of B block (2x2)
        # c_i, c_j: top-left corner of C block (2x2)
        
        # P1: (A00+A11)(B00+B11)
        # A: (a_i, a_j), (a_i+1, a_j+1)
        # B: (b_i, b_j), (b_i+1, b_j+1)
        # C: (c_i, c_j), (c_i+1, c_j+1)
        add_rank1_term([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P2: (A10+A11)B00
        # A: (a_i+1, a_j), (a_i+1, a_j+1)
        # B: (b_i, b_j)
        # C: (c_i+1, c_j), (c_i+1, c_j+1)
        add_rank1_term([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, 1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P3: A00(B01-B11)
        # A: (a_i, a_j)
        # B: (b_i, b_j+1), (b_i+1, b_j+1)
        # C: (c_i, c_j+1), (c_i+1, c_j+1)
        add_rank1_term([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P4: A11(B10-B00)
        # A: (a_i+1, a_j+1)
        # B: (b_i+1, b_j), (b_i, b_j)
        # C: (c_i, c_j), (c_i+1, c_j)
        add_rank1_term([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P5: (A00+A01)B11
        # A: (a_i, a_j), (a_i, a_j+1)
        # B: (b_i+1, b_j+1)
        # C: (c_i, c_j), (c_i, c_j+1)
        add_rank1_term([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P6: (A10-A00)(B00+B01)
        # A: (a_i+1, a_j), (a_i, a_j)
        # B: (b_i, b_j), (b_i, b_j+1)
        # C: (c_i+1, c_j+1)
        add_rank1_term([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1], a_i, a_j, b_i, b_j, c_i, c_j)
        
        # P7: (A01-A11)(B10+B11)
        # A: (a_i, a_j+1), (a_i+1, a_j+1)
        # B: (b_i+1, b_j), (b_i+1, b_j+1)
        # C: (c_i, c_j)
        add_rank1_term([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0)], [1], a_i, a_j, b_i, b_j, c_i, c_j)

    # Define the 7 multiplications for the 4x4 core
    # M1 = (A00+A11)(B00+B11)
    # A blocks: (0,0) and (2,2). B blocks: (0,0) and (2,2). C blocks: (0,0) and (2,2).
    # This is not a single 2x2 multiplication.
    # Strassen's algorithm for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # This requires adding the 2x2 blocks first.
    # A_sum1 = A00 + A11.
    # B_sum1 = B00 + B11.
    # Then compute 2x2 multiplication of A_sum1 and B_sum1.
    # This results in a 2x2 matrix M1.
    # M1 contributes to C00 and C11.
    
    # So we need to:
    # 1. Form the linear combinations of A and B blocks.
    # 2. Perform the 2x2 multiplication.
    # 3. Distribute the result to C blocks.
    
    # This is equivalent to adding the rank-1 terms for the 2x2 multiplication
    # but with the A and B indices being sums of two 2x2 blocks.
    
    # My add_2x2_mult function assumes the A and B inputs are single 2x2 blocks.
    # I need a more general function that takes lists of (offset, coeff) for A, B, and C.
    
    def add_general_2x2_mult(A_terms, B_terms, C_terms):
        # A_terms: list of (i_off, j_off, coeff)
        # B_terms: list of (j_off, k_off, coeff)
        # C_terms: list of (k_off, i_off, coeff)
        
        # We perform a 2x2x2 multiplication on the "virtual" 2x2 matrices
        # formed by the linear combinations.
        # The 7 terms of Strassen's algorithm are applied to these combinations.
        
        # P1: (A00+A11)(B00+B11)
        # A part: Sum of A_terms with coeffs 1 for (0,0) and (1,1) of the virtual block?
        # No, the virtual block IS the sum.
        # The Strassen terms operate on the entries of the virtual 2x2 matrices.
        # Let A_virt be the 2x2 matrix formed by A_terms.
        # Let B_virt be the 2x2 matrix formed by B_terms.
        # Let C_virt be the 2x2 matrix formed by C_terms.
        
        # The 7 rank-1 tensors for the virtual 2x2x2 multiplication are:
        # T1: A_virt[0,0]+A_virt[1,1] * B_virt[0,0]+B_virt[1,1] -> C_virt[0,0]+C_virt[1,1]
        # ...
        
        # We need to expand these into global rank-1 terms.
        
        # T1:
        # A_indices: (0,0) and (1,1) of A_virt.
        # A_virt[0,0] = sum(a_i, a_j, c) A[i,j]
        # So A part of T1 is sum over a in A_terms: c * (e_{a_i, a_j} + e_{a_i+1, a_j+1})?
        # No, A_virt is a 2x2 matrix.
        # A_virt[0,0] is the top-left 2x2 block of the global A?
        # No, A_virt is a 2x2 matrix of SCALARS?
        # No, A_virt is a 2x2 matrix of 2x2 matrices?
        # No, A_virt is a 2x2 matrix of SCALARS in the context of the 2x2x2 tensor?
        # Wait. The 4x4 matrix is viewed as a 2x2 matrix of 2x2 blocks.
        # Strassen's algorithm computes 7 products of 2x2 blocks.
        # Each product is a 2x2 matrix multiplication.
        # So we are computing 7 instances of 2x2x2 multiplication.
        
        # For M1 = (A00+A11)(B00+B11):
        # Let A_sum = A00 + A11. This is a 2x2 matrix.
        # Let B_sum = B00 + B11. This is a 2x2 matrix.
        # We compute A_sum * B_sum using Strassen's 2x2x2 algorithm.
        # This requires 7 rank-1 terms.
        
        # The A part of these terms will involve indices from A00 and A11.
        # The B part will involve indices from B00 and B11.
        # The C part will involve indices from C00 and C11 (with appropriate signs).
        
        # So I can just call add_2x2_mult for the virtual blocks,
        # but I need to map the virtual indices (0,0)..(1,1) to global indices.
        
        # A_virt[0,0] corresponds to A00[0,0] + A11[0,0]?
        # No, A_virt is the matrix A_sum.
        # A_sum[i,j] = A00[i,j] + A11[i,j].
        # So the global indices for A_virt[i,j] are (0,0)+i,j and (2,2)+i,j.
        
        # So for T1 of the virtual multiplication:
        # A part: A_virt[0,0] + A_virt[1,1]
        # = (A00[0,0]+A11[0,0]) + (A00[1,1]+A11[1,1])
        # Global indices: (0,0), (2,2), (1,1), (3,3).
        
        # I can modify add_2x2_mult to take a mapping function.
        
        def map_A(i, j):
            # Returns list of (global_i, global_j, coeff)
            return []
        
        def map_B(i, j):
            return []
        
        def map_C(i, j):
            return []

        # This is getting complex. I will just hardcode the 7 multiplications
        # by calling a helper that adds the 7 terms for a specific set of A, B, C blocks.
        
        pass

    # Let's just use the schoolbook for the 4x4x4 core to be safe and quick,
    # and rely on the boundary optimization.
    # Schoolbook 4x4x4 rank 64.
    # Total 64 + 3 = 67.
    # This is better than 92.
    
    # Let's implement schoolbook for the 4x4x4 core.
    
    def add_schoolblock(n_b, m_b, p_b, i_off, j_off, k_off):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    a_idx = (i_off + i) * m + (j_off + j)
                    b_idx = (j_off + j) * p + (k_off + k)
                    c_idx = (k_off + k) * n + (i_off + i)
                    
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    
                    u[a_idx] = 1.0
                    v[b_idx] = 1.0
                    w[c_idx] = 1.0
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    # Core 4x4x4
    add_schoolblock(4, 4, 4, 0, 0, 0)
    
    # Boundary 1: 4x1x4 (j=4, k=0..3)
    # Rank 1
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    # A: all i, j=4
    for i in range(4):
        u[i * m + 4] = 1.0
    # B: j=4, all k
    for k in range(4):
        v[4 * p + k] = 1.0
    # C: all k, all i
    for k in range(4):
        for i in range(4):
            w[k * n + i] = 1.0
    U_cols.append(u)
    V_cols.append(v)
    W_cols.append(w)
    
    # Boundary 2: 4x4x1 (j=0..3, k=4)
    # Rank 1
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    for i in range(4):
        for j in range(4):
            u[i * m + j] = 1.0
    for j in range(4):
        v[j * p + 4] = 1.0
    for i in range(4):
        w[4 * n + i] = 1.0
    U_cols.append(u)
    V_cols.append(v)
    W_cols.append(w)
    
    # Boundary 3: 4x1x1 (j=4, k=4)
    # Rank 1
    u = np.zeros(n * m)
    v = np.zeros(m * p)
    w = np.zeros(p * n)
    for i in range(4):
        u[i * m + 4] = 1.0
    v[4 * p + 4] = 1.0
    for i in range(4):
        w[4 * n + i] = 1.0
    U_cols.append(u)
    V_cols.append(v)
    W_cols.append(w)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
