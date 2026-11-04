import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition using Strassen for 4x4x4 and schoolbook for 4x4x1
    # Rank for 4x4x4 part: 49
    # Rank for 4x4x1 part: 16
    # Total rank: 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # coeff: coefficient for this term
    # col: column in U, V, W
    
    def add_term(u_idx, v_idx, w_idx, coeff, col):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff

    # --- Part 1: 4x4x4 using Strassen ---
    # A is 4x4, B1 is 4x4 (cols 0-3 of B), C1 is 4x4 (cols 0-3 of C)
    
    # Strassen 2x2x2 coefficients
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to map the 4x4 matrix indices to 2x2 block indices.
    # Block (bi, bj) of A contains A[2*bi + i, 2*bj + j]
    
    # For each of the 7 block products, we apply Strassen 2x2x2.
    # Each 2x2x2 product uses 7 scalar multiplications.
    # Total 49 scalar terms.
    
    col = 0
    
    # Define the 7 Strassen linear forms for A and B
    # Each form is a dict mapping (i, j) to coeff
    # A forms:
    # F1_A: a00=1, a11=1
    # F2_A: a10=1, a11=1
    # F3_A: a00=1
    # F4_A: a11=1
    # F5_A: a00=1, a01=1
    # F6_A: a10=1, a00=-1
    # F7_A: a01=1, a11=-1
    
    # B forms:
    # G1_B: b00=1, b11=1
    # G2_B: b00=1
    # G3_B: b01=1, b11=-1
    # G4_B: b10=1, b00=-1
    # G5_B: b11=1
    # G6_B: b00=1, b01=1
    # G7_B: b10=1, b11=1
    
    # C forms (output coefficients for M1..M7):
    # C00: M1=1, M4=1, M5=-1, M7=1
    # C01: M3=1, M5=1
    # C10: M2=1, M4=1
    # C11: M1=1, M2=-1, M3=1, M6=1
    
    # We iterate over the 7 block products (k=1..7)
    # For each, we iterate over the 7 scalar Strassen terms (s=1..7)
    # This gives 49 terms.
    
    # However, the Strassen algorithm computes M_k = L_k(A) * R_k(B).
    # Then C = sum M_k * E_k, where E_k are matrices.
    # So the tensor term is L_k(A) * R_k(B) * E_k(C).
    # This is a rank-1 term in the tensor decomposition?
    # No, it's a rank-1 term in the bilinear map if we consider L_k, R_k as vectors.
    # But L_k(A) is a scalar. R_k(B) is a scalar.
    # The product is a scalar.
    # Then we multiply by E_k(C) which is a matrix.
    # So the contribution to C is E_k * (L_k(A) R_k(B)).
    # This can be written as sum over c_idx of E_k[c_idx] * L_k(A) * R_k(B).
    # So for each non-zero entry in E_k, we have a rank-1 term.
    
    # Let's re-evaluate.
    # Strassen: C = sum_{k=1}^7 E_k (M_k)
    # M_k = L_k(A) R_k(B)
    # So C = sum_{k=1}^7 sum_{c_idx} E_k[c_idx] L_k(A) R_k(B)
    # Each term is (L_k(A)) * (R_k(B)) * (E_k[c_idx] at c_idx)
    # This is a rank-1 tensor term.
    
    # So for each of the 7 block products, we have 7 scalar multiplications (M_k is computed via Strassen on the 2x2 blocks? No).
    # Wait. The 4x4x4 multiplication is done by applying Strassen to the 2x2 blocks.
    # The "blocks" are 2x2 matrices.
    # So M_k is a 2x2 matrix.
    # M_k = A_block_k * B_block_k.
    # To compute M_k, we use Strassen's algorithm on the 2x2 matrices.
    # This requires 7 scalar multiplications.
    # So for each of the 7 block products M_k, we have 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # For each scalar multiplication, we have a term L_s(A_block) * R_s(B_block).
    # This contributes to M_k[i, j].
    # M_k[i, j] contributes to C[bi*2+i, bj*2+j] via the Strassen output formulas.
    
    # This is getting complex to index.
    # Alternative: Just hardcode the 49 terms?
    # Or use a recursive function that generates the terms.
    
    # Given the time limit and complexity, and that 65 is a good improvement,
    # I will implement the 49 terms by iterating through the Strassen structure.
    
    # Let's define the 7 Strassen forms for 2x2 matrices.
    # A forms (coeffs for a00, a01, a10, a11):
    A_forms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    # B forms (coeffs for b00, b01, b10, b11):
    B_forms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # C output coeffs for M1..M7 (for c00, c01, c10, c11):
    C_forms = [
        [1, 0, 0, 0], # M1 -> c00
        [0, 0, 1, 0], # M2 -> c10
        [0, 1, 0, 0], # M3 -> c01
        [1, 0, 0, 0], # M4 -> c00 (wait, M4 is in c00 and c10)
        # Let's use the standard formulas:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So for each M_k, we know which c_ij it contributes to.
    ]
    
    # Mapping M_k to c_ij contributions:
    # M1: c00 (+1), c11 (+1)
    # M2: c10 (+1), c11 (-1)
    # M3: c01 (+1), c11 (+1)
    # M4: c00 (+1), c10 (+1)
    # M5: c00 (-1), c01 (+1)
    # M6: c11 (+1)
    # M7: c00 (+1)
    
    M_to_C = {
        0: [(0, 1), (3, 1)], # M1: c00, c11
        1: [(2, 1), (3, -1)], # M2: c10, c11
        2: [(1, 1), (3, 1)], # M3: c01, c11
        3: [(0, 1), (2, 1)], # M4: c00, c10
        4: [(0, -1), (1, 1)], # M5: c00, c01
        5: [(3, 1)], # M6: c11
        6: [(0, 1)] # M7: c00
    }
    
    # For the 4x4x4 part:
    # We have 7 block products.
    # For each block product k (0..6), we compute M_k (2x2 matrix) using Strassen.
    # M_k[i, j] = sum_{s=0}^6 A_forms[s][i, j] * B_forms[s][i, j]? No.
    # M_k is computed by applying Strassen to the 2x2 blocks of A and B.
    # So M_k[i, j] is the (i, j) entry of the product of the 2x2 blocks.
    # This product is computed using 7 scalar multiplications.
    # Let's call the scalar multiplications for block k as S_{k, s} for s=0..6.
    # S_{k, s} = L_s(A_block_k) * R_s(B_block_k).
    # Then M_k[i, j] = sum_{s=0}^6 C_forms_s[i, j] * S_{k, s}.
    
    # And C[bi*2+i, bj*2+j] = sum_{k=0}^6 M_to_C[k][...].
    
    # This is too nested.
    # Let's just generate the 49 terms directly.
    
    # Term index t from 0 to 48.
    # t = k * 7 + s.
    # k is the block product index (0..6).
    # s is the scalar Strassen index (0..6).
    
    for k in range(7):
        for s in range(7):
            # Get the linear forms for A and B for this scalar multiplication
            # A_form_s: coeffs for a00, a01, a10, a11 of the block
            a_coeff = A_forms[s]
            b_coeff = B_forms[s]
            
            # The block k corresponds to specific 2x2 submatrices of A and B.
            # We need to map the 2x2 block indices to global 4x4 indices.
            # Block k in Strassen 4x4x4:
            # k=0: M1 = (A00+A11)(B00+B11) -> Blocks A00, A11 and B00, B11
            # This is not a single 2x2 block. It's a combination.
            
            # This approach is flawed. Strassen for 4x4x4 is not simply 7 independent 2x2x2 multiplications.
            # It is 7 products of 2x2 matrices.
            # Each product is a 2x2x2 multiplication.
            # So yes, it IS 7 independent 2x2x2 multiplications.
            # But the inputs to these multiplications are linear combinations of the 2x2 blocks.
            
            # So for k=0:
            # A_input = A00 + A11
            # B_input = B00 + B11
            # M0 = A_input * B_input
            # M0 is a 2x2 matrix.
            # To compute M0, we use Strassen on A_input and B_input.
            # This requires 7 scalar multiplications.
            
            # So for each k, we define A_input and B_input as linear combinations of the 4x4 blocks.
            # Then we apply Strassen 2x2x2 to A_input and B_input.
            
            # Block definitions for 4x4 Strassen:
            # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
            # Same for B.
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For each M_k, we have A_k and B_k (2x2 matrices).
            # We apply Strassen 2x2x2 to A_k and B_k.
            # This gives 7 scalar terms.
            
            # We need the definitions of A_k and B_k in terms of the 4x4 entries.
            
            # This is doable but verbose.
            
            # Given the constraints, I will return the schoolbook rank 80 if I can't implement Strassen quickly.
            # But 65 is a good target.
            
            # Let's try a simpler improvement: 4x4x5 = 4x4x4 + 4x4x1.
            # Rank 56 (Strassen 4x4x4 naive block) + 16 = 72.
            # This is easier to implement.
            
            pass # Skip complex Strassen 4x4x4 for now, use 56-rank block Strassen.

    # Re-implement with 56-rank 4x4x4 part.
    rank = 56 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 4x4x4 part using block Strassen (8 blocks * 7 mults = 56)
    # Blocks: A00, A01, A10, A11 and B00, B01, B10, B11
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # For each of the 8 products (e.g., A00B00), use Strassen 2x2x2.
    
    block_products = [
        (0, 0, 0, 0), # A00 B00 -> C00
        (0, 1, 0, 1), # A00 B01 -> C01
        (1, 0, 0, 0), # A10 B00 -> C10
        (1, 1, 0, 1), # A10 B01 -> C11
        (0, 0, 1, 0), # A01 B10 -> C00
        (0, 1, 1, 1), # A01 B11 -> C01
        (1, 0, 1, 0), # A11 B10 -> C10
        (1, 1, 1, 1)  # A11 B11 -> C11
    ]
    
    # Strassen 2x2x2 forms
    A_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    B_forms = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    C_forms = [
        [1, 0, 0, 0], # M1 -> c00
        [0, 1, 0, 0], # M2 -> c01? No.
        # Standard Strassen C forms:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So for each M_k, we add to specific c_ij.
    ]
    
    # Mapping M_k to c_ij
    M_to_C = [
        [(0, 1), (3, 1)], # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)], # M3
        [(0, 1), (2, 1)], # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)], # M6
        [(0, 1)] # M7
    ]
    
    for bi, bj, ci, cj in block_products:
        # A_block is A[2*bi:2*bi+2, 2*bj:2*bj+2]
        # B_block is B[2*bi:2*bi+2, 2*bj:2*bj+2] (Note: B indices are j, k. Here j is row of B, k is col of B.
        # In our 4x4x4, B is 4x4. Rows 0-3, Cols 0-3.
        # Block B[bi, bj] is B[2*bi:2*bi+2, 2*bj:2*bj+2].
        
        for s in range(7):
            # A_form_s gives coeffs for a00, a01, a10, a11 of the block
            a_coeff = A_forms[s]
            b_coeff = B_forms[s]
            
            # Add term to U, V, W
            # U: A entries
            # V: B entries
            # W: C entries
            
            # We need to sum over the 4 entries of the block for A and B.
            # This term is a product of a linear form in A_block and a linear form in B_block.
            # It contributes to C_block via M_to_C.
            
            # Since U, V, W are rank-1 terms, we can't directly put a linear form.
            # We must decompose the linear form into rank-1 terms?
            # No, the rank-1 term in the tensor decomposition is:
            # (sum a_u A_u) * (sum b_v B_v) * c_w C_w
            # This is NOT a rank-1 term in the tensor T.
            # A rank-1 term is u_i v_j w_k.
            # The product of linear forms is a sum of rank-1 terms.
            
            # So (sum a_u A_u) * (sum b_v B_v) = sum_{u, v} a_u b_v A_u B_v.
            # This is a sum of rank-1 terms.
            
            # So each Strassen scalar multiplication expands into 4x4=16 rank-1 terms?
            # No, the linear forms are sparse.
            # a_coeff has 2 non-zeros. b_coeff has 2 non-zeros.
            # So 4 rank-1 terms per Strassen scalar multiplication.
            
            # 8 blocks * 7 mults * 4 terms = 224 terms?
            # This is worse than schoolbook 80.
            
            # I made a mistake. Strassen's algorithm reduces the NUMBER OF MULTIPLICATIONS.
            # In the tensor decomposition, each multiplication corresponds to a rank-1 term.
            # The linear combinations are handled by the U, V, W matrices.
            # U, V, W can contain the coefficients of the linear forms.
            
            # So for a term L(A) * R(B) * C_out:
            # U has the coeffs of L(A).
            # V has the coeffs of R(B).
            # W has the coeffs of C_out.
            
            # So it IS a single rank-1 term.
            
            # So we can just add the coefficients to U, V, W.
            
            for u in range(4):
                if a_coeff[u] != 0:
                    # u is local index in block (0,1,2,3) -> (0,0), (0,1), (1,0), (1,1)
                    li = u // 2
                    lj = u % 2
                    gi = 2*bi + li
                    gj = 2*bj + lj
                    U[gi*m + gj][col] += a_coeff[u]
                    
            for v in range(4):
                if b_coeff[v] != 0:
                    li = v // 2
                    lj = v % 2
                    gi = 2*bi + li
                    gj = 2*bj + lj
                    V[gi*p + gj][col] += b_coeff[v]
                    
            for (ci_local, coeff_c) in M_to_C[s]:
                li = ci_local // 2
                lj = ci_local % 2
                gi = 2*ci + li
                gj = 2*cj + lj
                W[gj*n + gi][col] += coeff_c
            
            col += 1

    # 4x4x1 part
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1 # Col 4 of B
            W[4*n + i][col] = 1 # Col 4 of C
            col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
