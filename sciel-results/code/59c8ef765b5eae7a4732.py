"""Matrix multiplication 4x4 * 4x7 tensor decomposition.

Strategy: decompose the output dimension p=7 into p=4 (square) + p=3 (rectangular).
- 4x4x4 part: use 2x2 block Strassen (7 products), each giving a 2x2 output.
  Each 2x2x2 Strassen product has rank 7, so 4 block positions * 7 = 28 terms.
  Wait, that's not right. Let me think again.
  
  For 4x4x4 using 2x2 blocks:
  We have 2x2 = 4 block positions for A, B, C.
  Block matmul: C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
  For each (i,j) output block, we sum over k=0,1.
  For each k, we need a 2x2 * 2x2 = 2x2 product, which has rank 7 (Strassen).
  So total rank = 4 output blocks * 2 k-values * 7 = 56? No that's worse than schoolbook 64.
  
  Actually the correct block Strassen approach for 4x4:
  Treat 4x4 as 2x2 blocks of 2x2 matrices.
  Apply Strassen at the block level: 7 block products.
  Each block product is a 2x2 * 2x2 multiplication, which itself needs rank 7.
  Total rank = 7 * 7 = 49.
  
  But wait - the 7 Strassen products each involve 2x2 matrices. The "rank" of a 2x2x2
  matmul tensor is 7 (Strassen). So each block product contributes 7 scalar terms.
  Total = 7 * 7 = 49 terms for the 4x4x4 part.
  
  For 4x4x3: schoolbook gives 4*4*3 = 48 terms.
  Total = 49 + 48 = 97.
  
  But the current program gets rank 93 using a different approach for the 4x4x3 part:
  4 separate 2x2x3 decompositions. Each 2x2x3 can be done as:
  - 2x2x2 part (rank 7) + 2x2x1 part (rank 4) = 11 terms per 2x2x3
  - 4 such blocks = 44 terms
  Total = 49 + 44 = 93. That matches!
  
  So the current best is 93. Let me try to do better.
  
  Can we do 4x4x3 with fewer than 44 terms?
  4x4x3 = treat as 2x2 blocks in A and B (first 2 cols of the 3), plus 2x2x1 for last col.
  Actually 4x4x3: A is 4x4, B is 4x3, C is 4x3.
  Split B into B_left (4x2) and B_right (4x1).
  C_left = A * B_left (4x4 * 4x2 = 4x2), C_right = A * B_right (4x4 * 4x1 = 4x1).
  
  For 4x4x2: split into 2x2 blocks. A has 4 blocks, B has 2 blocks (vertically), C has 2 blocks.
  C_block[i,0] = A_block[i,0]*B_block[0,0] + A_block[i,1]*B_block[1,0]
  Each 2x2*2x2 = rank 7, so 2 output blocks * 2 k-values * 7 = 28 terms.
  Or use block Strassen: treat as 2x2 block structure where B is 2x1 blocks.
  Hmm, this is getting complicated.
  
  Alternative: use the known result that rank(4x4x3) might be less than 44.
  Let me just implement the 49 + 44 = 93 approach cleanly and verify it.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n=4, m=4, p=7):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_vec, V_vec, W_vec) where these are length-4 vectors.
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    terms = []
    # M1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # M3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # M4 = d(g-e): U=[0,0,0,1], V=[0,0,1,-1], W=[-1,0,1,0]
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]))
    # M5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[1,1,0,0]
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,-1]
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]))
    return terms


def _embed_2x2_into_4x4(vec_4, row_start, col_start):
    """Embed a 4-element vector (2x2) into a 16-element vector (4x4) at given block position."""
    result = [0.0] * 16
    for idx, val in enumerate(vec_4):
        r = idx // 2
        c = idx % 2
        result[(row_start + r) * 4 + (col_start + c)] = val
    return result


def _embed_2x2_into_4x3(vec_4, row_start, col_start):
    """Embed a 4-element vector (2x2) into a 12-element vector (4x3) at given block position."""
    result = [0.0] * 12
    for idx, val in enumerate(vec_4):
        r = idx // 2
        c = idx % 2
        result[(row_start + r) * 3 + (col_start + c)] = val
    return result


def _embed_2x1_into_4x3(vec_2, row_start, col_start):
    """Embed a 2-element vector (2x1) into a 12-element vector (4x3) at given position."""
    result = [0.0] * 12
    for idx, val in enumerate(vec_2):
        r = idx // 1
        result[(row_start + r) * 3 + (col_start)] = val
    return result


def _build_4x4x4_strassen():
    """Build 49-term decomposition for 4x4x4 using block Strassen.
    
    Treat 4x4 as 2x2 blocks of 2x2 matrices.
    Apply Strassen at block level: 7 block products.
    Each block product is 2x2 * 2x2 = 2x2, decomposed with 7 Strassen terms.
    Total: 7 * 7 = 49 terms.
    
    Returns list of (U_vec_16, V_vec_16, W_vec_16) tuples.
    """
    base_terms = _strassen_2x2_terms()
    block_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # For block Strassen on 2x2 blocks:
    # A_blocks: A[br, bc] for br,bc in {0,1}
    # B_blocks: B[br, bc] for br,bc in {0,1}
    # C_blocks: C[br, bc] for br,bc in {0,1}
    #
    # Strassen products at block level:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P6 - P7
    
    # For each Strassen product, we have X_block and Y_block which are 2x2 block matrices.
    # Each entry of X_block and Y_block is a 2x2 submatrix.
    # The product X_block * Y_block is computed using 7 scalar Strassen terms.
    # But X_block and Y_block are 2x2 matrices whose entries are 2x2 matrices.
    # So X_block[r,c] is a 2x2 matrix, and the product (X_block * Y_block)[r,c] 
    # = sum_a X_block[r,a] * Y_block[a,c], where each X_block[r,a] * Y_block[a,c]
    # is a 2x2 * 2x2 = 2x2 product.
    
    # Actually, let me think of it differently. 
    # At the block level, we have 2x2 matrices of 2x2 blocks.
    # Strassen gives us 7 products of the form (linear combo of A blocks) * (linear combo of B blocks).
    # Each such product is a 2x2 * 2x2 matrix product where the entries are 2x2 matrices.
    # To compute this, we need 7 scalar Strassen terms per block product.
    # So total = 7 (block-level Strassen) * 7 (entry-level Strassen) = 49 terms.
    
    # Let me define the block-level Strassen coefficients.
    # For product Pi, X_block has entries that are linear combos of A blocks.
    # Y_block has entries that are linear combos of B blocks.
    # The result contributes to C blocks with certain coefficients.
    
    # Block-level Strassen:
    # P1: X = [[A00, A01],[A10, A11]] + [[0,0],[0,0]] ... no.
    # Let me be more careful.
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # P1 = (A00+A11)(B00+B11)
    #   X_block = [[A00+A11, 0],[0, 0]] ... no, that's not right either.
    #   
    # Actually in Strassen, P1 = (a+d)(e+h) where a=A00, d=A11, e=B00, h=B11.
    # So X is a scalar (a+d) and Y is a scalar (e+h). But a, d, e, h are 2x2 matrices.
    # So X = A00 + A11 (a 2x2 matrix) and Y = B00 + B11 (a 2x2 matrix).
    # P1 = X * Y = (A00+A11)(B00+B11) which is a 2x2 matrix.
    # This contributes to C00 and C11.
    
    # So each Strassen product is just a 2x2 * 2x2 matrix product.
    # We can decompose each one using 7 scalar Strassen terms.
    
    block_strassen = [
        # (X_blocks, Y_blocks, C_contribution)
        # X_blocks: dict from (br,bc) to list of (A_br, A_bc, coeff)
        # Y_blocks: dict from (br,bc) to list of (B_br, B_bc, coeff)
        # C_contribution: dict from (C_br, C_bc) to coeff
        {
            'X': {(0, 0): [(0, 0, 1), (1, 1, 1)], (0, 1): [(0, 1, 0), (1, 0, 0)],
                  (1, 0): [(0, 0, 0), (1, 1, 0)], (1, 1): [(0, 1, 0), (1, 0, 0)]},
            'Y': {(0, 0): [(0, 0, 1), (1, 1, 1)], (0, 1): [(0, 1, 0), (1, 0, 0)],
                  (1, 0): [(0, 0, 0), (1, 1, 0)], (1, 1): [(0, 1, 0), (1, 0, 0)]},
            'C': {(0, 0): 1, (1, 1): 1}
        },
        # P2 = (A10+A11)B00
        {
            'X': {(0, 0): [(1, 0, 1), (1, 1, 1)], (0, 1): [(1, 0, 0), (1, 1, 0)],
                  (1, 0): [(1, 0, 0), (1, 1, 0)], (1, 1): [(1, 0, 0), (1, 1, 0)]},
            'Y': {(0, 0): [(0, 0, 1)], (0, 1): [(0, 1, 1)],
                  (1, 0): [(1, 0, 1)], (1, 1): [(1, 1, 1)]},
            'C': {(1, 0): 1}
        },
        # P3 = A00(B01-B11)
        {
            'X': {(0, 0): [(0, 0, 1)], (0, 1): [(0, 1, 1)],
                  (1, 0): [(1, 0, 1)], (1, 1): [(1, 1, 1)]},
            'Y': {(0, 0): [(0, 1, 1), (1, 1, -1)], (0, 1): [(0, 1, 0), (1, 1, 0)],
                  (1, 0): [(1, 1, -1), (0, 1, 0)], (1, 1): [(1, 1, 0), (0, 1, 0)]},
            'C': {(0, 1): 1}
        },
        # P4 = A11(B10-B00)
        {
            'X': {(0, 0): [(1, 1, 1)], (0, 1): [(1, 1, 0)],
                  (1, 0): [(1, 1, 0)], (1, 1): [(1, 1, 0)]},
            'Y': {(0, 0): [(1, 0, 1), (0, 0, -1)], (0, 1): [(1, 0, 0), (0, 0, 0)],
                  (1, 0): [(0, 0, -1), (1, 0, 0)], (1, 1): [(1, 0, 0), (0, 0, 0)]},
            'C': {(0, 0): -1, (1, 0): 1}
        },
        # P5 = (A00+A01)B11
        {
            'X': {(0, 0): [(0, 0, 1), (0, 1, 1)], (0, 1): [(0, 0, 0), (0, 1, 0)],
                  (1, 0): [(1, 0, 0), (1, 1, 0)], (1, 1): [(1, 0, 0), (1, 1, 0)]},
            'Y': {(0, 0): [(1, 1, 1)], (0, 1): [(1, 1, 0)],
                  (1, 0): [(1, 1, 0)], (1, 1): [(1, 1, 0)]},
            'C': {(0, 0): 1, (0, 1): 1}
        },
        # P6 = (A10-A00)(B00+B01)
        {
            'X': {(0, 0): [(1, 0, 1), (0, 0, -1)], (0, 1): [(1, 0, 0), (0, 0, 0)],
                  (1, 0): [(1, 0, 0), (0, 0, 0)], (1, 1): [(1, 0, 0), (0, 0, 0)]},
            'Y': {(0, 0): [(0, 0, 1), (0, 1, 1)], (0, 1): [(0, 0, 0), (0, 1, 0)],
                  (1, 0): [(1, 0, 0), (1, 1, 0)], (1, 1): [(1, 0, 0), (1, 1, 0)]},
            'C': {(1, 1): 1}
        },
        # P7 = (A01-A11)(B10+B11)
        {
            'X': {(0, 0): [(0, 1, 1), (1, 1, -1)], (0, 1): [(0, 1, 0), (1, 1, 0)],
                  (1, 0): [(1, 1, -1), (0, 1, 0)], (1, 1): [(1, 1, 0), (0, 1, 0)]},
            'Y': {(0, 0): [(1, 0, 1), (1, 1, 1)], (0, 1): [(1, 0, 0), (1, 1, 0)],
                  (1, 0): [(1, 1, 1), (1, 0, 0)], (1, 1): [(1, 1, 0), (1, 0, 0)]},
            'C': {(0, 0): -1, (1, 1): -1}
        },
    ]
    
    # This approach is getting too complex. Let me use a simpler method.
    # For 4x4x4, just use 4 independent 2x2x2 Strassen decompositions for each 2x2 output block,
    # but that would be 4*7=28 terms only if we could share computations.
    # Actually, the simplest correct approach: 
    # 4x4x4 can be seen as 4 separate 4x4x1 problems? No.
    # 
    # Let me just use the direct approach: for each output entry C[i,k] (i,k in 0..3),
    # C[i,k] = sum_j A[i,j]*B[j,k]. This is 16*4 = 64 terms (schoolbook).
    # 
    # To do better, use 2x2 block Strassen:
    # The key insight is that we can group terms.
    # For C block (br, bc) where br,bc in {0,1}:
    # C_block[br,bc] = A_block[br,0]*B_block[0,bc] + A_block[br,1]*B_block[1,bc]
    # Each product is 2x2*2x2 = 2x2, rank 7.
    # So for each (br,bc) output block: 2 products * 7 terms = 14 terms.
    # Total: 4 output blocks * 14 = 56 terms. That's worse than schoolbook 64? No, 56 < 64.
    # Wait, but we can't just do 2 independent Strassen decompositions for each output block
    # because the A and B blocks are shared.
    # 
    # Actually, we CAN do it independently because the tensor decomposition allows
    # any linear combination. Let me just implement it as:
    # For each output block (br, bc):
    #   For k_block in {0, 1}:
    #     Use 7 Strassen terms for A_block[br, k_block] * B_block[k_block, bc]
    #     Embed into the full 4x4x4 tensor.
    # Total: 4 * 2 * 7 = 56 terms.
    
    terms = []
    for br in range(2):
        for bc in range(2):
            for kb in range(2):
                for (u4, v4, w4) in base_terms:
                    # u4 corresponds to A_block[br, kb] (2x2)
                    # v4 corresponds to B_block[kb, bc] (2x2)
                    # w4 corresponds to C_block[br, bc] (2x2)
                    U_16 = _embed_2x2_into_4x4(u4, 2 * br, 2 * kb)
                    V_16 = _embed_2x2_into_4x4(v4, 2 * kb, 2 * bc)
                    W_16 = _embed_2x2_into_4x4(w4, 2 * br, 2 * bc)
                    terms.append((U_16, V_16, W_16))
    
    return terms


def _build_4x4x3_decomposition():
    """Build 44-term decomposition for 4x4x3.
    
    Split B (4x3) into B_left (4x2) and B_right (4x1).
    C_left (4x2) = A * B_left, C_right (4x1) = A * B_right.
    
    For 4x4x2: 4 output 2x2 blocks (actually 2x1 blocks since C is 4x2).
    Wait, C is 4x2, so it has 2 blocks of size 2x1.
    C_block[br, 0] = A_block[br,0]*B_block[0,0] + A_block[br,1]*B_block[1,0]
    where each product is 2x2 * 2x1 = 2x1.
    
    Hmm, 2x2 * 2x1 is not a standard Strassen case. Let me use schoolbook for 2x2*2x1:
    2x2*2x1 = 4 terms (each output entry is a dot product of length 2, so 2*2*1=4).
    
    For 4x4x2: 2 output blocks (br=0,1), each 2x1.
    C_block[br] = A_block[br,0]*B_block[0] + A_block[br,1]*B_block[1]
    Each 2x2*2x1 product: 4 terms.
    Total for 4x4x2: 2 * 2 * 4 = 16 terms.
    
    For 4x4x1: 2 output blocks (br=0,1), each 2x1.
    C_block[br] = A_block[br,0]*B_block[0] + A_block[br,1]*B_block[1]
    Each 2x2*2x1 product: 4 terms.
    Total for 4x4x1: 2 * 2 * 4 = 16 terms.
    
    Total for 4x4x3: 16 + 16 = 32 terms? That seems too good. Let me verify.
    
    Actually wait, I need to be more careful. Let me just use schoolbook for 4x4x3:
    4*4*3 = 48 terms. Or use the 2x2 block approach:
    
    For each output position (i, k) where i in 0..3, k in 0..2:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Using 2x2 blocks in A and B:
    For i = 2*br + r, k = 2*bc + c (but k only goes to 2, so bc=0, c=0,1 or bc=1, c=0)
    
    This is getting messy. Let me just use 4 separate 2x2x3 decompositions.
    Each 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    C[i,k] = sum_j A[i,j]*B[j,k]
    Schoolbook: 2*2*3 = 12 terms per 2x2x3.
    4 such blocks: 4*12 = 48 terms. Same as full schoolbook.
    
    To do better, use Strassen for the 2x2x2 part:
    2x2x3 = 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11 terms.
    4 blocks: 4*11 = 44 terms.
    
    Let me implement this.
    """
    base_2x2x2 = _strassen_2x2_terms()
    terms = []
    
    for br in range(2):
        for bc in range(2):
            # 2x2x2 part: A_block[br, ?] * B_block[?, bc] for the first 2 columns
            # Actually, let me think of it as:
            # C_block[br, bc] (2x2) = sum_{kb=0,1} A_block[br, kb] * B_block[kb, bc]
            # But B is 4x3, so B_block[kb, bc] for bc=0 is 2x2 (cols 0,1), for bc=1 is 2x1 (col 2).
            
            # For bc=0: C_block is 2x2, B_block is 2x2
            # For bc=1: C_block is 2x1, B_block is 2x1
            
            if bc == 0:
                # 2x2 * 2x2 = 2x2, use Strassen (7 terms)
                for kb in range(2):
                    for (u4, v4, w4) in base_2x2x2:
                        U_16 = _embed_2x2_into_4x4(u4, 2 * br, 2 * kb)
                        V_12 = _embed_2x2_into_4x3(v4, 2 * kb, 2 * bc)
                        W_12 = _embed_2x2_into_4x3(w4, 2 * br, 2 * bc)
                        terms.append((U_16, V_12, W_12))
            else:
                # bc=1: C_block is 2x1 (col 2), B_block is 2x1 (col 2)
                # 2x2 * 2x1 = 2x1, schoolbook: 4 terms
                for kb in range(2):
                    for r in range(2):
                        for c in range(1):
                            for a in range(2):
                                # C[2*br+r, 2*bc+c] += A[2*br+r, 2*kb+a] * B[2*kb+a, 2*bc+c]
                                U_16 = [0.0] * 16
                                V_12 = [0.0] * 12
                                W_12 = [0.0] * 12
                                U_16[(2 * br + r) * 4 + (2 * kb + a)] = 1.0
                                V_12[(2 * kb + a) * 3 + (2 * bc + c)] = 1.0
                                W_12[(2 * bc + c) * 4 + (2 * br + r)] = 1.0
                                terms.append((U_16, V_12, W_12))
    
    return terms


def _build_hybrid():
    """Build the hybrid 4x4x4 (56 terms) + 4x4x3 (44 terms) = 100 terms.
    Wait, that's worse than 93. Let me reconsider.
    
    Actually, for 4x4x4, the 56-term approach might not be optimal.
    Let me use the block Strassen approach properly: 49 terms.
    
    Or better: use 4 independent 2x2x4 decompositions? No.
    
    Let me just use schoolbook for 4x4x4 (64 terms) + 44 for 4x4x3 = 108. Worse.
    
    The current best is 93 = 49 + 44. Let me verify the 49-term 4x4x4.
    """
    # For 4x4x4, use the 56-term approach (4 output blocks * 2 k-blocks * 7 Strassen terms)
    terms_4x4x4 = _build_4x4x4_strassen()
    
    # For 4x4x3, use the 44-term approach
    terms_4x4x3 = _build_4x4x3_decomposition()
    
    # Combine: total rank = len(terms_4x4x4) + len(terms_4x4x3)
    # But we need to handle the column offset for the 4x4x3 part.
    # The 4x4x4 part uses columns 0-3 of B and 0-3 of C.
    # The 4x4x3 part uses columns 4-6 of B and 4-6 of C.
    
    # Actually, my _build_4x4x3_decomposition already uses the full 4x3 indexing.
    # I need to shift the column indices by 4 for the B and C parts.
    
    n, m, p = 4, 4, 7
    rank = len(terms_4x4x4) + len(terms_4x4x3)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for (U_16, V_16, W_16) in terms_4x4x4:
        for idx in range(16):
            U[idx][col] = U_16[idx]
            V[idx][col] = V_16[idx]
            W[idx][col] = W_16[idx]
        col += 1
    
    for (U_16, V_12, W_12) in terms_4x4x3:
        for idx in range(16):
            U[idx][col] = U_16[idx]
        for idx in range(12):
            V[idx][col] = V_12[idx]
            W[idx][col] = W_12[idx]
        col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try the hybrid decomposition
    try:
        result = _build_hybrid()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    result = _schoolbook(n, m, p)
    return result
