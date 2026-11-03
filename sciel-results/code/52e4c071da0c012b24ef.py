import time
import numpy as np
from itertools import product

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

# --- Laderman 3x3 matrix multiplication decomposition ---
# Laderman's algorithm uses 15 multiplications for 3x3 matrices.
# The 15 scalar products and their contributions to the output C = A*B:
#
# Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
# Let B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
#
# The 15 products are:
# P0 = a00*b00
# P1 = (a01+a02)*b00
# P2 = a00*(b01+b02)
# P3 = (a10+a11+a20+a21)*(b00+b10+b20+b21)
# P4 = a02*b11
# P5 = a22*b22
# P6 = (a01+a02-a21-a22)*(b21+b22-b01-b02)
# P7 = (a11+a12+a21+a22)*(b11+b12+b21+b22)
# P8 = (a10-a20)*(b20+b21)
# P9 = a20*(b10+b11)
# P10 = (a10+a11-a20-a21)*(b20+b21-b10-b11)
# P11 = (a11+a12-a21-a22)*(b21+b22-b11-b12)
# P12 = a12*(b01+b02)
# P13 = a01*b22
# P14 = (a10-a11+a20-a21)*(b10-b11+b20-b21)
#
# Output C[i][j] = sum of contributions:
# C00 = P0 + P4 + P5 + P10 - P12 - P13 + P14 - P3 + P7
# C01 = P1 + P2 + P12 + P13 + P6 + P7 - P3 - P11 - P14
# C02 = P4 + P5 + P10 - P12 - P13 + P14 - P3 + P7  (same pattern shifted)
#
# Let me be more careful. Let me look up Laderman's exact formulas.
#
# Laderman (1976) 3x3 matrix multiplication, 15 multiplications:
#
# Define the 15 scalar products:
#  1. (a11 + a21 + a12 + a22) * (b11 + b12 + b21 + b22)
#  2. (a11 + a12 + a21 + a22) * (b11 + b12 + b21 + b22)  -- wait, this is same
#
# Let me use a well-known reference. Laderman's algorithm:
#
# M1 = a00 * b00
# M2 = (a01 + a02) * b00
# M3 = a00 * (b01 + b02)
# M4 = (a10 + a11 + a20 + a21) * (b00 + b10 + b20 + b21)
# M5 = a02 * b11
# M6 = a22 * b22
# M7 = (a01 + a02 - a21 - a22) * (b21 + b22 - b01 - b02)
# M8 = (a11 + a12 + a21 + a22) * (b11 + b12 + b21 + b22)
# M9 = (a10 - a20) * (b20 + b21)
# M10 = a20 * (b10 + b11)
# M11 = (a10 + a11 - a20 - a21) * (b20 + b21 - b10 - b11)
# M12 = (a11 + a12 - a21 - a22) * (b21 + b22 - b11 - b12)
# M13 = a12 * (b01 + b02)
# M14 = a01 * b22
# M15 = (a10 - a11 + a20 - a21) * (b10 - b11 + b20 - b21)
#
# Then:
# c00 = M1 + M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
# c01 = M2 + M3 + M13 + M14 + M7 + M8 - M4 - M12 - M15
# c02 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
# c10 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8  -- hmm, this doesn't look right
#
# Let me try a different approach. I'll use a verified Laderman decomposition.
# After careful consideration, let me use the following well-known 15-product 
# Laderman scheme:
#
# Actually, let me just verify by construction. I'll build the factor matrices
# for a 3x3 Laderman decomposition and verify it's correct before using it
# for 6x6.

def _build_laderman_3x3():
    """Build Laderman 3x3 matrix multiplication decomposition.
    Returns U, V, W as lists of lists with shapes (9,15), (9,15), (9,15).
    U[i*3+j] corresponds to A[i][j], V[i*3+j] to B[i][j], W[i*3+j] to C[i][j].
    """
    n = 3
    rank = 15
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # Define the 15 bilinear forms.
    # Each is (list of A-indices with coeffs, list of B-indices with coeffs, list of C-indices with coeffs)
    # Index convention: A[i*n+j] = A[i][j]
    
    # M1 = a00 * b00
    # M2 = (a01+a02) * b00
    # M3 = a00 * (b01+b02)
    # M4 = (a10+a11+a20+a21) * (b00+b10+b20+b21)
    # M5 = a02 * b11
    # M6 = a22 * b22
    # M7 = (a01+a02-a21-a22) * (b21+b22-b01-b02)
    # M8 = (a11+a12+a21+a22) * (b11+b12+b21+b22)
    # M9 = (a10-a20) * (b20+b21)
    # M10 = a20 * (b10+b11)
    # M11 = (a10+a11-a20-a21) * (b20+b21-b10-b11)
    # M12 = (a11+a12-a21-a22) * (b21+b22-b11-b12)
    # M13 = a12 * (b01+b02)
    # M14 = a01 * b22
    # M15 = (a10-a11+a20-a21) * (b10-b11+b20-b21)
    
    A_forms = [
        [(0,1)],                          # M1: a00
        [(1,1),(2,1)],                    # M2: a01+a02
        [(0,1)],                          # M3: a00
        [(4,1),(5,1),(7,1),(8,1)],       # M4: a10+a11+a20+a21
        [(2,1)],                          # M5: a02
        [(8,1)],                          # M6: a22
        [(1,1),(2,1),(8,-1),(7,-1)],     # M7: a01+a02-a21-a22  (a21=idx 8? No. a[2][1]=idx 2*3+1=7)
        [(5,1),(6,1),(7,1),(8,1)],       # M8: a11+a12+a21+a22
        [(4,1),(7,-1)],                   # M9: a10-a20  (a20=idx 2*3+0=6)
        [(6,1)],                          # M10: a20
        [(4,1),(5,1),(6,-1),(7,-1)],     # M11: a10+a11-a20-a21
        [(5,1),(6,1),(7,-1),(8,-1)],     # M12: a11+a12-a21-a22
        [(6,1)],                          # M13: a12  (a12=idx 1*3+2=5? No. a[1][2]=5)
        [(1,1)],                          # M14: a01
        [(4,1),(5,-1),(6,1),(7,-1)],     # M15: a10-a11+a20-a21
    ]
    
    # Fix indices: A[i][j] = i*3+j
    # a00=0, a01=1, a02=2
    # a10=3, a11=4, a12=5
    # a20=6, a21=7, a22=8
    
    A_forms = [
        [(0,1)],                          # M1: a00
        [(1,1),(2,1)],                    # M2: a01+a02
        [(0,1)],                          # M3: a00
        [(3,1),(4,1),(6,1),(7,1)],       # M4: a10+a11+a20+a21
        [(2,1)],                          # M5: a02
        [(8,1)],                          # M6: a22
        [(1,1),(2,1),(7,-1),(8,-1)],     # M7: a01+a02-a21-a22
        [(4,1),(5,1),(7,1),(8,1)],       # M8: a11+a12+a21+a22
        [(3,1),(6,-1)],                   # M9: a10-a20
        [(6,1)],                          # M10: a20
        [(3,1),(4,1),(6,-1),(7,-1)],     # M11: a10+a11-a20-a21
        [(4,1),(5,1),(7,-1),(8,-1)],     # M12: a11+a12-a21-a22
        [(5,1)],                          # M13: a12
        [(1,1)],                          # M14: a01
        [(3,1),(4,-1),(6,1),(7,-1)],     # M15: a10-a11+a20-a21
    ]
    
    B_forms = [
        [(0,1)],                          # M1: b00
        [(0,1)],                          # M2: b00
        [(1,1),(2,1)],                    # M3: b01+b02
        [(0,1),(3,1),(6,1),(7,1)],       # M4: b00+b10+b20+b21
        [(4,1)],                          # M5: b11
        [(8,1)],                          # M6: b22
        [(7,1),(8,1),(1,-1),(2,-1)],     # M7: b21+b22-b01-b02
        [(4,1),(5,1),(7,1),(8,1)],       # M8: b11+b12+b21+b22
        [(6,1),(7,1)],                    # M9: b20+b21
        [(3,1),(4,1)],                    # M10: b10+b11
        [(6,1),(7,1),(3,-1),(4,-1)],     # M11: b20+b21-b10-b11
        [(7,1),(8,1),(4,-1),(5,-1)],     # M12: b21+b22-b11-b12
        [(1,1),(2,1)],                    # M13: b01+b02
        [(8,1)],                          # M14: b22
        [(3,1),(4,-1),(6,1),(7,-1)],     # M15: b10-b11+b20-b21
    ]
    
    # C forms: which C[i][j] get which products with what coefficients
    # C[i][j] = i*3+j: c00=0, c01=1, c02=2, c10=3, c11=4, c12=5, c20=6, c21=7, c22=8
    #
    # c00 = M1 + M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    # c01 = M2 + M3 + M13 + M14 + M7 + M8 - M4 - M12 - M15
    # c02 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    # c10 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    # c11 = M2 + M3 + M13 + M14 + M7 + M8 - M4 - M12 - M15
    # c12 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    # c20 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    # c21 = M2 + M3 + M13 + M14 + M7 + M8 - M4 - M12 - M15
    # c22 = M5 + M6 + M11 - M13 - M14 + M15 - M4 + M8
    
    # Hmm, this pattern seems off. Let me reconsider.
    # The Laderman algorithm has a specific structure. Let me use a verified version.
    # 
    # Actually, I realize I should just verify this numerically. Let me build the
    # matrices and check. But first, let me try a different well-known formulation.
    
    # Let me use the following verified Laderman 3x3 decomposition:
    # Source: various references on Laderman's 15-product algorithm
    
    # I'll define the C contributions as a list of 15 lists, each containing (c_idx, coeff)
    C_forms = [
        [(0,1), (1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1)],  # placeholder
    ]
    
    # This is getting complex. Let me just hardcode the full 3x3 Laderman
    # by defining U, V, W directly from a known-correct reference.
    
    # After research, here is a correct Laderman 3x3:
    # I'll define it as three lists of (index, coeff) for each of the 15 products.
    
    # Let me just verify the decomposition works by testing it.
    # For now, I'll use the Strassen 2x2 approach which is known to work (rank 189).
    # The 6x6 can be decomposed as 3x3 blocks of 2x2 matrices.
    # Rank = 3*3*7 = 63... wait, that's not right either.
    # 6x6 = (2x2 blocks arranged 3x3). Each 2x2 uses Strassen (7 mults).
    # Total rank = 3*3*7 = 63? No, that would be for block-level Strassen.
    # Actually, for 6x6 with 2x2 sub-blocks (3x3 grid of blocks):
    # The block matrix is 3x3, and we need to multiply 3x3 block matrices where
    # each block is 2x2. We can use Strassen for the 3x3 block multiplication
    # but Strassen is for 2x2. For 3x3 we'd need a different algorithm.
    # 
    # The current program uses: 6x6 as 3x3 grid of 2x2 blocks, and applies
    # Strassen 2x2 to each of the 9 (i_b, j_b, k_b) combinations.
    # This gives rank = 9 * 7 = 63? No, the current program has rank 189 = 27 * 7.
    # 27 = 3*3*3 (i_b, j_b, k_b combinations). So it's treating each element
    # of the 3x3x3 tensor separately with Strassen. That's just schoolbook
    # with Strassen within each 2x2.
    
    # OK so the current approach: 6x6 matrices, partition into 2x2 blocks (3x3 grid).
    # For each block position (i_b, j_b) in C, we need sum over k_b of A_block(i_b,k_b) * B_block(k_b,j_b).
    # Each 2x2*2x2 product uses Strassen with 7 multiplications.
    # Total: 3*3*3*7 = 189. This is what the current program does.
    
    # To improve, we could use a better 3x3 block-level algorithm.
    # Laderman's 15-product algorithm for 3x3 would give rank = 9*15 = 135? No.
    # If we use Laderman at the block level (3x3 blocks of 2x2 matrices),
    # we need 15 bilinear forms on the 3x3 block structure.
    # Each bilinear form involves products of 2x2 block matrices.
    # If we use schoolbook for each 2x2*2x2 product within a bilinear form,
    # we get 4 multiplications per block product.
    # But the bilinear forms combine multiple block products...
    
    # Actually, the cleaner approach: 
    # 6x6 = Kronecker product of 3x3 and 2x2.
    # M_6 = M_3 ⊗ M_2 (in terms of tensor decomposition).
    # rank(M_3 ⊗ M_2) ≤ rank(M_3) * rank(M_2).
    # If rank(M_3) = 15 (Laderman) and rank(M_2) = 7 (Strassen), rank ≤ 105.
    # If rank(M_3) = 19 (schoolbook) and rank(M_2) = 4 (schoolbook), rank = 76.
    
    # The Kronecker product structure: if we have decompositions
    # M_3 = sum_{r=1}^{15} U3_r ⊗ V3_r ⊗ W3_r
    # M_2 = sum_{s=1}^{7} U2_s ⊗ V2_s ⊗ W2_s
    # Then M_6 = M_3 ⊗ M_2 = sum_{r,s} (U3_r ⊗ U2_s) ⊗ (V3_r ⊗ V2_s) ⊗ (W3_r ⊗ W2_s)
    # This gives rank = 15*7 = 105.
    
    # But wait, the Kronecker product of matrix multiplication tensors:
    # (A⊗A')*(B⊗B') = (A*B)⊗(A'*B')
    # So M_6x6 is indeed related to M_3x3 and M_2x2 via Kronecker product.
    # The tensor M_6 = M_3 ⊗ M_2 (Kronecker product of tensors).
    # rank(M_3 ⊗ M_2) ≤ rank(M_3) * rank(M_2) = 15 * 7 = 105.
    
    # This would be a significant improvement from 189 to 105!
    
    # Let me implement this. I need:
    # 1. Laderman 3x3 decomposition (rank 15)
    # 2. Strassen 2x2 decomposition (rank 7)
    # 3. Kronecker product of the factor matrices
    
    return U, V, W

def _strassen_2x2_factors():
    """Return U, V, W for Strassen 2x2 matrix multiplication.
    Shapes: (4,7), (4,7), (4,7).
    Index convention: M[i*2+j] for matrix element [i][j].
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # A factors: U[i*2+j]
    # i,j in {0,1}: idx = i*2+j
    # A00=0, A01=1, A10=2, A11=3
    
    U[0][0] = 1  # M1: A00
    U[3][0] = 1  # M1: A11
    U[2][1] = 1  # M2: A10
    U[3][1] = 1  # M2: A11
    U[0][2] = 1  # M3: A00
    U[3][3] = 1  # M4: A11
    U[0][4] = 1  # M5: A00
    U[1][4] = 1  # M5: A01
    U[2][5] = 1  # M6: A10
    U[0][5] = -1 # M6: -A00
    U[1][6] = 1  # M7: A01
    U[3][6] = -1 # M7: -A11
    
    # B factors: V[i*2+j]
    V[0][0] = 1  # M1: B00
    V[1][0] = 1  # M1: B11
    V[0][1] = 1  # M2: B00
    V[0][2] = 1  # M3: B01
    V[1][2] = -1 # M3: -B11
    V[1][3] = 1  # M4: B10
    V[0][3] = -1 # M4: -B00
    V[1][4] = 1  # M5: B11
    V[0][5] = 1  # M6: B00
    V[1][5] = 1  # M6: B01
    V[1][6] = 1  # M7: B10
    V[3//2 if False else 1][6] = 1  # M7: B11 (V[1][6])
    
    # Fix: V indices: B00=0, B01=1, B10=2, B11=3
    V = [[0]*rank for _ in range(4)]
    V[0][0] = 1  # M1: B00
    V[3][0] = 1  # M1: B11
    V[0][1] = 1  # M2: B00
    V[1][2] = 1  # M3: B01
    V[3][2] = -1 # M3: -B11
    V[2][3] = 1  # M4: B10
    V[0][3] = -1 # M4: -B00
    V[3][4] = 1  # M5: B11
    V[0][5] = 1  # M6: B00
    V[1][5] = 1  # M6: B01
    V[2][6] = 1  # M7: B10
    V[3][6] = 1  # M7: B11
    
    # C factors: W[i*2+j]
    # C00=0, C01=1, C10=2, C11=3
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    W = [[0]*rank for _ in range(4)]
    W[0][0] = 1  # M1 -> C00
    W[0][3] = -1 # M4 -> -C00
    W[0][4] = -1 # M5 -> -C00
    W[0][6] = 1  # M7 -> C00
    W[1][2] = 1  # M3 -> C01
    W[1][4] = 1  # M5 -> C01
    W[2][1] = 1  # M2 -> C10
    W[2][3] = 1  # M4 -> C10
    W[3][0] = 1  # M1 -> C11
    W[3][1] = -1 # M2 -> -C11
    W[3][2] = 1  # M3 -> C11
    W[3][5] = -1 # M6 -> -C11
    
    return U, V, W

def _laderrman_3x3_factors():
    """Return U, V, W for Laderman 3x3 matrix multiplication (rank 15).
    Shapes: (9,15), (9,15), (9,15).
    Index convention: M[i*3+j] for matrix element [i][j].
    """
    rank = 15
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Indices: A[i][j] = i*3+j: a00=0,a01=1,a02=2,a10=3,a11=4,a12=5,a20=6,a21=7,a22=8
    # Same for B and C.
    
    # Laderman's 15 products:
    # P1 = a00*b00
    # P2 = (a01+a02)*b00
    # P3 = a00*(b01+b02)
    # P4 = (a10+a11+a20+a21)*(b00+b10+b20+b21)
    # P5 = a02*b11
    # P6 = a22*b22
    # P7 = (a01+a02-a21-a22)*(b21+b22-b01-b02)
    # P8 = (a11+a12+a21+a22)*(b11+b12+b21+b22)
    # P9 = (a10-a20)*(b20+b21)
    # P10 = a20*(b10+b11)
    # P11 = (a10+a11-a20-a21)*(b20+b21-b10-b11)
    # P12 = (a11+a12-a21-a22)*(b21+b22-b11-b12)
    # P13 = a12*(b01+b02)
    # P14 = a01*b22
    # P15 = (a10-a11+a20-a21)*(b10-b11+b20-b21)
    
    # Let me verify this decomposition numerically before using it.
    # For now, I'll construct it and verify.
    
    A_data = [
        [(0,1)],                          # P1
        [(1,1),(2,1)],                    # P2
        [(0,1)],                          # P3
        [(3,1),(4,1),(6,1),(7,1)],       # P4
        [(2,1)],                          # P5
        [(8,1)],                          # P6
        [(1,1),(2,1),(7,-1),(8,-1)],     # P7
        [(4,1),(5,1),(7,1),(8,1)],       # P8
        [(3,1),(6,-1)],                   # P9
        [(6,1)],                          # P10
        [(3,1),(4,1),(6,-1),(7,-1)],     # P11
        [(4,1),(5,1),(7,-1),(8,-1)],     # P12
        [(5,1)],                          # P13
        [(1,1)],                          # P14
        [(3,1),(4,-1),(6,1),(7,-1)],     # P15
    ]
    
    B_data = [
        [(0,1)],                          # P1
        [(0,1)],                          # P2
        [(1,1),(2,1)],                    # P3
        [(0,1),(3,1),(6,1),(7,1)],       # P4
        [(4,1)],                          # P5
        [(8,1)],                          # P6
        [(7,1),(8,1),(1,-1),(2,-1)],     # P7
        [(4,1),(5,1),(7,1),(8,1)],       # P8
        [(6,1),(7,1)],                    # P9
        [(3,1),(4,1)],                    # P10
        [(6,1),(7,1),(3,-1),(4,-1)],     # P11
        [(7,1),(8,1),(4,-1),(5,-1)],     # P12
        [(1,1),(2,1)],                    # P13
        [(8,1)],                          # P14
        [(3,1),(4,-1),(6,1),(7,-1)],     # P15
    ]
    
    # C contributions:
    # c00 = P1 + P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    # c01 = P2 + P3 + P13 + P14 + P7 + P8 - P4 - P12 - P15
    # c02 = P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    # c10 = P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    # c11 = P2 + P3 + P13 + P14 + P7 + P8 - P4 - P12 - P15
    # c12 = P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    # c20 = P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    # c21 = P2 + P3 + P13 + P14 + P7 + P8 - P4 - P12 - P15
    # c22 = P5 + P6 + P11 - P13 - P14 + P15 - P4 + P8
    
    # Hmm, this pattern looks suspicious. Let me just verify numerically.
    # I'll fill in the C contributions and then verify.
    
    c_patterns = [
        [(0,1), (4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],  # c00: P1+P5+P6+P11-P13-P14+P15-P4+P8
        [(1,1), (2,1), (12,1), (13,1), (6,1), (7,1), (3,-1), (11,-1), (14,-1)],  # c01: P2+P3+P13+P14+P7+P8-P4-P12-P15
        [(4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],          # c02
        [(4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],          # c10
        [(1,1), (2,1), (12,1), (13,1), (6,1), (7,1), (3,-1), (11,-1), (14,-1)],  # c11
        [(4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],          # c12
        [(4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],          # c20
        [(1,1), (2,1), (12,1), (13,1), (6,1), (7,1), (3,-1), (11,-1), (14,-1)],  # c21
        [(4,1), (8,1), (10,1), (12,-1), (13,-1), (14,1), (3,-1), (7,1)],          # c22
    ]
    
    for idx, (ci, coeff) in enumerate(c_patterns[0]):
        pass
    
    for ci, coeffs in enumerate(c_patterns):
        for pi, coeff in coeffs:
            W[ci][pi] += coeff
    
    for pi in range(rank):
        for idx, coeff in A_data[pi]:
            U[idx][pi] += coeff
        for idx, coeff in B_data[pi]:
            V[idx][pi] += coeff
    
    return U, V, W

def _verify_decomposition(n, U, V, W, verbose=False):
    """Verify that the decomposition is correct: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the multiplication tensor
    tensor = np.zeros((n*n, n*n, n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a_idx = i*n + k  # A[i][k]
                b_idx = k*n + j  # B[k][j]
                c_idx = i*n + j  # C[i][j]
                tensor[a_idx, b_idx, c_idx] = 1.0
    
    # Compute the decomposition
    recon = np.zeros((n*n, n*n, n*n))
    for r in range(rank):
        recon += np.einsum('i,j,k->ijk', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    error = np.max(np.abs(recon - tensor))
    if verbose:
        print(f"Verification: max error = {error:.2e}")
    return error < 1e-10

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: Use Kronecker product structure.
    # 6 = 3 * 2, so 6x6 matrix mult = (3x3) ⊗ (2x2) in some sense.
    # Specifically, if we write 6x6 matrices as 3x3 blocks of 2x2 matrices,
    # then C = A*B where A, B, C are 6x6 is equivalent to block-matrix multiplication
    # where each block is 2x2.
    
    # The tensor for 6x6 matrix multiplication can be viewed as:
    # M_6 = M_3 ⊗ M_2 (Kronecker product of tensors)
    # where M_3 is the 3x3 multiplication tensor and M_2 is the 2x2 tensor.
    
    # If we have rank-r3 decomposition of M_3 and rank-r2 decomposition of M_2,
    # then we get a rank r3*r2 decomposition of M_6.
    
    # Let me get Strassen 2x2 (rank 7) and try Laderman 3x3 (rank 15).
    # Total rank = 15 * 7 = 105.
    
    # First, let me get Strassen 2x2 factors.
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Verify Strassen 2x2
    err2 = _verify_decomposition(2, U2, V2, W2, verbose=True)
    
    # Now build Laderman 3x3 factors.
    U3, V3, W3 = _laderrman_3x3_factors()
    err3 = _verify_decomposition(3, U3, V3, W3, verbose=True)
    
    if not (err2 and err3):
        # Fallback to schoolbook 6x6 (rank 216) or Strassen-based (rank 189)
        # Use the known-good Strassen 2x2 with 3x3 grid approach (rank 189)
        rank = 189
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        A_combs = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,1,-1)]
        ]
        B_combs = [
            [(0,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (0,0,-1)],
            [(1,1,1)],
            [(0,0,1), (0,1,1)],
            [(1,0,1), (1,1,1)]
        ]
        C_combs = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)]
        ]
        
        block_size = 2
        num_blocks = 3
        
        for i_b in range(num_blocks):
            for j_b in range(num_blocks):
                for k_b in range(num_blocks):
                    for s in range(7):
                        col = ((i_b * num_blocks + j_b) * num_blocks + k_b) * 7 + s
                        for rb, cb, coeff in A_combs[s]:
                            i_glob = i_b * block_size + rb
                            j_glob = k_b * block_size + cb
                            U[i_glob * m + j_glob][col] += coeff
                        for rb, cb, coeff in B_combs[s]:
                            j_glob = k_b * block_size + rb
                            k_glob = j_b * block_size + cb
                            V[j_glob * p + k_glob][col] += coeff
                        for rb, cb, coeff in C_combs[s]:
                            i_glob = i_b * block_size + rb
                            j_glob = j_b * block_size + cb
                            W[j_glob * n + i_glob][col] += coeff
        
        return {"U": U, "V": V, "W": W}
    
    # Kronecker product construction
    # M_6 = M_3 ⊗ M_2
    # U_6 = U_3 ⊗ U_2 (Kronecker product)
    # But we need to be careful about index ordering.
    
    # The 6x6 matrix A can be viewed as a 3x3 block matrix where each block is 2x2.
    # A[i3*2+i2, j3*2+j2] = A_block[i3][j3][i2][j2]
    # In linear indexing: idx6 = (i3*2+i2)*6 + (j3*2+j2)
    # = (i3*2+i2)*6 + j3*2 + j2
    # = i3*12 + i2*6 + j3*2 + j2
    
    # For the Kronecker product of tensors:
    # M_6[a6, b6, c6] = M_3[a3, b3, c3] * M_2[a2, b2, c2]
    # where a6 = a3*2 + a2 (in the appropriate sense)
    
    # Actually, let me think about this more carefully.
    # The 6x6 multiplication tensor T_6[i,j,k] = 1 if j = i+k (in some sense)
    # More precisely: T_6[a_idx, b_idx, c_idx] where a_idx = i*6+k (A[i][k]),
    # b_idx = k*6+j (B[k][j]), c_idx = i*6+j (C[i][j])
    
    # If we write i = i3*2+i2, j = j3*2+j2, k = k3*2+k2:
    # a_idx = (i3*2+i2)*6 + (k3*2+k2) = i3*12 + i2*6 + k3*2 + k2
    # b_idx = (k3*2+k2)*6 + (j3*2+j2) = k3*12 + k2*6 + j3*2 + j2
    # c_idx = (i3*2+i2)*6 + (j3*2+j2) = i3*12 + i2*6 + j3*2 + j2
    
    # The tensor factorizes as:
    # T_6[a_idx, b_idx, c_idx] = T_3[a3_idx, b3_idx, c3_idx] * T_2[a2_idx, b2_idx, c2_idx]
    # where a3_idx = i3*3+k3, b3_idx = k3*3+j3, c3_idx = i3*3+j3
    # and a2_idx = i2*2+k2, b2_idx = k2*2+j2, c2_idx = i2*2+j2
    
    # So the decomposition of M_6 is the Kronecker product of decompositions of M_3 and M_2.
    # If M_3 = sum_r U3_r ⊗ V3_r ⊗ W3_r and M_2 = sum_s U2_s ⊗ V2_s ⊗ W2_s
    # Then M_6 = sum_{r,s} (U3_r ⊗ U2_s) ⊗ (V3_r ⊗ V2_s) ⊗ (W3_r ⊗ W2_s)
    
    rank3 = 15
    rank2 = 7
    rank_total = rank3 * rank2  # 105
    
    U = [[0] * rank_total for _ in range(36)]
    V = [[0] * rank_total for _ in range(36)]
    W = [[0] * rank_total for _ in range(36)]
    
    for r in range(rank3):
        for s in range(rank2):
            col = r * rank2 + s
            # U3_r is a vector of length 9, U2_s is a vector of length 4
            # Kronecker product: (U3_r ⊗ U2_s) has length 36
            # Index: (i3*2+i2)*3... no.
            # a6_idx = i3*12 + i2*6 + k3*2 + k2
            # = (i3*3+k3)*... no, this isn't a simple Kronecker in the linear index.
            
            # Let me map it out:
            # a6 = i3*12 + i2*6 + k3*2 + k2
            # a3 = i3*3 + k3
            # a2 = i2*2 + k2
            # a6 = i3*12 + i2*6 + k3*2 + k2
            #    = (i3*3 + k3)*... let's see:
            #    = i3*(12) + k3*(2) + i2*(6) + k2*(1)
            # a3*a2 would give (i3*3+k3)*(i2*2+k2) = i3*i2*6 + i3*k2*3 + k3*i2*2 + k3*k2
            # That's not the same as a6.
            
            # So the linear indices don't directly Kronecker. I need to be more careful.
            # Let me just construct the factors by iterating over all indices.
            
            for a3_idx in range(9):
                for a2_idx in range(4):
                    # Find i3, k3 from a3_idx
                    i3 = a3_idx // 3
                    k3 = a3_idx % 3
                    # Find i2, k2 from a2_idx
                    i2 = a2_idx // 2
                    k2 = a2_idx % 2
                    a6_idx = i3 * 12 + i2 * 6 + k3 * 2 + k2
                    U[a6_idx][col] += U3[a3_idx][r] * U2[a2_idx][s]
            
            for b3_idx in range(9):
                for b2_idx in range(4):
                    k3 = b3_idx // 3
                    j3 = b3_idx % 3
                    k2 = b2_idx // 2
                    j2 = b2_idx % 2
                    b6_idx = k3 * 12 + k2 * 6 + j3 * 2 + j2
                    V[b6_idx][col] += V3[b3_idx][r] * V2[b2_idx][s]
            
            for c3_idx in range(9):
                for c2_idx in range(4):
                    i3 = c3_idx // 3
                    j3 = c3_idx % 3
                    i2 = c2_idx // 2
                    j2 = c2_idx % 2
                    c6_idx = i3 * 12 + i2 * 6 + j3 * 2 + j2
                    W[c6_idx][col] += W3[c3_idx][r] * W2[c2_idx][s]
    
    # Verify the 6x6 decomposition
    err6 = _verify_decomposition(6, U, V, W, verbose=True)
    if err6:
        return {"U": U, "V": "V", "W": W} if False else {"U": U, "V": V, "W": W}
    
    # Fallback if verification fails
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_combs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    B_combs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]
    C_combs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    block_size = 2
    num_blocks = 3
    
    for i_b in range(num_blocks):
        for j_b in range(num_blocks):
            for k_b in range(num_blocks):
                for s in range(7):
                    col = ((i_b * num_blocks + j_b) * num_blocks + k_b) * 7 + s
                    for rb, cb, coeff in A_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = k_b * block_size + cb
                        U[i_glob * m + j_glob][col] += coeff
                    for rb, cb, coeff in B_combs[s]:
                        j_glob = k_b * block_size + rb
                        k_glob = j_b * block_size + cb
                        V[j_glob * p + k_glob][col] += coeff
                    for rb, cb, coeff in C_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = j_b * block_size + cb
                        W[j_glob * n + i_glob][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
