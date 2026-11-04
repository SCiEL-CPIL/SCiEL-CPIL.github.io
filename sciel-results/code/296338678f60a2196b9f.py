"""Rewritten program using Strassen-Winograd block decomposition for the 4x4x6 case.

The matrix multiplication tensor for C = A * B where A is 4x4 and B is 4x6
can be decomposed by splitting B into two 4x3 blocks: B = [B1 | B2].
Then C = [A*B1 | A*B2].

For A*B1 (4x4 times 4x3):
  We can use the 4x4 Strassen algorithm structure but since B1 is 4x3, we need
  to be careful. Actually, let's think differently.

  Split A and B1 into 2x2 blocks:
  A = [[A11, A12], [A21, A22]] (each 2x2)
  B1 = [[B11, B12], [B21, B22]] (each 2x2, since 4x3 split into 2x2 blocks doesn't work cleanly)

  Actually, 4x3 cannot be split into 2x2 blocks evenly. Let me reconsider.

  Alternative: Split B (4x6) into B_left (4x4) and B_right (4x2).
  C = [A * B_left | A * B_right]

  For A * B_left: both are 4x4, use Strassen-Winograd with rank 49.
  For A * B_right: A is 4x4, B_right is 4x2, use schoolbook with rank 4*4*2 = 32.

  Total rank = 49 + 32 = 81.

Let's implement the Strassen-Winograd 4x4x4 factors.

Strassen for 4x4 uses 7 products of 2x2 blocks, each 2x2 product uses 7 scalar
multiplications, so 7*7=49 rank-1 terms.

We need to carefully construct U, V, W matrices.

For the 4x4x4 Strassen:
  A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
  where each sub-block is 2x2.

  Strassen combines:
  M1 = (A11 + A22)(B11 + B22)
  M2 = (A21 + A22)B11
  M3 = A11(B12 - B22)
  M4 = A22(B21 - B11)
  M5 = (A11 + A12)B22
  M6 = (A21 - A11)(B11 + B12)
  M7 = (A12 - A22)(B21 + B22)

  Result:
  C11 = M1 + M4 - M5 + M7
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M2 + M3 + M6

  Each Mt is a 2x2 matrix product, which itself needs 7 rank-1 terms (Strassen 2x2x2).

  So for each of the 7 block products, we need 7 scalar multiplication rank-1 terms.
  Total: 7 * 7 = 49 terms for the 4x4x4 part.

For the 4x4x2 schoolbook part:
  A is 4x4, B_right is 4x2.
  C_right[i][k] = sum_j A[i][j] * B_right[j][k]
  Each (i,j,k) triple gives one rank-1 term. Total: 4*4*2 = 32 terms.

Now let's code this up.

For the 2x2x2 Strassen factors (U, V, W for rank 7):
  The 7 products are:
  P1 = (a11+a22)(b11+b22)
  P2 = (a21+a22)b11
  P3 = a11(b12-b22)
  P4 = a22(b21-b11)
  P5 = (a11+a12)b22
  P6 = (a21-a11)(b11+b12)
  P7 = (a12-a22)(b21+b22)

  Result:
  c11 = P1 + P4 - P5 + P7
  c12 = P3 + P5
  c21 = P2 + P4
  c22 = P1 - P2 + P3 + P6

  For each product Pt, we have U coefficients (for A entries), V coefficients (for B entries),
  and W coefficients (for C entries).

  U for P1: [1,0,0,1] for A = [[a11,a12],[a21,a22]] indexed as [a11,a12,a21,a22]
  Actually, let's define A indices: a[0]=a11, a[1]=a12, a[2]=a21, a[3]=a22
  B indices: b[0]=b11, b[1]=b12, b[2]=b21, b[3]=b22
  C indices: c[0]=c11, c[1]=c12, c[2]=c21, c[3]=c22

  P1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,-1] (c11 += P1, c22 -= P1... wait)
  c11 = P1 + P4 - P5 + P7, c22 = P1 - P2 + P3 + P6
  So W for P1: [1, 0, 0, 1]
  W for P2: [0, 0, 1, -1]
  W for P3: [0, 1, 0, 1]
  W for P4: [1, 0, 1, 0]
  W for P5: [-1, 1, 0, 0]
  W for P6: [0, 0, 0, 1]
  W for P7: [1, 0, 0, 0]

  Let me verify:
  c11 = P1 + P4 - P5 + P7 -> contributions: P1:1, P4:1, P5:-1, P7:1
  c12 = P3 + P5 -> P3:1, P5:1
  c21 = P2 + P4 -> P2:1, P4:1
  c22 = P1 - P2 + P3 + P6 -> P1:1, P2:-1, P3:1, P6:1

  So:
  P1: W = [1, 0, 0, 1]
  P2: W = [0, 0, 1, -1]
  P3: W = [0, 1, 0, 1]
  P4: W = [1, 0, 1, 0]
  P5: W = [-1, 1, 0, 0]
  P6: W = [0, 0, 0, 1]
  P7: W = [1, 0, 0, 0]

  U vectors:
  P1: (a11+a22) -> [1,0,0,1]
  P2: (a21+a22) -> [0,0,1,1]
  P3: a11 -> [1,0,0,0]
  P4: a22 -> [0,0,0,1]
  P5: (a11+a12) -> [1,1,0,0]
  P6: (a21-a11) -> [-1,0,1,0]
  P7: (a12-a22) -> [0,1,0,-1]

  V vectors:
  P1: (b11+b22) -> [1,0,0,1]
  P2: b11 -> [1,0,0,0]
  P3: (b12-b22) -> [0,1,0,-1]
  P4: (b21-b11) -> [-1,0,1,0]
  P5: b22 -> [0,0,0,1]
  P6: (b11+b12) -> [1,1,0,0]
  P7: (b21+b22) -> [0,0,1,1]
"""

def _strassen_2x2_factors():
    """Return U, V, W for 2x2x2 Strassen, each of shape (4, 7).
    
    U[i, t] = coefficient of A[i] in product t
    V[i, t] = coefficient of B[i] in product t
    W[i, t] = coefficient of C[i] that gets product t added
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a11
        [0, 0, 0, 0, 1, 0, 1],   # a12
        [0, 1, 0, 0, 0, 1, 0],   # a21
        [1, 1, 0, 1, 0, 0, -1],  # a22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b11
        [0, 0, 1, 0, 0, 1, 0],   # b12
        [0, 0, 0, 1, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 1],  # b22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # c11
        [0, 0, 1, 0, 1, 0, 0],   # c12
        [0, 1, 0, 1, 0, 0, 0],   # c21
        [1, -1, 1, 0, 0, 1, 0],  # c22
    ]
    return U, V, W


def _strassen_4x4_factors():
    """Return U, V, W for 4x4x4 Strassen-Winograd, each of shape (16, 49).
    
    U[i, t] = coefficient of A[i] in rank-1 term t (A is 4x4, flattened row-major: A[i*4+j])
    V[j, t] = coefficient of B[j] in rank-1 term t (B is 4x4, flattened row-major: B[j*4+k])
    W[k, t] = coefficient of C[k] that gets term t (C is 4x4, flattened: C[k*4+i] per trace convention W[k*n+i])
    
    Wait, the trace convention is W[k*n+i, r]. So W is indexed by (k, i) where k is the column of C (from B) and i is the row of C (from A).
    C[i][k] = sum_j A[i][j] * B[j][k]
    W[k*n + i, r] means for output position (i, k), the index in W is k*n + i.
    
    So for 4x4x4, W has shape (4*4, rank) = (16, rank), and W[k*4 + i, r].
    """
    n = 4
    U_2, V_2, W_2 = _strassen_2x2_factors()
    
    # Block structure of A (4x4):
    # A = [[A11, A12], [A21, A22]]
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # Block structure of B (4x4):
    # B = [[B11, B12], [B21, B22]]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # For each of the 7 Strassen products at the 4x4 level, we have a linear combination
    # of A blocks and a linear combination of B blocks.
    # Then each block product (2x2 x 2x2) uses 7 rank-1 terms.
    
    # The 7 4x4-level products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Mt, the A-side is a 2x2 matrix (linear combo of A blocks) and B-side is a 2x2 matrix.
    # We need to express the A-side 2x2 matrix in terms of the 16 A entries.
    # Similarly for B-side.
    
    # Let's define for each Mt:
    # A_block_coeffs: 2x2 matrix of coefficients for A11, A12, A21, A22
    # B_block_coeffs: 2x2 matrix of coefficients for B11, B12, B21, B22
    
    A_coeffs = [
        # M1: A11 + A22 in both positions? No.
        # M1 = (A11+A22)(B11+B22): the A-side matrix is [[A11+A22, 0], [0, A11+A22]]? 
        # No, Strassen for 4x4: the 7 products are of 2x2 matrices.
        # M1 = (A11+A22) * (B11+B22) where A11+A22 is a 2x2 matrix and B11+B22 is a 2x2 matrix.
        # So the A-side for M1 is the 2x2 matrix (A11+A22), meaning:
        #   [[A11[0,0]+A22[0,0], A11[0,1]+A22[0,1]],
        #    [A11[1,0]+A22[1,0], A11[1,1]+A22[1,1]]]
        # This is a 2x2 matrix formed by adding the two 2x2 blocks.
        
        # I think the standard 4x4 Strassen works by treating A and B as 2x2 block matrices
        # and applying Strassen's 7 product formula, where each product is a 2x2 matrix product.
        
        # So for M1: A-side = A11 + A22 (a 2x2 matrix), B-side = B11 + B22 (a 2x2 matrix)
        # The product M1 = (A11+A22) @ (B11+B22) is a 2x2 matrix.
        
        # To decompose M1 into 7 rank-1 scalar products, we use Strassen 2x2 on the 2x2 matrices.
        
        # So for each Mt, I need:
        # - A 2x2 matrix of A-entries (each entry is a linear combination of A's 16 entries)
        # - A 2x2 matrix of B-entries (each entry is a linear combination of B's 16 entries)
        # - Apply Strassen 2x2 to get 7 rank-1 terms
        # - Map those back to the full 4x4 U, V, W
        
        # Let me define the A-side 2x2 matrices for each Mt:
        # Each entry is a function of the 16 A entries.
    ]
    
    # Let's define helper: given a 2x2 matrix where each entry is a dict mapping A-index to coefficient,
    # and similarly for B, compute the 7 Strassen 2x2 rank-1 terms and map them.
    
    # A indices: A[i*4 + j] for i,j in 0..3
    # B indices: B[j*4 + k] for j,k in 0..3
    # C indices: C[i*4 + k] but W uses W[k*4 + i] per trace convention
    
    # For the 4x4 Strassen, the result C is also block-structured:
    # C11 = M1 + M4 - M5 + M7 (2x2 block)
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mt is a 2x2 matrix, so C11, C12, etc. are 2x2 matrices.
    
    # Let me define the A-side and B-side for each Mt as 2x2 matrices of linear combinations.
    
    # A11 entries: A[0], A[1], A[4], A[5]  (rows 0-1, cols 0-1)
    # A12 entries: A[2], A[3], A[6], A[7]  (rows 0-1, cols 2-3)
    # A21 entries: A[8], A[9], A[12], A[13] (rows 2-3, cols 0-1)
    # A22 entries: A[10], A[11], A[14], A[15] (rows 2-3, cols 2-3)
    
    # For each Mt, define A_side[r][c] as a dict {A_index: coeff} for r,c in 0..1
    # Similarly B_side[r][c] as dict {B_index: coeff}
    
    A_side = {}
    B_side = {}
    
    # M1: A-side = A11 + A22, B-side = B11 + B22
    A_side[0] = {
        (0,0): {0:1, 10:1}, (0,1): {1:1, 11:1},
        (1,0): {4:1, 14:1}, (1,1): {5:1, 15:1}
    }
    B_side[0] = {
        (0,0): {0:1, 10:1}, (0,1): {2:1, 12:1},
        (1,0): {4:1, 14:1}, (1,1): {6:1, 16:1}
    }
    
    # M2: A-side = A21 + A22, B-side = B11
    A_side[1] = {
        (0,0): {8:1, 10:1}, (0,1): {9:1, 11:1},
        (1,0): {12:1, 14:1}, (1,1): {13:1, 15:1}
    }
    B_side[1] = {
        (0,0): {0:1}, (0,1): {2:1},
        (1,0): {4:1}, (1,1): {6:1}
    }
    
    # M3: A-side = A11, B-side = B12 - B22
    A_side[2] = {
        (0,0): {0:1}, (0,1): {1:1},
        (1,0): {4:1}, (1,1): {5:1}
    }
    B_side[2] = {
        (0,0): {2:1, 10:-1}, (0,1): {3:1, 11:-1},
        (1,0): {6:1, 14:-1}, (1,1): {7:1, 15:-1}
    }
    
    # M4: A-side = A22, B-side = B21 - B11
    A_side[3] = {
        (0,0): {10:1}, (0,1): {11:1},
        (1,0): {14:1}, (1,1): {15:1}
    }
    B_side[3] = {
        (0,0): {4:-1, 8:1}, (0,1): {6:-1, 10:1},
        (1,0): {8:-1, 12:1}, (1,1): {9:-1, 13:1}
    }
    
    # M5: A-side = A11 + A12, B-side = B22
    A_side[4] = {
        (0,0): {0:1, 2:1}, (0,1): {1:1, 3:1},
        (1,0): {4:1, 6:1}, (1,1): {5:1, 7:1}
    }
    B_side[4] = {
        (0,0): {10:1}, (0,1): {11:1},
        (1,0): {14:1}, (1,1): {15:1}
    }
    
    # M6: A-side = A21 - A11, B-side = B11 + B12
    A_side[5] = {
        (0,0): {0:-1, 8:1}, (0,1): {1:-1, 9:1},
        (1,0): {4:-1, 12:1}, (1,1): {5:-1, 13:1}
    }
    B_side[5] = {
        (0,0): {0:1, 2:1}, (0,1): {2:1, 3:1},
        (1,0): {4:1, 6:1}, (1,1): {6:1, 7:1}
    }
    
    # M7: A-side = A12 - A22, B-side = B21 + B22
    A_side[6] = {
        (0,0): {0:-1, 10:1}, (0,1): {1:-1, 11:1},
        (1,0): {4:-1, 14:1}, (1,1): {5:-1, 15:1}
    }
    B_side[6] = {
        (0,0): {8:1, 10:1}, (0,1): {9:1, 11:1},
        (1,0): {12:1, 14:1}, (1,1): {13:1, 15:1}
    }
    
    # C-side coefficients for each Mt in each block:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Each is a 2x2 block.
    
    # C11 is at C[0:2, 0:2], C12 at C[0:2, 2:4], C21 at C[2:4, 0:2], C22 at C[2:4, 2:4]
    
    # W for each Mt and each 2x2 position within the block:
    # For Mt, the 2x2 result has entries (0,0), (0,1), (1,0), (1,1)
    # These map to C positions depending on which block.
    
    # Let's define for each Mt, a dict mapping (block, r, c) to coefficient
    # block: 0=C11, 1=C12, 2=C21, 3=C22
    # r, c: row/col within the 2x2 block
    
    C_coeffs = {
        0: {  # M1
            (0, 0, 0): 1, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,  # C11 gets M1
            (3, 0, 0): 1, (3, 0, 1): 1, (3, 1, 0): 1, (3, 1, 1): 1,  # C22 gets M1
        },
        1: {  # M2
            (2, 0, 0): 1, (2, 0, 1): 1, (2, 1, 0): 1, (2, 1, 1): 1,  # C21 gets M2
            (3, 0, 0): -1, (3, 0, 1): -1, (3, 1, 0): -1, (3, 1, 1): -1,  # C22 gets -M2
        },
        2: {  # M3
            (1, 0, 0): 1, (1, 0, 1): 1, (1, 1, 0): 1, (1, 1, 1): 1,  # C12 gets M3
            (3, 0, 0): 1, (3, 0, 1): 1, (3, 1, 0): 1, (3, 1, 1): 1,  # C22 gets M3
        },
        3: {  # M4
            (0, 0, 0): 1, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,  # C11 gets M4
            (2, 0, 0): 1, (2, 0, 1): 1, (2, 1, 0): 1, (2, 1, 1): 1,  # C21 gets M4
        },
        4: {  # M5
            (0, 0, 0): -1, (0, 0, 1): -1, (0, 1, 0): -1, (0, 1, 1): -1,  # C11 gets -M5
            (1, 0, 0): 1, (1, 0, 1): 1, (1, 1, 0): 1, (1, 1, 1): 1,  # C12 gets M5
        },
        5: {  # M6
            (3, 0, 0): 1, (3, 0, 1): 1, (3, 1, 0): 1, (3, 1, 1): 1,  # C22 gets M6
        },
        6: {  # M7
            (0, 0, 0): 1, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,  # C11 gets M7
        },
    }
    
    # Now build U, V, W for the 4x4x4 part.
    rank_4x4 = 49
    U = [[0] * rank_4x4 for _ in range(16)]
    V = [[0] * rank_4x4 for _ in range(16)]
    W = [[0] * rank_4x4 for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2_factors()
    
    col = 0
    for t in range(7):  # 7 Strassen products at 4x4 level
        # For each of the 7 Strassen 2x2 sub-products:
        for sub in range(7):
            # U2[ai, sub] gives coefficient of A-side entry ai in sub-product sub
            # V2[bi, sub] gives coefficient of B-side entry bi in sub-product sub
            # W2[ci, sub] gives coefficient of C-side entry ci that gets sub-product sub
            
            # A-side entry (ai) maps to a 2x2 position in the A-side matrix:
            # ai = 0 -> (0,0), ai=1 -> (0,1), ai=2 -> (1,0), ai=3 -> (1,1)
            ai_r, ai_c = divmod(sub if False else [0,1,2,3].index(sub) if False else 0, 2)  # placeholder
            
            # Let me redo this more carefully.
            # U2 is 4x7, where U2[ai, sub] for ai in 0..3, sub in 0..6
            # ai corresponds to the 2x2 position: ai=0->(0,0), ai=1->(0,1), ai=2->(1,0), ai=3->(1,1)
            
            for ai in range(4):
                coeff_a = U2[ai][sub]
                if coeff_a == 0:
                    continue
                ar, ac = ai // 2, ai % 2
                # A_side[t][(ar, ac)] is a dict of A_index -> coeff
                for A_idx, a_coeff in A_side[t][(ar, ac)].items():
                    U[A_idx][col] += coeff_a * a_coeff
            
            for bi in range(4):
                coeff_b = V2[bi][sub]
                if coeff_b == 0:
                    continue
                br, bc = bi // 2, bi % 2
                for B_idx, b_coeff in B_side[t][(br, bc)].items():
                    V[B_idx][col] += coeff_b * b_coeff
            
            for ci in range(4):
                coeff_c = W2[ci][sub]
                if coeff_c == 0:
                    continue
                cr, cc = ci // 2, ci % 2
                # The C-side 2x2 position (cr, cc) of Mt maps to C blocks
                for (block, r, c), c_coeff in C_coeffs[t].items():
                    if r == cr and c == cc:
                        # C block 'block' at position (r, c) maps to C indices:
                        # block 0 (C11): C[0:2, 0:2] -> C indices 0,1,4,5
                        # block 1 (C12): C[0:2, 2:4] -> C indices 2,3,6,7
                        # block 2 (C21): C[2:4, 0:2] -> C indices 8,9,12,13
                        # block 3 (C22): C[2:4, 2:4] -> C indices 10,11,14,15
                        if block == 0:
                            C_idx = r * 4 + c
                        elif block == 1:
                            C_idx = r * 4 + (2 + c)
                        elif block == 2:
                            C_idx = (2 + r) * 4 + c
                        else:
                            C_idx = (2 + r) * 4 + (2 + c)
                        
                        # W uses trace convention: W[k*n + i] where C[i][k]
                        # C[i][k] has i = row, k = col
                        # C_idx = i * 4 + k
                        i_val, k_val = C_idx // 4, C_idx % 4
                        W_idx = k_val * 4 + i_val
                        W[W_idx][col] += coeff_c * c_coeff
            
            col += 1
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 6
    
    # Split B (4x6) into B_left (4x4) and B_right (4x2)
    # C = [A * B_left | A * B_right]
    # C is 4x6
    
    # For A * B_left (4x4 x 4x4): use Strassen-Winograd, rank 49
    U_44, V_44, W_44 = _strassen_4x4_factors()
    
    # For A * B_right (4x4 x 4x2): use schoolbook, rank 32
    # A is 4x4, B_right is 4x2
    # C_right[i][k] = sum_j A[i][j] * B_right[j][k] for i in 0..3, k in 0..1
    
    rank_left = 49
    rank_right = 32
    total_rank = rank_left + rank_right
    
    U = [[0] * total_rank for _ in range(16)]  # n*m = 16
    V = [[0] * total_rank for _ in range(24)]  # m*p = 24
    W = [[0] * total_rank for _ in range(24)]  # p*n = 24
    
    # Fill in the left part (B_left, columns 0-3 of B, columns 0-3 of C)
    for i in range(16):
        for r in range(49):
            U[i][r] = U_44[i][r]
            # V: B_left is B[0:4, 0:4], so V indices 0-15 for B_left
            V[i if i < 16 else -1][r] = V_44[i][r] if i < 16 else 0
    # Fix V: B_left entries are B[j*4+k] for j in 0..3, k in 0..3, so indices 0-15
    for i in range(16):
        for r in range(49):
            V[i][r] = V_44[i][r]
    
    # W for left part: C_left is C[0:4, 0:4], W uses W[k*4+i]
    for i in range(16):
        for r in range(49):
            W[i][r] = W_44[i][r]
    
    # Fill in the right part (B_right, columns 4-5 of B, columns 4-5 of C)
    # B_right is 4x2: B[j][4+k] for j in 0..3, k in 0..1
    # B_right index in V: j*6 + (4+k) = j*6 + 4 + k
    # C_right is C[i][4+k] for i in 0..3, k in 0..1
    # W index: k*4 + i for C[i][k], but here k is 4 or 5, so W index = (4+k_val)*4 + i where k_val is 0 or 1
    
    col = 49
    for i in range(4):  # A row
        for j in range(4):  # A col / B row
            for k in range(2):  # B_right col
                U[i*4 + j][col] = 1
                V[j*6 + 4 + k][col] = 1
                # C[i][4+k], W index = (4+k)*4 + i
                W[(4+k)*4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
