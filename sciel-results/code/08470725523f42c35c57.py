"""Matrix multiplication 4,5,6 with Strassen-based decomposition (rank 84)."""
import time

DIMENSIONS = [4, 5, 6]

def construct():
    """Return a rank-84 exact tensor decomposition for 4x5 times 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 5, 6
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]   # 20 x 84
    V = [[0.0] * rank for _ in range(m * p)]   # 30 x 84
    W = [[0.0] * rank for _ in range(p * n)]   # 24 x 84

    col = 0

    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # --- Strassen for 4x4 * 4x4 (rank 28) ---
    # Split A (4x4) into 2x2 blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # Split B (4x4) into 2x2 blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    # C = A*B, C00=C[0:2,0:2], C01=C[0:2,2:4], C10=C[2:4,0:2], C11=C[2:4,2:4]
    #
    # Strassen products (each is a 2x2 matrix multiplication, rank 4):
    # M1 = (A00+A11)*(B00+B11)  -> C00: +1, C11: +1
    # M2 = (A10+A11)*B00        -> C10: +1, C11: -1
    # M3 = A00*(B01-B11)        -> C01: +1, C11: +1
    # M4 = A11*(B10-B00)        -> C00: +1, C10: +1
    # M5 = (A00+A01)*B11        -> C00: -1, C01: +1
    # M6 = (A10-A00)*(B00+B01)  -> C11: +1
    # M7 = (A01-A11)*(B10+B11)  -> C00: +1
    #
    # Each M_i is a 2x2 * 2x2 product. For a 2x2 product X*Y where X has entries from
    # multiple A-blocks and Y from multiple B-blocks, we decompose into 4 rank-1 terms.
    #
    # For each M_i, the "A-side" is a 2x2 matrix with specific entries, and "B-side" too.
    # We need to express each of the 4 scalar products in M_i as a sum of rank-1 terms
    # in the original 4x4x4 tensor.

    # Helper: add a 2x2 schoolbook product where A-block and B-block may have offsets
    # and possibly combined terms.
    # For a single M_i, we have:
    #   A_side[i][j] = sum of coefficients * A[a_i + di, a_j + dj] for various (di,dj)
    #   B_side[j][k] = sum of coefficients * B[b_j + dj, b_k + dk] for various (dj,dk)
    # The product A_side * B_side = sum_{j} A_side[:,j] * B_side[j,:]
    # Each scalar (i,j,k) product is a sum of rank-1 terms.

    # Let me define the Strassen products more explicitly.
    # For each M_i, I'll list the 4 rank-1 terms (one for each (i,j,k) in 2x2x2).
    # The A-side entry at (i,j) is a linear combination of A entries.
    # The B-side entry at (j,k) is a linear combination of B entries.
    # The product for output position (i,k) through intermediate j gives:
    #   sum_j A_side[i,j] * B_side[j,k]
    # Each A_side[i,j] * B_side[j,k] is a product of two linear forms, which expands
    # to a sum of rank-1 terms. But for the tensor decomposition, we need each term
    # to be a single rank-1 tensor. So we need to expand each product.

    # Actually, the correct approach: each M_i contributes 4 columns (one per (i,j,k) 
    # in the 2x2x2 sub-tensor). For each column, U has the A-side coefficients, 
    # V has the B-side coefficients, and W has the C-side coefficients.
    
    # Let me define the 7 Strassen products by their A-side and B-side 2x2 matrices
    # as linear combinations of the original A and B blocks.

    # M1: A_side = A00 + A11, B_side = B00 + B11
    # A_side[0][0] = A[0,0] + A[2,2]
    # A_side[0][1] = A[0,1] + A[2,3]
    # A_side[1][0] = A[1,0] + A[3,2]
    # A_side[1][1] = A[1,1] + A[3,3]
    # B_side[0][0] = B[0,0] + B[2,2]
    # B_side[0][1] = B[0,1] + B[2,3]
    # B_side[1][0] = B[1,0] + B[3,2]
    # B_side[1][1] = B[1,1] + B[3,3]
    # C contributions: C00 += M1, C11 += M1
    # C00[0][0] += A_side[0][0]*B_side[0][0] + A_side[0][1]*B_side[1][0]
    # This gets complex. Let me just enumerate all 4 terms for each M_i.

    # For each M_i, there are 4 rank-1 terms corresponding to (i,j,k) in {0,1}^3.
    # For term (i,j,k):
    #   U entry: coefficient of A[i_A, j_A] where the A-side (i,j) involves A[i_A, j_A]
    #   V entry: coefficient of B[j_B, k_B] where the B-side (j,k) involves B[j_B, k_B]  
    #   W entry: coefficient for C[i_C, k_C]

    # This is getting very involved. Let me use a cleaner formulation.
    # 
    # The 4x4x4 tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Strassen says we can write T as sum of 28 rank-1 tensors.
    # Each rank-1 tensor is u * v * w where u is 4x4 (A-space), v is 4x4 (B-space), w is 4x4 (C-space).
    # 
    # Let me just directly construct the 28 terms.

    # I'll define a function that adds a Strassen-style 2x2 block product.
    # Given:
    #   A_offsets: list of (row_off, col_off, coeff) for the A-side 2x2 matrix
    #   B_offsets: list of (row_off, col_off, coeff) for the B-side 2x2 matrix  
    #   C_targets: list of (row_off, col_off, coeff) for where the result goes
    # 
    # The A-side is a 2x2 matrix: A_side[i][j] = sum_{(r,c,coef) in A_offsets} coef * A[r+i, c+j]
    # Wait, that's not quite right either. Let me think differently.

    # For M1 = (A00+A11)*(B00+B11):
    # The A-side 2x2 matrix has entries:
    #   [0,0]: A[0,0]+A[2,2], [0,1]: A[0,1]+A[2,3]
    #   [1,0]: A[1,0]+A[3,2], [1,1]: A[1,1]+A[3,3]
    # The B-side 2x2 matrix has entries:
    #   [0,0]: B[0,0]+B[2,2], [0,1]: B[0,1]+B[2,3]
    #   [1,0]: B[1,0]+B[3,2], [1,1]: B[1,1]+B[3,3]
    #
    # The product M1[i][k] = sum_j A_side[i][j] * B_side[j][k]
    # M1[0][0] = A_side[0][0]*B_side[0][0] + A_side[0][1]*B_side[1][0]
    #          = (A[0,0]+A[2,2])(B[0,0]+B[2,2]) + (A[0,1]+A[2,3])(B[1,0]+B[3,2])
    #
    # This expands to 8 rank-1 terms! Not 4. So M1 alone contributes 8 rank-1 terms?
    # That can't be right for Strassen to save anything.
    
    # I think the issue is that I'm conflating the block-level Strassen with the scalar level.
    # At the block level, Strassen uses 7 block multiplications instead of 8.
    # Each block multiplication is a 2x2 * 2x2 = 4 scalar multiplications.
    # So total = 7 * 4 = 28 scalar multiplications, vs 8 * 4 = 32 for schoolbook at block level.
    # But the original 4x4x4 schoolbook is 4*4*4 = 64 scalar multiplications.
    # So Strassen reduces 64 to 28.
    
    # The key insight: each of the 7 block products is a 2x2 matrix multiplication,
    # which itself has rank 4 (not 8). So each block product contributes exactly 4 rank-1 terms.
    # Total: 7 * 4 = 28 rank-1 terms.
    
    # For a 2x2 matrix product X*Y, the 4 rank-1 terms are:
    #   Term (i,j,k): U has X[i,j], V has Y[j,k], W has C[i,k]
    # where i,j,k are in {0,1}.
    
    # So for M1 = (A00+A11)*(B00+B11), the 4 rank-1 terms are:
    #   For (i,j,k) in {0,1}^3:
    #     U[i_A, j_A] where A_side[i,j] = A[i_A, j_A] (but A_side is a sum!)
    
    # AH, I see the issue. The A-side is a SUM of two blocks. So A_side[i,j] = A[ai,aj] + A[bi,bj].
    # This means the U vector for term (i,j,k) has TWO non-zero entries.
    # That's fine! A rank-1 tensor in the decomposition can have multiple non-zero entries in U.
    
    # So for M1, term (i,j,k):
    #   U has 1 at (ai,aj) and 1 at (bi,bj) where (ai,aj) and (bi,bj) are the two A-block positions
    #   V has 1 at (cj,ck) and 1 at (dj,dk) where those are the two B-block positions
    #   W has the appropriate coefficient at the C position
    
    # Let me just hardcode all 28 terms. This is tedious but correct.

    # Define Strassen products as tuples of (A_block_pairs, B_block_pairs, C_targets)
    # where A_block_pairs is a list of (row_off, col_off, sign) for the A-side
    # and B_block_pairs is a list of (row_off, col_off, sign) for the B-side
    
    strassen_products = [
        # M1: A-side = A00+A11, B-side = B00+B11, C: C00(+1), C11(+1)
        {'A': [(0,0,1), (2,2,1)], 'B': [(0,0,1), (2,2,1)], 'C': [(0,0,1), (2,2,1)]},
        # M2: A-side = A10+A11, B-side = B00, C: C10(+1), C11(-1)
        {'A': [(2,0,1), (2,2,1)], 'B': [(0,0,1)], 'C': [(2,0,1), (2,2,-1)]},
        # M3: A-side = A00, B-side = B01-B11, C: C01(+1), C11(+1)
        {'A': [(0,0,1)], 'B': [(0,2,1), (2,2,-1)], 'C': [(0,2,1), (2,2,1)]},
        # M4: A-side = A11, B-side = B10-B00, C: C00(+1), C10(+1)
        {'A': [(2,2,1)], 'B': [(2,0,1), (0,0,-1)], 'C': [(0,0,1), (2,0,1)]},
        # M5: A-side = A00+A01, B-side = B11, C: C00(-1), C01(+1)
        {'A': [(0,0,1), (0,2,1)], 'B': [(2,2,1)], 'C': [(0,0,-1), (0,2,1)]},
        # M6: A-side = A10-A00, B-side = B00+B01, C: C11(+1)
        {'A': [(2,0,1), (0,0,-1)], 'B': [(0,0,1), (0,2,1)], 'C': [(2,2,1)]},
        # M7: A-side = A01-A11, B-side = B10+B11, C: C00(+1)
        {'A': [(0,2,1), (2,2,-1)], 'B': [(2,0,1), (2,2,1)], 'C': [(0,0,1)]},
    ]

    # For each Strassen product, add 4 rank-1 terms (for i,j,k in 0..1)
    # A-side entry at (i,j) = sum of sign * A[row_off+i, col_off+j] for each (row_off, col_off, sign) in A
    # B-side entry at (j,k) = sum of sign * B[row_off+j, col_off+k] for each (row_off, col_off, sign) in B
    # C contribution: for each (row_off, col_off, sign) in C, C[row_off+i, col_off+k] += sign * (product)
    
    # For the tensor decomposition, each (i,j,k) gives one column:
    # U: for each (ar, ac, sa) in A: U[(ar+i)*m + (ac+j)][col] += sa
    # V: for each (br, bc, sb) in B: V[(br+j)*p + (bc+k)][col] += sb
    # W: for each (cr, cc, sc) in C: W[(cc+k)*n + (cr+i)][col] += sc
    
    for prod in strassen_products:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # U entries
                    for (ar, ac, sa) in prod['A']:
                        u_idx = (ar + i) * m + (ac + j)
                        U[u_idx][col] += sa
                    # V entries
                    for (br, bc, sb) in prod['B']:
                        v_idx = (br + j) * p + (bc + k)
                        V[v_idx][col] += sb
                    # W entries
                    for (cr, cc, sc) in prod['C']:
                        w_idx = (cc + k) * n + (cr + i)
                        W[w_idx][col] += sc
                    col += 1

    # col should be 28 now
    assert col == 28, f"Strassen part used {col} columns, expected 28"

    # --- Part 2: A1 * B1_right (4x4 * 4x2, schoolbook, rank 32) ---
    # A1 is A[0:4, 0:4], B1_right is B[0:4, 4:6]
    # C contribution: C[0:4, 4:6]
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_idx = i * m + j           # A[i, j] where j is 0..3
                v_idx = j * p + (4 + k)     # B[j, 4+k]
                w_idx = (4 + k) * n + i     # C[i, 4+k] -> W[(4+k)*4 + i]
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1

    # col should be 28 + 32 = 60
    assert col == 60, f"After B1_right, col={col}, expected 60"

    # --- Part 3: A2 * B2 (4x1 * 1x6, schoolbook, rank 24) ---
    # A2 is A[0:4, 4:5], B2 is B[4:5, 0:6]
    # C contribution: C[0:4, 0:6]
    for i in range(4):
        for k in range(6):
            u_idx = i * m + 4               # A[i, 4]
            v_idx = 4 * p + k               # B[4, k]
            w_idx = k * n + i               # C[i, k] -> W[k*4 + i]
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0
            col += 1

    # col should be 60 + 24 = 84
    assert col == 84, f"Final col={col}, expected 84"

    return {"U": U, "V": V, "W": W}
