import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit

    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Build a valid Strassen-based decomposition for 4x4 * 4x7
    # Strategy:
    #   - Use 2x2 Strassen (rank 7) on 2x2 blocks of A and B.
    #   - For the 4x4 * 4x4 part: 4 blocks of 2x2*2x2, each using Strassen => 4*7 = 28 terms.
    #   - For the 4x4 * 4x3 part: split into 4x2 + 4x1.
    #       4x2: two blocks of 2x2*2x1... this gets complicated.
    #
    # Simpler robust approach: use Strassen on 2x2 blocks for the full 4x7.
    # A is 4x4: split into four 2x2 blocks.
    # B is 4x7: split into 2x7 and 2x7 (row-wise), each 2x7 split into 2x4 + 2x3? No.
    #
    # Let's use a different approach: block Strassen with 2x2 blocks.
    # A = [[A11, A12],[A21, A22]] where each is 2x2.
    # B = [[B11, B12],[B21, B22]] where B11 is 2x7, B12 is 2x7, etc.
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],[A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Each 2x2 * 2x7 multiplication can be done with a "Strassen-like" method.
    # Actually, 2x2 * 2x7: we can use the standard 8 multiplications for 2x2*2x2,
    # but since B is 2x7, we treat it as 2x2*2x2 for first 2 cols, 2x2*2x2 for next 2, etc.
    # That gives 4 groups of 8 = 32 per 2x2*2x7 block. Too many.
    #
    # Alternative: use the fact that we only need to beat rank 112.
    # Let's use a known efficient decomposition.
    #
    # For 4x4*4x7, a simple improvement: use Strassen for the 4x4*4x4 part (rank 28)
    # and schoolbook for 4x4*4x3 (rank 48). Total = 76.
    #
    # Let me implement Strassen for 4x4*4x4 carefully.
    # 4x4 = four 2x2 blocks. Use Strassen recursively.
    # Actually, standard Strassen on 4x4 uses 7 multiplications of 2x2 matrices,
    # each of which is 8 scalar multiplications => 56? No.
    # Strassen reduces 2x2*2x2 from 8 to 7 multiplications.
    # For 4x4*4x4: split into 2x2 blocks, use 7 Strassen multiplications of 2x2*2x2,
    # each requiring 8 scalar mults => 7*8 = 56. That's worse than 64? No, 56 < 64.
    # But wait, the Strassen 7 multiplications are of 2x2 matrices, each done with 8 scalar mults.
    # So total = 7*8 = 56. Schoolbook for 4x4*4x4 is 64. So Strassen gives 56.
    #
    # Hmm, but actually for the tensor decomposition, Strassen for 2x2*2x2 has rank 7.
    # For 4x4*4x4, using block Strassen with 2x2 blocks gives rank 7 * (number of 2x2 multiplications)
    # but the 2x2 multiplications themselves need rank 7 each... this is getting complex.
    #
    # Let me just use a verified approach: implement Strassen for 2x2*2x2 (rank 7)
    # and combine for 4x4*4x4.
    #
    # Actually, the cleanest approach: use the known rank-28 Strassen decomposition for 4x4*4x4.
    # This is well-known. Let me build it from 2x2 Strassen blocks.
    #
    # 4x4 * 4x4:
    # A = [[A11, A12],[A21, A22]] (2x2 blocks)
    # B = [[B11, B12],[B21, B22]] (2x2 blocks)
    # C = [[C11, C12],[C21, C22]]
    #
    # Using Strassen at the 2x2 level (7 multiplications of 2x2 matrices):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2*2x2 multiplication, which has rank 7.
    # But we need to expand each Mi into scalar terms.
    #
    # This is getting very complex. Let me use a different strategy.
    #
    # Strategy: Use numpy to construct the tensor and then try to find a low-rank decomposition
    # via randomized SVD or similar. But we need exact integer/half-integer entries.
    #
    # Let me try a simpler hybrid: 
    # For the 4x4*4x7 problem, split B into B1 (4x4) and B2 (4x3).
    # Use schoolbook for both but with some sharing.
    #
    # Actually, the simplest valid improvement: use the fact that C = A*B where A is 4x4 and B is 4x7.
    # We can write B = [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    # C = [A*B1 | A*B2]
    #
    # For A*B1 (4x4*4x4): use Strassen with rank 28.
    # For A*B2 (4x4*4x3): use schoolbook with rank 48.
    # Total rank = 76.
    #
    # Let me implement Strassen for 4x4*4x4 by composing 2x2 Strassen blocks.
    #
    # 2x2 Strassen decomposition (rank 7):
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    #
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a scalar multiplication, so rank 7 for 2x2*2x2.
    #
    # Now for 4x4*4x4, we have 4 blocks of 2x2*2x2.
    # But the Strassen formula at the 2x2-block level requires 7 "multiplications" of 2x2 matrices,
    # each of which is a 2x2*2x2 problem with rank 7.
    # So total rank = 7 * 7 = 49? That seems high.
    #
    # Wait, no. The block Strassen approach:
    # We have 4x4 matrices, split into 2x2 blocks.
    # The Strassen formula gives 7 matrix multiplications of 2x2 matrices.
    # Each 2x2 matrix multiplication has rank 7 (using scalar Strassen).
    # So total rank = 7 * 7 = 49.
    #
    # But actually, we can be smarter. The 7 "multiplications" in block Strassen are not independent
    # in the sense that we can share some of the inner Strassen terms. But for simplicity, 49 is fine
    # if it's correct. 49 + 48 = 97, which is still better than 112.
    #
    # Actually, let me reconsider. The known rank for 4x4*4x4 is at most 49 via block Strassen,
    # but it might be lower. Let's just use 49 + 48 = 97 for now.
    #
    # Hmm, but I want to be sure it's correct. Let me implement it carefully.
    #
    # Actually, let me try a different, simpler approach that I can verify more easily.
    #
    # Approach: Use the "middle product" trick.
    # A is 4x4, B is 4x7.
    # We can write the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's use a decomposition where we factor through a smaller dimension.
    # For example, if we can find U, V, W such that the tensor decomposes with rank < 112.
    #
    # Let me try to use numpy to find a low-rank approximation and then round.
    #
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # Actually, the tensor for matrix multiplication is:
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if they correspond to valid i,j,k, else 0.
    #
    # Let me build this tensor and try to decompose it.

    # Build the multiplication tensor
    N = n * m * p  # 4*4*7 = 112
    # The tensor has indices (i*m+j, j*p+k, k*n+i)
    # Let's build it as a 3D array
    U_dim = n * m  # 16
    V_dim = m * p  # 28
    W_dim = p * n  # 28

    # T[a, b, c] = 1 if there exist i, j, k such that a = i*m+j, b = j*p+k, c = k*n+i
    # Otherwise T[a, b, c] = 0

    # For each valid (i,j,k), set T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.

    # Let's use numpy to try a decomposition.
    # We can try to find a rank-r decomposition by solving for U, V, W.
    # This is a non-convex problem, but we can try randomized methods.

    # Given the complexity, let me just return the schoolbook for now if I can't find a better one quickly.
    # But the task says to improve, so let me try a bit harder.

    # Let me implement the 2x2 Strassen for the 4x4*4x4 part and schoolbook for 4x4*4x3.
    # I'll do it carefully with explicit indices.

    def build_2x2_strassen_terms(i_offset, j_offset, k_offset, col_start):
        """
        Build Strassen terms for a 2x2*2x2 block.
        A block at rows i_offset..i_offset+1, cols j_offset..j_offset+1
        B block at rows j_offset..j_offset+1, cols k_offset..k_offset+1
        C block at rows i_offset..i_offset+1, cols k_offset..k_offset+1
        
        Returns list of (u_indices, v_indices, w_indices, value) for 7 terms.
        """
        # A indices: (i_offset+di, j_offset+dj) for di,dj in {0,1}
        # A[di,dj] = (i_offset+di)*m + (j_offset+dj)
        a = lambda di, dj: (i_offset + di) * m + (j_offset + dj)
        b = lambda di, dj: (j_offset + di) * p + (k_offset + dj)
        c = lambda di, dj: (k_offset + dj) * n + (i_offset + di)
        
        # a00, a01, a10, a11
        a00, a01 = a(0,0), a(0,1)
        a10, a11 = a(1,0), a(1,1)
        
        # b00, b01, b10, b11
        b00, b01 = b(0,0), b(0,1)
        b10, b11 = b(1,0), b(1,1)
        
        # c00, c01, c10, c11
        c00, c01 = c(0,0), c(0,1)
        c10, c11 = c(1,0), c(1,1)
        
        terms = []
        # M1 = (a00+a11)(b00+b11)
        terms.append(([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)], col_start))
        col_start += 1
        # M2 = (a10+a11)(b00)
        terms.append(([(a10, 1), (a11, 1)], [(b00, 1)], [(c00, 1), (c10, 1)], col_start))
        col_start += 1
        # M3 = (a00)(b01-b11)
        terms.append(([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)], col_start))
        col_start += 1
        # M4 = (a11)(b10-b00)
        terms.append(([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)], col_start))
        col_start += 1
        # M5 = (a00+a01)(b11)
        terms.append(([(a00, 1), (a01, 1)], [(b11, 1)], [(c01, 1), (c11, 1)], col_start))
        col_start += 1
        # M6 = (a10-a00)(b00+b01)
        terms.append(([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c01, 1), (c11, 1)], col_start))
        col_start += 1
        # M7 = (a01-a11)(b10+b11)
        terms.append(([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c10, 1), (c11, 1)], col_start))
        col_start += 1
        
        return terms, col_start

    # For 4x4*4x4: use 4 blocks of 2x2*2x2, each with Strassen rank 7. Total = 28.
    # For 4x4*4x3: use schoolbook, rank 48.
    # Total rank = 76.
    
    # Let's build it.
    rank_total = 76
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(28)]
    W = [[0] * rank_total for _ in range(28)]
    
    col = 0
    # 4 blocks of 2x2*2x2 for the 4x4*4x4 part
    for ii in [0, 2]:
        for jj in [0, 2]:
            kk = jj  # B's column offset matches A's column offset for square part
            # Actually, for A*B where B is 4x7, the first 4 columns of B form a 4x4 matrix.
            # So B's first 4 columns: cols 0-3.
            # Block (ii,jj) of A multiplies block (jj, kk) of B where kk = jj for the square part.
            # Wait, B is 4x7. For the 4x4 part, B is 4x4 with cols 0-3.
            # A block (ii, jj) multiplies B block (jj, kk) where kk = jj? No.
            # A is 4x4, B is 4x4 (first part). C is 4x4.
            # A[ii:ii+2, jj:jj+2] * B[jj:jj+2, kk:kk+2] contributes to C[ii:ii+2, kk:kk+2]
            # But in Strassen, we do block operations. Let me just do 4 independent 2x2*2x2 Strassen
            # for the diagonal blocks? No, that's not right.
            #
            # Actually, for the 4x4*4x4 multiplication, we have:
            # C = A * B where both are 4x4.
            # We can split into 2x2 blocks:
            # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
            # C11 = A11*B11 + A12*B21
            # C12 = A11*B12 + A12*B22
            # C21 = A21*B11 + A22*B21
            # C22 = A21*B12 + A22*B22
            #
            # Each of these is a sum of two 2x2*2x2 products.
            # We can use Strassen for each 2x2*2x2 product, giving rank 7 each.
            # So C11 uses 14 terms, C12 uses 14, C21 uses 14, C22 uses 14. Total = 56.
            #
            # That's 56 for the 4x4*4x4 part, plus 48 for 4x4*4x3 = 104. Still better than 112.
            #
            # Let me do that.
            pass
    
    # Let me redo: for 4x4*4x4, use 8 independent 2x2*2x2 Strassen decompositions.
    # C11 = A11*B11 + A12*B21: 2 * 7 = 14 terms
    # C12 = A11*B12 + A12*B22: 2 * 7 = 14 terms
    # C21 = A21*B11 + A22*B21: 2 * 7 = 14 terms
    # C22 = A21*B12 + A22*B22: 2 * 7 = 14 terms
    # Total = 56 for 4x4*4x4.
    # Plus 48 for 4x4*4x3.
    # Total = 104.
    
    rank_total = 56 + 48
    U = [[0] * rank_total for _ in range(16)]
    V = [[0] * rank_total for _ in range(28)]
    W = [[0] * rank_total for _ in range(28)]
    
    col = 0
    # 8 blocks of 2x2*2x2 for 4x4*4x4
    # Block (ai, aj, bi, bj, ci, cj) where ai,aj for A, bi,bj for B, ci,cj for C
    # A rows ai:ai+2, cols aj:aj+2
    # B rows bi:bi+2, cols bj:bj+2
    # C rows ci:ci+2, cols cj:cj+2
    
    blocks_4x4 = [
        # (A_row, A_col, B_row, B_col, C_row, C_col)
        (0, 0, 0, 0, 0, 0),  # A11*B11 -> C11
        (0, 2, 2, 0, 0, 0),  # A12*B21 -> C11
        (0, 0, 0, 2, 0, 2),  # A11*B12 -> C12
        (0, 2, 2, 2, 0, 2),  # A12*B22 -> C12
        (2, 0, 0, 0, 2, 0),  # A21*B11 -> C21
        (2, 2, 2, 0, 2, 0),  # A22*B21 -> C21
        (2, 0, 0, 2, 2, 2),  # A21*B12 -> C22
        (2, 2, 2, 2, 2, 2),  # A22*B22 -> C22
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks_4x4:
        terms, col = build_2x2_strassen_terms(ar, ac, bc, cr, col)
        for (u_idx_vals, v_idx_vals, w_idx_vals, tcol) in terms:
            for (idx, val) in u_idx_vals:
                U[idx][tcol] += val
            for (idx, val) in v_idx_vals:
                V[idx][tcol] += val
            for (idx, val) in w_idx_vals:
                W[idx][tcol] += val
    
    # Now add schoolbook for 4x4*4x3 (columns 4,5,6 of B)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                Bcol = 4 + k
                U[i*4+j][col] = 1
                V[j*7+Bcol][col] = 1
                W[Bcol*4+i][col] = 1
                col += 1
    
    # Verify
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(7):
                total = 0
                for c in range(rank_total):
                    total += U[i*4+j][c] * V[j*7+k][c] * W[k*4+i][c]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END

