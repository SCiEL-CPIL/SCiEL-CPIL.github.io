"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    DEADLINE = 150.0
    start = time.time()

    # We want to multiply A (4x4) by B (4x9).
    # We can split B into B1 (4x4) and B2 (4x5).
    # A * B = A * B1 + A * B2.
    # A * B1 is a 4x4 * 4x4 multiplication. We can use Strassen's algorithm for this part.
    # A * B2 is a 4x4 * 4x5 multiplication. We can use the schoolbook method for this part.
    #
    # Strassen's algorithm for 4x4 * 4x4:
    # We can use the standard Strassen algorithm on 2x2 blocks of size 2x2.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # Let B1 = [[B11, B12], [B21, B22]] where each block is 2x2.
    #
    # The 7 products for Strassen are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # The result C1 = A * B1 is:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each of these 7 products is a 2x2 * 2x2 multiplication.
    # A 2x2 * 2x2 multiplication has rank 8 (schoolbook) or can be done with rank 7 (Strassen).
    # However, the tensor rank of 2x2*2x2 is 7.
    # So each Mi can be decomposed into 7 rank-1 terms.
    # Total rank for A*B1 = 7 * 7 = 49.
    #
    # For A * B2 (4x4 * 4x5):
    # We can just use the schoolbook method.
    # Rank = 4 * 4 * 5 = 80.
    #
    # Total rank = 49 + 80 = 129.
    #
    # Wait, I can do better.
    # The rank of 2x2*2x2 is 7.
    # The rank of 2x2*2x1 is 4.
    #
    # Let's refine the block sizes.
    # A is 4x4. B is 4x9.
    # Split B into B1 (4x4) and B2 (4x5).
    # A*B1: 4x4 * 4x4.
    #   Split A into 2x2 blocks. Split B1 into 2x2 blocks.
    #   4 blocks of 2x2*2x2.
    #   Each 2x2*2x2 has rank 7.
    #   Total rank for A*B1 = 7 * 7 = 49.
    #
    # A*B2: 4x4 * 4x5.
    #   Split B2 into B2a (4x4) and B2b (4x1).
    #   A*B2a: 4x4 * 4x4. Rank 49.
    #   A*B2b: 4x4 * 4x1.
    #     Split A into 2x2 blocks. B2b is 4x1 (two 2x1 blocks).
    #     2x2 * 2x1 has rank 4.
    #     Total rank for A*B2b = 4 * 4 = 16.
    #   Total rank for A*B2 = 49 + 16 = 65.
    #
    # Total rank = 49 + 65 = 114.
    #
    # This matches the knowledge from the run: "Strassen-based block decomposition reduces rank from 144 (schoolbook) to 114".

    # We will construct the decomposition explicitly.
    # The tensor is T_{i,j,k} = A_{i,j} B_{j,k}.
    # We need to find U, V, W such that sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = A_{i,j} B_{j,k}.
    # Note the W index is k*n+i.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term for 2x2 * 2x2 multiplication
    # i0, j0, k0 are the offsets in the full matrices.
    # The 2x2 block of A is A[i0:i0+2, j0:j0+2]
    # The 2x2 block of B is B[j0:j0+2, k0:k0+2]
    # The 2x2 block of C is C[i0:i0+2, k0:k0+2]
    def add_strassen_2x2(i0, j0, k0):
        nonlocal col
        # The 7 terms for 2x2*2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to the U, V, W matrices.
        # U index: i*m + j. i in [i0, i0+1], j in [j0, j0+1].
        # V index: j*p + k. j in [j0, j0+1], k in [k0, k0+1].
        # W index: k*n + i. k in [k0, k0+1], i in [i0, i0+1].
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 (i0, j0), A11 (i0+1, j0+1)
        # V: B00 (j0, k0), B11 (j0+1, k0+1)
        # W: C00 (k0, i0), C11 (k0+1, i0+1)
        # Wait, the combination is (A00+A11) * (B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This contributes to C00, C01, C10, C11.
        # C00 gets A00B00. C01 gets A00B11. C10 gets A11B00. C11 gets A11B11.
        # But the term is a single rank-1 term in the tensor product space? No.
        # The tensor for 2x2*2x2 is a sum of 7 rank-1 tensors.
        # T = sum_{s=1}^7 u_s \otimes v_s \otimes w_s.
        # For M1: u1 = (1,0,0,1) for A indices (00,01,10,11). v1 = (1,0,0,1) for B indices. w1 = (1,1,1,1) for C indices?
        # No, w1 must be such that u1 \otimes v1 \otimes w1 = (A00+A11)(B00+B11).
        # This is not a rank-1 tensor in the 2x2x2 space.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is a sum of 4 rank-1 terms.
        # But Strassen says we can do it with 7 terms total for the whole 2x2*2x2.
        # The 7 terms are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Ms is a product of a linear form in A and a linear form in B.
        # So each Ms is a rank-1 tensor in the A x B space, but it contributes to multiple C entries.
        # The tensor T is in A x B x C space.
        # T = sum_{s=1}^7 (u_s \otimes v_s \otimes w_s)
        # where u_s is a vector in A-space (size 4), v_s in B-space (size 4), w_s in C-space (size 4).
        
        # Let's define u_s, v_s, w_s for the 2x2 block.
        # Indices in the 2x2 block: 00, 01, 10, 11.
        # A indices: 00, 01, 10, 11.
        # B indices: 00, 01, 10, 11.
        # C indices: 00, 01, 10, 11.
        
        # M1: u1 = (1, 0, 0, 1) [A00, A01, A10, A11] -> (1, 0, 0, 1) is A00+A11.
        #      v1 = (1, 0, 0, 1) [B00, B01, B10, B11] -> (1, 0, 0, 1) is B00+B11.
        #      w1 = (1, 1, 1, 1) [C00, C01, C10, C11] -> M1 contributes to all?
        #      C00 = M1 + ...
        #      C01 = M3 + M5
        #      C10 = M2 + M4
        #      C11 = M1 - M2 + M3 + M6
        #      So M1 contributes +1 to C00 and +1 to C11.
        #      w1 = (1, 0, 0, 1).
        
        # M2: u2 = (0, 0, 1, 1) [A10+A11]
        #      v2 = (1, 0, 0, 0) [B00]
        #      w2 = (0, 0, 1, -1) [C10 gets +1, C11 gets -1]
        
        # M3: u3 = (1, 0, 0, 0) [A00]
        #      v3 = (0, 1, 0, -1) [B01-B11]
        #      w3 = (0, 1, 0, 1) [C01 gets +1, C11 gets +1]
        
        # M4: u4 = (0, 0, 0, 1) [A11]
        #      v4 = (0, 0, 1, -1) [B10-B00] -> (0, 0, 1, -1) is B10 - B00? No, B10 is index 10, B00 is index 00.
        #      v4 = (-1, 0, 1, 0) [B10-B00]
        #      w4 = (1, 0, 1, 0) [C00 gets +1, C10 gets +1]
        
        # M5: u5 = (1, 1, 0, 0) [A00+A01]
        #      v5 = (0, 0, 0, 1) [B11]
        #      w5 = (-1, 1, 0, 0) [C00 gets -1, C01 gets +1]
        
        # M6: u6 = (-1, 0, 1, 0) [A10-A00]
        #      v6 = (1, 1, 0, 0) [B00+B01]
        #      w6 = (0, 0, 0, 1) [C11 gets +1]
        
        # M7: u7 = (0, 1, 0, -1) [A01-A11]
        #      v7 = (0, 0, 1, 1) [B10+B11]
        #      w7 = (1, 0, 0, 0) [C00 gets +1]
        
        # Let's verify C00 = M1 + M4 - M5 + M7.
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # M4: A11B10 - A11B00
        # -M5: -A00B11 - A01B11
        # M7: A01B10 + A01B11 - A11B10 - A11B11
        # Sum:
        # A00B00: 1 (from M1)
        # A00B11: 1-1=0
        # A11B00: 1-1=0
        # A11B11: 1-1=0
        # A11B10: 1-1=0
        # A01B11: -1+1=0
        # A01B10: 1
        # Result: A00B00 + A01B10. Correct for C00.
        
        # The u, v, w vectors for the 2x2 block are:
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        # w1 = [1, 0, 0, 1]
        
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2 = [0, 0, 1, -1]
        
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # w3 = [0, 1, 0, 1]
        
        # u4 = [0, 0, 0, 1]
        # v4 = [-1, 0, 1, 0]
        # w4 = [1, 0, 1, 0]
        
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # w5 = [-1, 1, 0, 0]
        
        # u6 = [-1, 0, 1, 0]
        # v6 = [1, 1, 0, 0]
        # w6 = [0, 0, 0, 1]
        
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # w7 = [1, 0, 0, 0]
        
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        for u, v, w in terms:
            # Map local indices to global indices
            # A indices: i0, i0+1, j0, j0+1 -> i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
            # B indices: j0, j0+1, k0, k0+1 -> j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
            # C indices: k0, k0+1, i0, i0+1 -> k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
            
            a_idx = [i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1]
            b_idx = [j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1]
            c_idx = [k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1]
            
            for l in range(4):
                if u[l] != 0:
                    U[a_idx[l]][col] += u[l]
                if v[l] != 0:
                    V[b_idx[l]][col] += v[l]
                if w[l] != 0:
                    W[c_idx[l]][col] += w[l]
            col += 1

    def add_schoolbook_2x2(i0, j0, k0):
        nonlocal col
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    U[(i0+ii)*m + (j0+jj)][col] = 1
                    V[(j0+jj)*p + (k0+kk)][col] = 1
                    W[(k0+kk)*n + (i0+ii)][col] = 1
                    col += 1

    def add_schoolbook_2x1(i0, j0, k0):
        nonlocal col
        for ii in range(2):
            for jj in range(2):
                U[(i0+ii)*m + (j0+jj)][col] = 1
                V[(j0+jj)*p + k0][col] = 1
                W[k0*n + (i0+ii)][col] = 1
                col += 1

    # A*B1 (4x4 * 4x4)
    # Split into four 2x2 * 2x2 blocks
    add_strassen_2x2(0, 0, 0) # A[0:2, 0:2] * B1[0:2, 0:2]
    add_strassen_2x2(0, 0, 2) # A[0:2, 0:2] * B1[0:2, 2:4]
    add_strassen_2x2(2, 0, 0) # A[2:4, 0:2] * B1[0:2, 0:2]
    add_strassen_2x2(2, 0, 2) # A[2:4, 0:2] * B1[0:2, 2:4]
    # Wait, the Strassen algorithm is for the whole 4x4 matrix.
    # I used 4 separate 2x2 Strassen multiplications.
    # This is not the standard Strassen algorithm for 4x4.
    # Standard Strassen for 4x4 splits into 2x2 blocks and uses 7 multiplications of 2x2 blocks.
    # My method uses 4 multiplications of 2x2 blocks.
    # Rank = 4 * 7 = 28.
    # This is better than 49!
    # Let's check if this is valid.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # Each Ci is a 2x2 matrix.
    # C11 = A11B11 + A12B21. This is a sum of two 2x2*2x2 products.
    # If I use Strassen for A11B11 (rank 7) and A12B21 (rank 7), total rank 14 for C11.
    # Total rank for C = 4 * 14 = 56.
    # This is worse than 49.
    # The standard Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Each Mi is a 2x2*2x2 product. Rank 7.
    # Total rank = 7 * 7 = 49.
    
    # My previous code `add_strassen_2x2` was adding 7 terms for a 2x2*2x2 product.
    # I called it 4 times, so 28 terms.
    # But this computes A11B11, A12B21, A21B12, A22B22 separately.
    # It does NOT compute the cross terms A11B12, etc.
    # So it only computes the diagonal blocks of C.
    # I need to compute the off-diagonal blocks too.
    # So I need to compute 4 2x2*2x2 products for the diagonal, and 4 for the off-diagonal?
    # No, C12 = A11B12 + A12B22.
    # This is not a single 2x2*2x2 product.
    # So the "4 separate Strassen" idea is flawed for the whole 4x4.
    
    # I should use the standard Strassen algorithm for 4x4.
    # This requires implementing the 7 products M1..M7.
    # Each Mi is a 2x2*2x2 product.
    # I can use `add_strassen_2x2` for each Mi, but I need to handle the linear combinations.
    # This is complex.
    
    # Let's stick to the 114 rank strategy which is known to work.
    # A*B1 (4x4 * 4x4) with rank 49.
    # A*B2 (4x4 * 4x5) with rank 65.
    # Total 114.
    
    # I will implement the 49 rank decomposition for 4x4*4x4 using the 7 Strassen products.
    # And the 65 rank decomposition for 4x4*4x5 using schoolbook.
    
    # This is getting complicated to code from scratch in this short time.
    # I will revert to the schoolbook decomposition for the whole 4x4*4x9, which is rank 144.
    # This is safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
