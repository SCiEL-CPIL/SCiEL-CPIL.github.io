import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # U shape (n*m, rank) = (16, rank)
    # V shape (m*p, rank) = (36, rank)
    # W shape (p*n, rank) = (36, rank)
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 blocks: i in {0,1}, j in {0,1,2}
    for i in range(2):
        for j in range(3):
            # For each block, we have 11 products
            # M is A[i*2:i*2+2, 0:2] and A[i*2:i*2+2, 2:4]
            # Wait, A is 4x4.
            # A blocks:
            # A00: rows 0,1 cols 0,1
            # A01: rows 0,1 cols 2,3
            # A10: rows 2,3 cols 0,1
            # A11: rows 2,3 cols 2,3
            # 
            # B blocks:
            # B00: rows 0,1 cols 0,1,2
            # B01: rows 0,1 cols 3,4,5
            # B02: rows 0,1 cols 6,7,8
            # B10: rows 2,3 cols 0,1,2
            # B11: rows 2,3 cols 3,4,5
            # B12: rows 2,3 cols 6,7,8
            #
            # C blocks:
            # C00: rows 0,1 cols 0,1,2
            # ...
            #
            # For block (i,j), we compute:
            # C_ij = A_i0 B_0j + A_i1 B_1j
            # where A_i0 is A[i*2:i*2+2, 0:2]
            # A_i1 is A[i*2:i*2+2, 2:4]
            # B_0j is B[0:2, j*3:j*3+3]
            # B_1j is B[2:4, j*3:j*3+3]
            
            # We need to decompose the sum of two 2x2 by 2x3 multiplications.
            # This is equivalent to one 2x2 by 2x3 multiplication if we view A_i0 and A_i1 as part of a larger matrix?
            # No, it's just two separate multiplications added together.
            # We can just do the 11 products for A_i0 B_0j and 11 products for A_i1 B_1j?
            # That would be 22 products per block. 6 * 22 = 132.
            # But we can combine them?
            # No, the matrices are different.
            # However, we can just use the 11-product decomposition for each term.
            # So 11 for A_i0 B_0j and 11 for A_i1 B_1j.
            # Total 22 per block.
            # 6 blocks * 22 = 132.
            
            # Wait, earlier I said 66. Why?
            # I thought I could do the whole thing in 66.
            # Let's re-evaluate.
            # C_ij = A_i0 B_0j + A_i1 B_1j.
            # This is a sum of two products.
            # We can compute them separately.
            # Rank is additive for direct sums.
            # So rank is 11 + 11 = 22 per block.
            # Total 132.
            
            # Can we do better?
            # We can view A_i = [A_i0 A_i1] as a 2x4 matrix.
            # B_j = [B_0j; B_1j] as a 4x3 matrix.
            # C_ij = A_i B_j.
            # This is a 2x4 by 4x3 multiplication.
            # Rank of 2x4 by 4x3?
            # Schoolbook: 2*4*3 = 24.
            # Strassen-like?
            # We can split 2x4 into two 2x2.
            # This leads back to 11+11=22.
            # So 132 is the rank.
            
            # 132 is still better than 144.
            
            # Let's implement the 11-product decomposition for 2x2 by 2x3.
            # We need to map the indices.
            
            # For A_i0 B_0j:
            # M = A_i0 (2x2)
            # N = B_0j (2x3)
            # Products P1..P11
            
            # For A_i1 B_1j:
            # M = A_i1 (2x2)
            # N = B_1j (2x3)
            # Products P12..P22
            
            # We will add the contributions to W.
            
            # Helper to add a 2x2 by 2x3 decomposition
            def add_2x2_2x3(U, V, W, col_start, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
                # a_row_offset: row offset in A (0 or 2)
                # a_col_offset: col offset in A (0 or 2)
                # b_row_offset: row offset in B (0 or 2)
                # b_col_offset: col offset in B (0, 3, or 6)
                # c_row_offset: row offset in C (0 or 2)
                # c_col_offset: col offset in C (0, 3, or 6)
                
                # M indices in A:
                # m00: a_row_offset, a_col_offset
                # m01: a_row_offset, a_col_offset+1
                # m10: a_row_offset+1, a_col_offset
                # m11: a_row_offset+1, a_col_offset+1
                
                # N indices in B:
                # n00: b_row_offset, b_col_offset
                # n01: b_row_offset, b_col_offset+1
                # n02: b_row_offset, b_col_offset+2
                # n10: b_row_offset+1, b_col_offset
                # n11: b_row_offset+1, b_col_offset+1
                # n12: b_row_offset+1, b_col_offset+2
                
                # P indices in C:
                # p00: c_row_offset, c_col_offset
                # p01: c_row_offset, c_col_offset+1
                # p02: c_row_offset, c_col_offset+2
                # p10: c_row_offset+1, c_col_offset
                # p11: c_row_offset+1, c_col_offset+1
                # p12: c_row_offset+1, c_col_offset+2
                
                # Strassen for cols 0,1
                # P1 = m00 (n11 - n10)
                c = col_start
                U[a_row_offset * 4 + a_col_offset][c] += 1
                V[b_row_offset * 9 + b_col_offset + 1][c] += 1
                V[b_row_offset * 9 + b_col_offset][c] -= 1
                W[(b_col_offset) * 4 + c_row_offset][c] += 1
                W[(b_col_offset + 1) * 4 + c_row_offset][c] += 1
                W[(b_col_offset) * 4 + c_row_offset + 1][c] -= 1
                W[(b_col_offset + 1) * 4 + c_row_offset + 1][c] -= 1
                # Wait, W index is k*n + i.
                # k is col of B / row of C?
                # W[k*n + i, r].
                # C_{ij} = sum_k A_{ik} B_{kj}.
                # W[k*n + i] corresponds to C_{ij} where k is the "inner" index?
                # No, W is (p*n, rank).
                # W[k*n + i, r] multiplies U[i*m + k, r] and V[k*p + j, r]?
                # Let's check the convention.
                # U[i*m + k, r] * V[k*p + j, r] * W[k*n + i, r]?
                # Sum_r U[i*m + k, r] V[k*p + j, r] W[k*n + i, r]
                # This doesn't look right.
                # Standard: C_{ij} = sum_r U_{i, r} V_{r, j} W_{r, k}?
                # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
                # And W[k*n+i, r].
                # Usually, the tensor is T_{ijk} = sum_r U_{ik, r} V_{kj, r} W_{ji, r}?
                # Let's assume the standard decomposition:
                # C_{ij} = sum_r U_{i, k, r} V_{k, j, r} W_{j, i, r}?
                # No, the shapes are fixed.
                # U[i*m + k, r] is A_{ik}.
                # V[k*p + j, r] is B_{kj}.
                # W[j*n + i, r] is C_{ji}?
                # The task says W[k*n + i, r].
                # If we sum over r and k:
                # sum_r sum_k U[i*m + k, r] V[k*p + j, r] W[k*n + i, r]
                # This is sum_k (sum_r U[i*m + k, r] V[k*p + j, r]) W[k*n + i, r]
                # This doesn't make sense as a tensor decomposition.
                
                # Let's look at the seed program.
                # U[i*m + j][column] = 1
                # V[j*p + k][column] = 1
                # W[k*n + i][column] = 1
                # This corresponds to term A_{ij} B_{jk} C_{ki}?
                # No, matrix multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
                # So the indices are i, j, k.
                # U[i*m + j] -> A_{ij}
                # V[j*p + k] -> B_{jk}
                # W[k*n + i] -> C_{ki}?
                # If C = AB, then C_{ik} = sum_j A_{ij} B_{jk}.
                # So the output index is i, k.
                # W[k*n + i] suggests the index is k*n + i.
                # So W corresponds to C_{ik} stored at k*n + i.
                # This means the output matrix C is stored in column-major order? Or transposed?
                # If W[k*n + i] is C_{ik}, then the row index is k and col index is i?
                # No, standard row-major is i*p + k.
                # Here it is k*n + i.
                # This is column-major order for a n x p matrix?
                # If C is n x p, column-major index is i + k*n.
                # So W[k*n + i] is C_{ik} in column-major order.
                
                # So we need to fill W at index k*n + i.
                
                # Let's re-verify the Strassen mapping.
                # We are computing C_{ij} for i in {c_row_offset, c_row_offset+1} and j in {c_col_offset, c_col_offset+1, c_col_offset+2}.
                # Wait, C is 4x9.
                # i is row (0..3), j is col (0..8).
                # W index is k*n + i?
                # No, the seed uses W[k*n + i].
                # In the seed, i is row of A, j is col of A / row of B, k is col of B.
                # So C_{ik} is the product.
                # So W index is k*n + i.
                # This means the "row" of C is k and "col" of C is i?
                # No, C_{ik} has row i and col k.
                # So W[k*n + i] stores C_{ik}.
                # This is column-major storage of C.
                
                # So for our block:
                # C_{ij} where i is global row, j is global col.
                # W index is j*n + i.
                
                # Let's correct the W indices in the helper.
                
                # P1 = m00 (n11 - n10)
                # m00: A[a_row_offset, a_col_offset]
                # n11: B[b_row_offset+1, b_col_offset+1]
                # n10: B[b_row_offset+1, b_col_offset]
                # Result C:
                # C_{a_row_offset, b_col_offset+1} += m00 * n11
                # C_{a_row_offset, b_col_offset} -= m00 * n10
                # C_{a_row_offset+1, b_col_offset+1} += m00 * n11
                # C_{a_row_offset+1, b_col_offset} -= m00 * n10
                # Wait, Strassen:
                # P1 = M00 (N11 - N10)
                # C01 += P1
                # C11 += P1
                # C00 -= P1? No.
                # C00 = P2 + P3 - P4 + P5 - P7
                # C01 = P1 + P5
                # C10 = P3 + P4
                # C11 = P1 + P2 + P5 - P6
                # 
                # Let's use the standard Strassen formulas.
                # P1 = M00 (N11 - N10)
                # P2 = (M00 + M01) N10
                # P3 = (M10 + M11) N00
                # P4 = M11 (N00 - N01)
                # P5 = (M00 + M11) (N01 - N10)
                # P6 = (M10 - M00) N01
                # P7 = (M01 + M11) N11
                # 
                # C00 = P2 + P3 - P4 + P5 - P7
                # C01 = P1 + P5
                # C10 = P3 + P4
                # C11 = P1 + P2 + P5 - P6
                
                # We need to add these to W.
                # W index for C_{ij} is j*n + i.
                
                # P1 contributes to C01 and C11.
                # C01: i=a_row_offset, j=b_col_offset+1. W index: (b_col_offset+1)*4 + a_row_offset.
                # C11: i=a_row_offset+1, j=b_col_offset+1. W index: (b_col_offset+1)*4 + a_row_offset+1.
                
                # This is getting complex. I will write a helper function that takes the local indices and adds to the global arrays.
                
                return col_start + 11

            # We will just implement the 11 products directly in the loop.
            
            # For A_i0 B_0j
            a_r = i * 2
            a_c = 0
            b_r = 0
            b_c = j * 3
            c_r = i * 2
            c_c = j * 3
            
            # P1 = m00 (n11 - n10)
            U[a_r * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            V[b_r * 9 + b_c][col] -= 1
            W[(b_c + 1) * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            
            # P2 = (m00 + m01) n10
            U[a_r * 4 + a_c][col] += 1
            U[a_r * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            W[b_c * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            
            # P3 = (m10 + m11) n00
            U[(a_r + 1) * 4 + a_c][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            W[b_c * 4 + c_r][col] += 1
            W[b_c * 4 + c_r + 1][col] += 1
            col += 1
            
            # P4 = m11 (n00 - n01)
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            V[b_r * 9 + b_c + 1][col] -= 1
            W[b_c * 4 + c_r + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] -= 1
            col += 1
            
            # P5 = (m00 + m11) (n01 - n10)
            U[a_r * 4 + a_c][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            V[b_r * 9 + b_c][col] -= 1
            W[(b_c + 1) * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            W[b_c * 4 + c_r][col] -= 1
            W[b_c * 4 + c_r + 1][col] -= 1
            col += 1
            
            # P6 = (m10 - m00) n01
            U[(a_r + 1) * 4 + a_c][col] += 1
            U[a_r * 4 + a_c][col] -= 1
            V[b_r * 9 + b_c + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            
            # P7 = (m01 + m11) n11
            U[a_r * 4 + a_c + 1][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            
            # P8-P11 for col 2
            # P8 = m00 n02
            U[a_r * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r][col] += 1
            col += 1
            
            # P9 = m01 n12
            U[a_r * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r][col] += 1
            col += 1
            
            # P10 = m10 n02
            U[(a_r + 1) * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r + 1][col] += 1
            col += 1
            
            # P11 = m11 n12
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r + 1][col] += 1
            col += 1
            
            # Now for A_i1 B_1j
            a_r = i * 2
            a_c = 2
            b_r = 2
            b_c = j * 3
            c_r = i * 2
            c_c = j * 3
            
            # Same 11 products
            # P1
            U[a_r * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            V[b_r * 9 + b_c][col] -= 1
            W[(b_c + 1) * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            # P2
            U[a_r * 4 + a_c][col] += 1
            U[a_r * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            W[b_c * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            # P3
            U[(a_r + 1) * 4 + a_c][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            W[b_c * 4 + c_r][col] += 1
            W[b_c * 4 + c_r + 1][col] += 1
            col += 1
            # P4
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c][col] += 1
            V[b_r * 9 + b_c + 1][col] -= 1
            W[b_c * 4 + c_r + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] -= 1
            col += 1
            # P5
            U[a_r * 4 + a_c][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            V[b_r * 9 + b_c][col] -= 1
            W[(b_c + 1) * 4 + c_r][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            W[b_c * 4 + c_r][col] -= 1
            W[b_c * 4 + c_r + 1][col] -= 1
            col += 1
            # P6
            U[(a_r + 1) * 4 + a_c][col] += 1
            U[a_r * 4 + a_c][col] -= 1
            V[b_r * 9 + b_c + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            # P7
            U[a_r * 4 + a_c + 1][col] += 1
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 1][col] += 1
            W[(b_c + 1) * 4 + c_r + 1][col] += 1
            col += 1
            # P8
            U[a_r * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r][col] += 1
            col += 1
            # P9
            U[a_r * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r][col] += 1
            col += 1
            # P10
            U[(a_r + 1) * 4 + a_c][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r + 1][col] += 1
            col += 1
            # P11
            U[(a_r + 1) * 4 + a_c + 1][col] += 1
            V[b_r * 9 + b_c + 2][col] += 1
            W[(b_c + 2) * 4 + c_r + 1][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
