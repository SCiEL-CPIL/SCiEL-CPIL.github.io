"""Matrix multiplication 3x4 x 4x6 with rank 66 decomposition using Strassen on 2x2x2 subblocks."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # Decompose: split m=4 into 2+2, split p=6 into 2+2+2
    # For each middle block (2 of them) and each column block (3 of them):
    #   Compute A_i (3x2) * B_j (2x2) = C_k (3x2)
    #   3x2x2 = [2,2,2] (rows 0,1) + [1,2,2] (row 2)
    #   [2,2,2]: rank 7 (Strassen)
    #   [1,2,2]: rank 4 (schoolbook)
    #   Subtotal: 11 per (mid, col) block
    # Total: 2 * 3 * 11 = 66

    rank_total = 66
    U = [[0.0] * rank_total for _ in range(n * m)]  # 12 rows
    V = [[0.0] * rank_total for _ in range(m * p)]  # 24 rows
    W = [[0.0] * rank_total for _ in range(p * n)]  # 18 rows

    col = 0
    for mid in range(2):  # mid=0: A cols 0-1, B rows 0-1; mid=1: A cols 2-3, B rows 2-3
        for cblk in range(3):  # cblk=0: C cols 0-1; cblk=1: C cols 2-3; cblk=2: C cols 4-5
            # [2,2,2] Strassen for A[0:2, mid*2:mid*2+2] * B[mid*2:mid*2+2, cblk*2:cblk*2+2]
            # a00 = A[0][mid*2], a01 = A[0][mid*2+1]
            # a10 = A[1][mid*2], a11 = A[1][mid*2+1]
            # e00 = B[mid*2][cblk*2], e01 = B[mid*2][cblk*2+1]
            # e10 = B[mid*2+1][cblk*2], e11 = B[mid*2+1][cblk*2+1]
            
            # M1 = (a00+a11)(e00+e11)
            U[0 * m + mid*2 + 0][col] += 1   # a00
            U[1 * m + mid*2 + 1][col] += 1   # a11
            V[mid*2 + 0][col * 1 + cblk*2 + 0][0] if False else None  # careful
            # V index: row = mid*2 + 0, col = cblk*2 + 0
            V[mid*2 + 0][cblk*2 + 0] if len(V[mid*2+0]) > cblk*2+1 else None  # wrong, V is 24x66
            
            # I'm confusing myself. Let me be very explicit.
            # U is 12x66: U[row_in_A, col]
            # V is 24x66: V[row_in_B, col]  
            # W is 18x66: W[row_in_C, col]
            # A[i][j] contributes to U[i*4 + j, col]
            # B[i][j] contributes to V[i*6 + j, col]
            # C[i][j] contributes to W[j*3 + i, col]
            
            # M1 = (a00 + a11) * (e00 + e11)
            # a00: A[0][mid*2] -> U[0*4 + mid*2, col]
            # a11: A[1][mid*2+1] -> U[1*4 + mid*2+1, col]
            # e00: B[mid*2][cblk*2] -> V[mid*2*6 + cblk*2, col]
            # e11: B[mid*2+1][cblk*2+1] -> V[(mid*2+1)*6 + cblk*2+1, col]
            # C00 = C[0][cblk*2] -> W[cblk*2*3 + 0, col]
            # C11 = C[1][cblk*2+1] -> W[(cblk*2+1)*3 + 1, col]
            # But Strassen combines them. For M1, it contributes +1 to C00 and +1 to C11 (with appropriate signs)
            
            # Actually, in the rank-1 decomposition, each "M" is a single scalar product, and W tells how it contributes to each output.
            # M1 = (a00+a11)(e00+e11)
            # This M1 contributes to C00 with coeff +1, C01 with coeff 0, C10 with coeff 0, C11 with coeff +1? No.
            # In Strassen: C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
            # So M1 contributes: C00 += 1, C11 += 1
            # In our case, the 2x2 output is C[0:2, cblk*2:cblk*2+2]
            # C00 -> C[0][cblk*2] -> W[cblk*2*3 + 0, col]
            # C11 -> C[1][cblk*2+1] -> W[(cblk*2+1)*3 + 1, col]
            
            U[0*4 + mid*2 + 0][col] += 1
            U[1*4 + mid*2 + 1][col] += 1
            V[mid*2*6 + cblk*2 + 0][col] += 1
            V[(mid*2+1)*6 + cblk*2 + 1][col] += 1
            W[cblk*2*3 + 0][col] += 1    # C[0][cblk*2]
            W[(cblk*2+1)*3 + 1][col] += 1  # C[1][cblk*2+1]
            col += 1

            # M2 = (a10+a11) * e00
            # C10 += 1
            U[1*4 + mid*2 + 0][col] += 1
            U[1*4 + mid*2 + 1][col] += 1
            V[mid*2*6 + cblk*2 + 0][col] += 1
            W[cblk*2*3 + 1][col] += 1    # C[1][cblk*2]
            col += 1

            # M3 = a00 * (e10-e11)
            # C01 += 1
            U[0*4 + mid*2 + 0][col] += 1
            V[(mid*2+1)*6 + cblk*2 + 0][col] += 1
            V[(mid*2+1)*6 + cblk*2 + 1][col] += -1
            W[(cblk*2+1)*3 + 0][col] += 1  # C[0][cblk*2+1]
            col += 1

            # M4 = a11 * (e01-e00)
            # C00 += 1, C10 += 1
            U[1*4 + mid*2 + 1][col] += 1
            V[mid*2*6 + cblk*2 + 1][col] += 1
            V[mid*2*6 + cblk*2 + 0][col] += -1
            W[cblk*2*3 + 0][col] += 1
            W[cblk*2*3 + 1][col] += 1
            col += 1

            # M5 = (a00+a01) * e11
            # C00 += -1, C01 += 1
            U[0*4 + mid*2 + 0][col] += 1
            U[0*4 + mid*2 + 1][col] += 1
            V[(mid*2+1)*6 + cblk*2 + 1][col] += 1
            W[cblk*2*3 + 0][col] += -1
            W[(cblk*2+1)*3 + 0][col] += 1
            col += 1

            # M6 = (a10-a00) * (e00+e01)
            # C11 += 1
            U[1*4 + mid*2 + 0][col] += 1
            U[0*4 + mid*2 + 0][col] += -1
            V[mid*2*6 + cblk*2 + 0][col] += 1
            V[mid*2*6 + cblk*2 + 1][col] += 1
            W[(cblk*2+1)*3 + 1][col] += 1
            col += 1

            # M7 = (a01-a11) * (e10+e11)
            # C00 += 1
            U[0*4 + mid*2 + 1][col] += 1
            U[1*4 + mid*2 + 1][col] += -1
            V[(mid*2+1)*6 + cblk*2 + 0][col] += 1
            V[(mid*2+1)*6 + cblk*2 + 1][col] += 1
            W[cblk*2*3 + 0][col] += 1
            col += 1

            # Now [1,2,2]: row 2 of A_block * B_block
            # A[2][mid*2+j] * B[mid*2+i][cblk*2+k] for i,k in 0,1
            # This gives C[2][cblk*2+k] = sum_j A[2][mid*2+j] * B[mid*2+j][cblk*2+k]
            # 4 terms, each rank 1
            
            for j in range(2):
                for k in range(2):
                    U[2*4 + mid*2 + j][col] += 1
                    V[(mid*2+j)*6 + cblk*2 + k][col] += 1
                    W[(cblk*2+k)*3 + 2][col] += 1
                    col += 1

    assert col == rank_total, f"Expected {rank_total}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
