import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :rank], V[j * p + k, :rank] * W[k * n + i, :rank])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def verify_decomp_fast(U, V, W, rank):
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # T should be all ones
        for r in range(rank):
            u_r = U[:, r]  # length nm = 8
            v_r = V[:, r]  # length mp = 32
            w_r = W[:, r]  # length pn = 16
            # We need to check contribution to each (i,j,k)
            # This is a check, so we do it directly
            pass
        # Full verification
        return verify_decomp(U, V, W, rank)

    # Strassen-based 56-rank decomposition (4 blocks of 2x4x4x2, each 14-rank)
    def strassen_56():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        for v in range(4):  # which pair of output columns
            base = 14 * v

            # A partition: rows 0,1 of A (each of which is a 1x4 row)
            # B partition: 2x2 blocks for columns 2v, 2v+1
            # Output: C rows for columns 2v, 2v+1

            a0 = 0  # A[0, :] indices: A[0,0],A[0,1],A[0,2],A[0,3]
            a1 = 4  # A[1, :] indices: A[1,0],A[1,1],A[1,2],A[1,3]

            # B is 4x8, take columns 2v and 2v+1
            b11 = (2 * v) * p  # row 0, col 2v
            b12 = (2 * v) * p + 1  # row 0, col 2v+1
            b21 = (2 * v + 1) * p  # row 1, col 2v
            b22 = (2 * v + 1) * p + 1  # row 1, col 2v+1
            b31 = (2 * v + 2) * p  # row 2, col 2v
            b32 = (2 * v + 2) * p + 1  # row 2, col 2v+1
            b41 = (2 * v + 3) * p  # row 3, col 2v
            b42 = (2 * v + 3) * p + 1  # row 3, col 2v+1

            # Output C: 2x2 block for columns 2v, 2v+1
            c11 = (2 * v) * n + 0  # C[0, 2v]
            c12 = (2 * v + 1) * n + 0  # C[0, 2v+1]
            c21 = (2 * v) * n + 1  # C[1, 2v]
            c22 = (2 * v + 1) * n + 1  # C[1, 2v+1]

            # We need 2x4 x 4x2 = 2x2, which has rank at most 16 (schoolbook)
            # Strassen for 2x4 x 4x2: split the 4 into 2+2, giving 2 (2x2x2x2) blocks
            # Each 2x2x2x2 has rank 7 (Strassen), so total 14

            # Block 1: A[:, 0:2] * B[0:2, :] = C_block1
            # Block 2: A[:, 2:4] * B[2:4, :] = C_block2 (same C)

            # For block 1 (A rows 0,1 cols 0,1; B rows 0,1 cols 2v,2v+1):
            # M1 = (A00+A11)(B00+B11): 
            col = base
            # r0: M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,0])
            U[a0 + 0, col] += 1  # A[0,0]
            U[a1 + 1, col] += 1  # A[1,1]
            V[b11, col] += 1  # B[0,0]
            V[b21, col] += 1  # B[1,0]
            W[c11, col] += 1  # C[0,0]
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r1: M2 = (A[1,0]+A[1,1]) * B[0,0]
            U[a1 + 0, col] += 1
            U[a1 + 1, col] += 1
            V[b11, col] += 1
            W[c21, col] += 1  # C[1,0]
            W[c22, col] -= 1  # C[1,1]
            col += 1

            # r2: M3 = A[0,0] * (B[0,1] - B[1,1])
            U[a0 + 0, col] += 1
            V[b12, col] += 1  # B[0,1]
            V[b22, col] -= 1  # B[1,1]
            W[c12, col] += 1  # C[0,1]
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r3: M4 = A[1,1] * (B[1,0] - B[0,0])
            U[a1 + 1, col] += 1
            V[b21, col] += 1  # B[1,0]
            V[b11, col] -= 1  # B[0,0]
            W[c11, col] += 1  # C[0,0]
            W[c21, col] += 1  # C[1,0]
            col += 1

            # r4: M5 = (A[0,0]+A[0,1]) * B[1,1]
            U[a0 + 0, col] += 1
            U[a0 + 1, col] += 1
            V[b22, col] += 1  # B[1,1]
            W[c11, col] -= 1  # C[0,0]
            W[c12, col] += 1  # C[0,1]
            col += 1

            # r5: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
            U[a1 + 0, col] += 1
            U[a0 + 0, col] -= 1
            V[b11, col] += 1
            V[b12, col] += 1
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r6: M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
            U[a0 + 1, col] += 1
            U[a1 + 1, col] -= 1
            V[b21, col] += 1
            V[b22, col] += 1
            W[c11, col] += 1  # C[0,0]
            col += 1

            # For block 2 (A rows 0,1 cols 2,3; B rows 2,3 cols 2v,2v+1):
            col = base + 7

            # r7: M8 = (A[0,2]+A[1,3])(B[2,0]+B[3,0])
            U[a0 + 2, col] += 1  # A[0,2]
            U[a1 + 3, col] += 1  # A[1,3]
            V[b31, col] += 1  # B[2,0]
            V[b41, col] += 1  # B[3,0]
            W[c11, col] += 1  # C[0,0]
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r8: M9 = (A[1,2]+A[1,3]) * B[2,0]
            U[a1 + 2, col] += 1
            U[a1 + 3, col] += 1
            V[b31, col] += 1
            W[c21, col] += 1  # C[1,0]
            W[c22, col] -= 1  # C[1,1]
            col += 1

            # r9: M10 = A[0,2] * (B[2,1] - B[3,1])
            U[a0 + 2, col] += 1
            V[b32, col] += 1
            V[b42, col] -= 1
            W[c12, col] += 1  # C[0,1]
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r10: M11 = A[1,3] * (B[3,0] - B[2,0])
            U[a1 + 3, col] += 1
            V[b41, col] += 1
            V[b31, col] -= 1
            W[c11, col] += 1  # C[0,0]
            W[c21, col] += 1  # C[1,0]
            col += 1

            # r11: M12 = (A[0,2]+A[0,3]) * B[3,1]
            U[a0 + 2, col] += 1
            U[a0 + 3, col] += 1
            V[b42, col] += 1
            W[c11, col] -= 1  # C[0,0]
            W[c12, col] += 1  # C[0,1]
            col += 1

            # r12: M13 = (A[1,2]-A[0,2]) * (B[2,0]+B[2,1])
            U[a1 + 2, col] += 1
            U[a0 + 2, col] -= 1
            V[b31, col] += 1
            V[b32, col] += 1
            W[c22, col] += 1  # C[1,1]
            col += 1

            # r13: M14 = (A[0,3]-A[1,3]) * (B[3,0]+B[3,1])
            U[a0 + 3, col] += 1
            U[a1 + 3, col] -= 1
            V[b41, col] += 1
            V[b42, col] += 1
            W[c11, col] += 1  # C[0,0]
            col += 1

        return U, V, W, rank

    # Generate the 56-rank decomposition
    U_cur, V_cur, W_cur, rank_cur = strassen_56()

    if not verify_decomp(U_cur, V_cur, W_cur, rank_cur):
        U_cur, V_cur, W_cur, rank_cur = schoolbook()

    # Search for lower rank by column elimination
    best_U, best_V, best_W, best_rank = U_cur.copy(), V_cur.copy(), W_cur.copy(), rank_cur
    current_U, current_V, current_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    current_rank = rank_cur
    removed = []

    while current_rank > 16 and time.time() < deadline - 5:
        found_improvement = False
        for col_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Try removing column col_idx
            keep_cols = [c for c in range(current_rank) if c != col_idx]
            if not keep_cols:
                continue

            U_test = current_U[:, keep_cols]
            V_test = current_V[:, keep_cols]
            W_test = current_W[:, keep_cols]

            if verify_decomp(U_test, V_test, W_test, len(keep_cols)):
                # Accept
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank = len(keep_cols)
                removed.append(col_idx)
                found_improvement = True
                break

        if not found_improvement:
            break

    if current_rank < best_rank:
        best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank

    # Final verification
    if not verify_decomp(best_U, best_V, best_W, best_rank):
        best_U, best_V, best_W, best_rank = schoolbook()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
