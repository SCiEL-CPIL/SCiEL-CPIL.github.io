import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    # Build the rank-36 base using block-Strassen (4 blocks of 7 + 8 direct terms)
    rank_base = 36
    U = [[0.0] * rank_base for _ in range(n * m)]
    V = [[0.0] * rank_base for _ in range(m * p)]
    W = [[0.0] * rank_base for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    col = 0

    # Strassen 2x2x2 template
    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w

        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1

        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1

        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1

        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1

        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1

        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1

        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    # Define blocks
    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(0,1),(1,0),(1,1)]  # w_idx(k,i): k=0,1; i=0,1
    C12 = [(2,0),(2,1),(3,0),(3,1)]  # k=2,3; i=0,1

    # Apply 4 Strassen blocks
    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # 8 direct terms for k=4
    for (ai, aj) in [(0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3)]:
        U[u_idx(ai, aj)][col] += 1
        V[v_idx(aj, 4)][col] += 1
        W[w_idx(4, ai)][col] += 1
        col += 1

    assert col == 36, f"Expected 36, got {col}"

    # Convert to numpy for fast verification
    base_U = np.array(U, dtype=float)
    base_V = np.array(V, dtype=float)
    base_W = np.array(W, dtype=float)

    def verify_fast(Ua, Va, Wa, R):
        """Check if decomposition is exact for rank R (first R columns)."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    s = np.dot(Ua[i*m+j, :R], Va[j*p+k, :R] * Wa[k*n+i, :R])
                    if abs(s - 1.0) > 1e-7:
                        return False
        return True

    # Verify base
    if not verify_fast(base_U, base_V, base_W, 36):
        # Fallback to schoolbook rank 40
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col_s] = 1
                    V_s[j*p+k, col_s] = 1
                    W_s[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    best_rank = 36
    best_U = base_U[:, :36].copy()
    best_V = base_V[:, :36].copy()
    best_W = base_W[:, :36].copy()

    # Search for rank-35 by merging columns
    # Strategy: try removing one column by absorbing it into another
    # Column layout: 0-6: block1(AL,B11), 7-13: block2(AR,B21),
    #                14-20: block3(AL,B12), 21-27: block4(AR,B22), 28-35: direct
    # Compatible merge pairs: columns from different blocks that cover same output
    # Block1 (AL,B11) and Block2 (AR,B21) both feed C11 (k=0,1; i=0,1)
    # Block3 (AL,B12) and Block4 (AR,B22) both feed C12 (k=2,3; i=0,1)

    def try_remove_column(Ua, Va, Wa, R, remove_idx, target_idx, sign):
        """Try to absorb column remove_idx into column target_idx with given sign."""
        U2 = Ua[:, :R].copy()
        V2 = Va[:, :R].copy()
        W2 = Wa[:, :R].copy()
        # Add sign * col[remove_idx] to col[target_idx]
        U2[:, target_idx] += sign * U2[:, remove_idx]
        V2[:, target_idx] += sign * V2[:, remove_idx]
        W2[:, target_idx] += sign * W2[:, remove_idx]
        # Zero out the remove column
        U2[:, remove_idx] = 0
        V2[:, remove_idx] = 0
        W2[:, remove_idx] = 0
        return verify_fast(U2, V2, W2, R)

    # Try single-column removal (rank 35)
    # Focus on compatible pairs: same output block, different A-block
    pairs = [
        # Block1 vs Block2 (both feed C11)
        (0, 7), (1, 8), (2, 9), (3, 10), (4, 11), (5, 12), (6, 13),
        # Block3 vs Block4 (both feed C12)
        (14, 21), (15, 22), (16, 23), (17, 24), (18, 25), (19, 26), (20, 27),
        # Also try within-block merges (less likely but cheap to check)
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
        (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
        (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
        # Direct terms among themselves
        (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        # Cross between direct and Strassen
        (6, 28), (13, 29), (20, 30), (27, 31),
    ]

    found_35 = False
    for (c1, c2) in pairs:
        if time.time() > deadline - 20:
            break
        for sign in (1.0, -1.0):
            if try_remove_column(base_U, base_V, base_W, 36, c1, c2, sign):
                # Found rank 35!
                U35 = base_U[:, :36].copy()
                V35 = base_V[:, :36].copy()
                W35 = base_W[:, :36].copy()
                U35[:, c2] += sign * U35[:, c1]
                V35[:, c2] += sign * V35[:, c1]
                W35[:, c2] += sign * W35[:, c1]
                U35[:, c1] = 0
                V35[:, c1] = 0
                W35[:, c1] = 0
                # Remove the zero column
                mask = np.ones(36, dtype=bool)
                mask[c1] = False
                U35 = U35[:, mask]
                V35 = V35[:, mask]
                W35 = W35[:, mask]
                best_rank = 35
                best_U = U35
                best_V = V35
                best_W = W35
                found_35 = True
                break
        if found_35:
            break

    # If we found rank 35, try to go further to rank 34
    if found_35 and time.time() < deadline - 15:
        U35 = best_U
        V35 = best_V
        W35 = best_W
        R35 = 35
        # Try removing one more column from the rank-35 solution
        for c1 in range(R35):
            if time.time() > deadline - 10:
                break
            for c2 in range(R35):
                if c1 == c2:
                    continue
                for sign in (1.0, -1.0):
                    if try_remove_column(U35, V35, W35, R35, c1, c2, sign):
                        U34 = U35[:, :R35].copy()
                        V34 = V35[:, :R35].copy()
                        W34 = W35[:, :R35].copy()
                        U34[:, c2] += sign * U34[:, c1]
                        V34[:, c2] += sign * V34[:, c1]
                        W34[:, c2] += sign * W34[:, c1]
                        U34[:, c1] = 0
                        V34[:, c1] = 0
                        W34[:, c1] = 0
                        mask = np.ones(R35, dtype=bool)
                        mask[c1] = False
                        U34 = U34[:, mask]
                        V34 = V34[:, mask]
                        W34 = W34[:, mask]
                        best_rank = 34
                        best_U = U34
                        best_V = V34
                        best_W = W34
                        break
                if best_rank == 34:
                    break
            if best_rank == 34:
                break

    # If we have time, try a broader search: linear combination of 2 columns into 1
    # (i.e., find alpha, beta such that col_a * alpha + col_b * beta replaces both)
    # This is more expensive, so only if time permits
    if best_rank == 35 and time.time() < deadline - 30:
        # Try pairwise linear combinations for the rank-35 solution
        U35 = best_U
        V35 = best_V
        W35 = best_W
        R35 = 35
        # Try merging two columns into one with coefficients
        # For speed, only try a subset
        for c1 in range(R35):
            if time.time() > deadline - 15:
                break
            for c2 in range(c1+1, R35):
                if time.time() > deadline - 15:
                    break
                # Try a few coefficient combinations
                for alpha in [1.0, -1.0, 0.5, -0.5]:
                    for beta in [1.0, -1.0]:
                        U2 = U35[:, :R35].copy()
                        V2 = V35[:, :R35].copy()
                        W2 = W35[:, :R35].copy()
                        U2[:, c2] = alpha * U2[:, c1] + beta * U2[:, c2]
                        V2[:, c2] = alpha * V2[:, c1] + beta * V2[:, c2]
                        W2[:, c2] = alpha * W2[:, c1] + beta * W2[:, c1]
                        U2[:, c1] = 0
                        V2[:, c1] = 0
                        W2[:, c1] = 0
                        if verify_fast(U2, V2, W2, R35):
                            mask = np.ones(R35, dtype=bool)
                            mask[c1] = False
                            U34 = U2[:, mask]
                            V34 = V2[:, mask]
                            W34 = W2[:, mask]
                            best_rank = 34
                            best_U = U34
                            best_V = V34
                            best_W = W34
                            break
                    if best_rank == 34:
                        break
            if best_rank == 34:
                break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
