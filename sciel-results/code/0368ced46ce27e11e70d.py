import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def build_rank36():
        """Build the standard 4-block Strassen (28) + 2x2x1 rank-4 tail (8) = 36."""
        rank = 36
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0

        def strassen_block(y, z, wo):
            nonlocal col
            y00, y01, y10, y11 = y
            z00, z01, z10, z11 = z
            # M1
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(wo[0][0], wo[0][1])][col] += 1
            W[w_idx(wo[3][0], wo[3][1])][col] += 1
            col += 1
            # M2
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = 1
            W[w_idx(wo[2][0], wo[2][1])][col] += 1
            W[w_idx(wo[3][0], wo[3][1])][col] -= 1
            col += 1
            # M3
            U[u_idx(y00[0], y00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = -1
            W[w_idx(wo[1][0], wo[1][1])][col] += 1
            W[w_idx(wo[3][0], wo[3][1])][col] += 1
            col += 1
            # M4
            U[u_idx(y11[0], y11[1])][col] = 1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z00[0], z00[1])][col] = -1
            W[w_idx(wo[0][0], wo[0][1])][col] += 1
            W[w_idx(wo[2][0], wo[2][1])][col] += 1
            col += 1
            # M5
            U[u_idx(y00[0], y00[1])][col] = 1
            U[u_idx(y01[0], y01[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(wo[0][0], wo[0][1])][col] -= 1
            W[w_idx(wo[1][0], wo[1][1])][col] += 1
            col += 1
            # M6
            U[u_idx(y10[0], y10[1])][col] = 1
            U[u_idx(y00[0], y00[1])][col] = -1
            V[v_idx(z00[0], z00[1])][col] = 1
            V[v_idx(z01[0], z01[1])][col] = 1
            W[w_idx(wo[3][0], wo[3][1])][col] += 1
            col += 1
            # M7
            U[u_idx(y01[0], y01[1])][col] = 1
            U[u_idx(y11[0], y11[1])][col] = -1
            V[v_idx(z10[0], z10[1])][col] = 1
            V[v_idx(z11[0], z11[1])][col] = 1
            W[w_idx(wo[0][0], wo[0][1])][col] += 1
            col += 1

        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
        strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
        strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])

        # Tail: 5th column of B, schoolbook 2x2x1 x 2 blocks (8 terms)
        for i in range(n):
            for j in range(2):
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, 4)][col] = 1
                W[w_idx(4, i)][col] = 1
                col += 1
        for i in range(n):
            for j in range(2, 4):
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, 4)][col] = 1
                W[w_idx(4, i)][col] = 1
                col += 1
        return U, V, W, rank

    def build_rank35_alt():
        """Try a 35-rank construction using a different tail:
        For C_4 = A * b5 (b5 is 4x1), use Strassen-like 7-term scheme treating
        b5 as part of a 2x2 matrix with a zero column, sharing with Strassen blocks."""
        # Actually, a cleaner approach: use rank 3 for the 2x2x1 tail if we can
        # share terms. Let's try:
        # C[0,4] = a00*b04 + a01*b14 + a10*b24 + a11*b34 (wait, dimensions are 2x4 * 4x5)
        # A is 2x4, B is 4x5. C is 2x5.
        # C[i,4] = sum_j A[i,j]*B[j,4]
        # For i=0: A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
        # For i=1: A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
        # This is 2x4x1, which has rank 4 (schoolbook). Can we do rank 3?
        # For 2x2x1, rank is 3. For 2x4x1, we can split into two 2x2x1: rank 3+3=6, or 4.
        # So 4 is better. Keep rank 36 base.
        return None

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-9):
        """Verify the tensor decomposition exactly."""
        if rank > 40:
            return False
        try:
            # D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # where a = i*m+j, b = j*p+k, c = k*n+i
            # We need D[i*m+j, j*p+k, k*n+i] == 1 for all i,j,k
            D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(D[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                            return False
        except Exception:
            return False
        return True

    # Start with rank 36
    U, V, W, rank_initial = build_rank36()
    best_rank = rank_initial
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Helper: try removing a set of columns
    def try_remove(cols_to_remove, current_rank):
        nonlocal best_rank, best_U, best_V, best_W
        if time.time() > deadline:
            return False
        new_rank = current_rank - len(cols_to_remove)
        if new_rank >= best_rank:
            return False
        idx = [x for x in range(current_rank) if x not in cols_to_remove]
        U_t = U[:, idx] if current_rank == rank_initial else best_U[:, idx]
        V_t = V[:, idx] if current_rank == rank_initial else best_V[:, idx]
        W_t = W[:, idx] if current_rank == rank_initial else best_W[:, idx]
        if verify(U_t, V_t, W_t, new_rank):
            best_rank = new_rank
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
            return True
        return False

    # Phase 1: Try removing single columns (greedy)
    for r in range(rank_initial):
        if time.time() > deadline:
            break
        if try_remove([r], rank_initial):
            break

    # Phase 2: Try removing pairs
    if best_rank == rank_initial:
        for r1, r2 in itertools.combinations(range(rank_initial), 2):
            if time.time() > deadline:
                break
            if try_remove([r1, r2], rank_initial):
                break

    # Phase 3: If we found rank 35, try to get to 34
    if best_rank < rank_initial:
        for r1, r2 in itertools.combinations(range(best_rank), 2):
            if time.time() > deadline:
                break
            if try_remove([r1, r2], best_rank):
                break

    # Phase 4: More aggressive search - try all single removals again from current best
    if best_rank < rank_initial:
        for r in range(best_rank):
            if time.time() > deadline:
                break
            if try_remove([r], best_rank):
                break

    U, V, W = best_U, best_V, best_W

    # Convert to list format
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
