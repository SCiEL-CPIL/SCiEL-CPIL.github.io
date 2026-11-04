"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Matrix multiplication tensor T[i,j,k] = 1 for C[i,k] = sum_j A[i,j]*B[j,k]
    # Convention: W[k*n+i, r] contributes to C[i,k]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Build rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # A is 2x4: indices i*m+j
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11 (left 2 cols)
    a_R = [2, 3, 6, 7]  # right 2 cols

    # B is 4x6: indices j*p+k
    # B col block 0: cols 0,1
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]  # C[0,0]->W[0], C[0,1]->W[2], C[1,0]->W[1], C[1,1]->W[3]

    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)

    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]

    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)

    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]

    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)

    def verify_decomp(U_arr, V_arr, W_arr, rank_val):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank_val):
                        total += U_arr[i*m+j][r] * V_arr[j*p+k][r] * W_arr[k*n+i][r]
                    if total != 1:
                        return False
        return True

    # Verify the Strassen decomposition
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by finding linear dependencies among the 42 rank-1 terms
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Flatten each rank-1 term u_r ⊗ v_r ⊗ w_r into a vector
    # Shape: (n*m, m*p, p*n) = (8, 24, 12) = 2304
    flat_terms = []
    for r in range(rank):
        term = np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        flat_terms.append(term.flatten())

    # Stack into matrix: shape (2304, 42)
    M = np.array(flat_terms).T

    # Check rank of M - if < 42, there are linear dependencies
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # Find singular values that are effectively zero
        tol = 1e-10
        effective_rank = int(np.sum(s_vals > tol))

        if effective_rank < rank:
            # There are (rank - effective_rank) linear dependencies
            # The null space is spanned by the last (rank - effective_rank) rows of Vt_svd
            null_space = Vt_svd[effective_rank:, :]  # shape ((rank-eff_rank), rank)

            # For each null space vector, eliminate one column
            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()

            for null_vec in null_space:
                if time.time() > deadline - 10:
                    break
                # Find the column with the largest absolute coefficient
                candidates = [c for c in range(rank) if c not in eliminated]
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda c: abs(null_vec[c]))
                max_val = abs(null_vec[max_idx])
                if max_val < 1e-8:
                    continue

                # Express column max_idx as linear combination of others
                # null_vec[c] * term_c = 0 for all c
                # term_max_idx = -sum_{c != max_idx} (null_vec[c] / null_vec[max_idx]) * term_c
                for c in candidates:
                    if c == max_idx:
                        continue
                    coeff = -null_vec[c] / null_vec[max_idx]
                    if abs(coeff) > 1e-10:
                        new_U[:, c] += coeff * new_U[:, max_idx]
                        new_V[:, c] += coeff * new_V[:, max_idx]
                        new_W[:, c] += coeff * new_W[:, max_idx]
                eliminated.add(max_idx)

            if eliminated:
                # Reconstruct without eliminated columns
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]

                # Verify the new decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(new_rank):
                                total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                            if abs(total - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    # Convert to list of lists with integer/half-integer entries
                    def to_fraction_list(arr):
                        result = []
                        for row in arr:
                            result.append([])
                            for val in row:
                                # Round to nearest half-integer
                                frac = round(val * 2) / 2
                                # If it's very close to an integer, make it integer
                                if abs(frac - round(frac)) < 1e-10:
                                    frac = int(round(frac))
                                result[-1].append(frac)
                        return result

                    return {
                        "U": to_fraction_list(final_U),
                        "V": to_fraction_list(final_V),
                        "W": to_fraction_list(final_W)
                    }
    except Exception:
        pass

    return {"U": U, "V": "V", "W": W}
# EVOLVE-BLOCK-END
