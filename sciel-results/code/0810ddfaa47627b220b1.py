import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def build_rank_28_decomposition():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(rank_offset, i0, j0, k0):
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)

            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6

            U[a_idx][r0] += 1; U[d_idx][r0] += 1
            V[e_idx][r0] += 1; V[h_idx][r0] += 1
            W[c00_idx][r0] += 1; W[c11_idx][r0] += 1

            U[c_idx][r1] += 1; U[d_idx][r1] += 1
            V[e_idx][r1] += 1
            W[c10_idx][r1] += 1

            U[a_idx][r2] += 1
            V[f_idx][r2] += 1; V[h_idx][r2] -= 1
            W[c01_idx][r2] += 1; W[c11_idx][r2] -= 1

            U[d_idx][r3] += 1
            V[g_idx][r3] += 1; V[e_idx][r3] -= 1
            W[c00_idx][r3] -= 1; W[c10_idx][r3] += 1

            U[a_idx][r4] += 1; U[b_idx][r4] += 1
            V[h_idx][r4] += 1
            W[c01_idx][r4] += 1; W[c11_idx][r4] += 1

            U[c_idx][r5] += 1; U[a_idx][r5] -= 1
            V[e_idx][r5] += 1; V[f_idx][r5] += 1
            W[c00_idx][r5] += 1

            U[b_idx][r6] += 1; U[d_idx][r6] -= 1
            V[g_idx][r6] += 1; V[h_idx][r6] += 1
            W[c00_idx][r6] -= 1; W[c11_idx][r6] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(7, 0, 0, 2)

        col = 14
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        for i in range(2):
            for k in range(5):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        assert col == rank
        return {"U": U, "V": V, "W": W}

    def validate_float(U, V, W, tol=1e-9):
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_recon, 1.0, atol=tol)

    def validate_exact(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def try_rank_27_via_linear_dependency():
        """
        Check if the rank-28 decomposition has a linear dependency among its terms.
        If removing one term still satisfies the tensor identity, we have rank 27.
        Use efficient numpy-based checks.
        """
        if time.time() > deadline - 30:
            return None

        base = build_rank_28_decomposition()
        U_base = np.array(base["U"], dtype=float)
        V_base = np.array(base["V"], dtype=float)
        W_base = np.array(base["W"], dtype=float)

        # For each column, check if it's linearly dependent on others
        # by testing if the tensor contribution can be replicated
        for elim_col in range(28):
            if time.time() > deadline - 25:
                break

            U_27 = np.delete(U_base, elim_col, axis=1)
            V_27 = np.delete(V_base, elim_col, axis=1)
            W_27 = np.delete(W_base, elim_col, axis=1)

            U_3d = U_27.reshape(n, m, 27)
            V_3d = V_27.reshape(m, p, 27)
            W_3d = W_27.reshape(p, n, 27)
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

            if np.allclose(T_recon, 1.0, atol=1e-8):
                # Convert to exact fractions for final verification
                U_frac = [[Fraction(int(round(x))) for x in row] for row in U_27.tolist()]
                V_frac = [[Fraction(int(round(x))) for x in row] for row in V_27.tolist()]
                W_frac = [[Fraction(int(round(x))) for x in row] for row in W_27.tolist()]

                if validate_exact(U_frac, V_frac, W_frac):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

        return None

    def try_rank_27_via_coefficient_search():
        """
        Try small integer coefficient modifications to the rank-28 decomposition
        to find a rank-27 solution. Focus on the schoolbook blocks where
        dependencies are most likely.
        """
        if time.time() > deadline - 20:
            return None

        base = build_rank_28_decomposition()
        U_base = [row[:] for row in base["U"]]
        V_base = [row[:] for row in base["V"]]
        W_base = [row[:] for row in base["W"]]

        # The last 10 terms (cols 18-27) are schoolbook: A[i,2]*B[2,k] -> C[i,k]
        # Try to see if any can be expressed via the others with small adjustments
        for elim_col in range(18, 28):
            if time.time() > deadline - 15:
                break

            U_27 = []
            for row in U_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                U_27.append(new_row)
            V_27 = []
            for row in V_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                V_27.append(new_row)
            W_27 = []
            for row in W_base:
                new_row = row[:elim_col] + row[elim_col+1:]
                W_27.append(new_row)

            if validate_float(U_27, V_27, W_27, tol=1e-8):
                U_frac = [[Fraction(x) for x in row] for row in U_27]
                V_frac = [[Fraction(x) for x in row] for row in V_27]
                W_frac = [[Fraction(x) for x in row] for row in W_27]
                if validate_exact(U_frac, V_frac, W_frac):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

        # Try modifying Strassen blocks with small perturbations
        for mod_col in range(0, 14):
            if time.time() > deadline - 10:
                break
            for delta in [-1, 1]:
                if time.time() > deadline - 10:
                    break
                for mat_name, mat_data in [('U', U_base), ('V', V_base), ('W', W_base)]:
                    if time.time() > deadline - 10:
                        break
                    dims = {'U': n*m, 'V': m*p, 'W': p*n}[mat_name]
                    for idx in range(dims):
                        orig = mat_data[idx][mod_col]
                        if orig == 0:
                            continue
                        mat_data[idx][mod_col] = orig + delta

                        U_check = U_base if mat_name == 'U' else U_base
                        V_check = V_base if mat_name == 'V' else V_base
                        W_check = W_base if mat_name == 'W' else W_base

                        if validate_float(U_check, V_check, W_check, tol=1e-8):
                            U_frac = [[Fraction(x) for x in row] for row in U_check]
                            V_frac = [[Fraction(x) for x in row] for row in V_check]
                            W_frac = [[Fraction(x) for x in row] for row in W_check]
                            if validate_exact(U_frac, V_frac, W_frac):
                                return {"U": U_frac, "V": V_frac, "W": W_frac}

                        mat_data[idx][mod_col] = orig

        return None

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try rank 27 first
    result_27 = try_rank_27_via_linear_dependency()
    if result_27 is not None:
        return result_27

    result_27b = try_rank_27_via_coefficient_search()
    if result_27b is not None:
        return result_27b

    # Fall back to rank 28
    result_28 = build_rank_28_decomposition()
    if validate_float(result_28["U"], result_28["V"], result_28["W"]):
        U_frac = [[Fraction(x) for x in row] for row in result_28["U"]]
        V_frac = [[Fraction(x) for x in row] for row in result_28["V"]]
        W_frac = [[Fraction(x) for x in row] for row in result_28["W"]]
        if validate_exact(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}

    return get_schoolbook()
# EVOLVE-BLOCK-END
