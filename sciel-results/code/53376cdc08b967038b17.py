import time
import numpy as np
from fractions import Fraction
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def get_rank28():
        """Known rank-28 decomposition using Strassen blocks + schoolbook."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(rank_offset, j_offset, k_offset):
            a_idx = 0 * m + j_offset
            b_idx = 0 * m + (j_offset + 1)
            c_idx = 1 * m + j_offset
            d_idx = 1 * m + (j_offset + 1)

            e_idx = j_offset * p + k_offset
            f_idx = j_offset * p + (k_offset + 1)
            g_idx = (j_offset + 1) * p + k_offset
            h_idx = (j_offset + 1) * p + (k_offset + 1)

            r = list(range(rank_offset, rank_offset + 7))

            # U
            U[a_idx][r[0]] += 1; U[d_idx][r[0]] += 1
            U[c_idx][r[1]] += 1; U[d_idx][r[1]] += 1
            U[a_idx][r[2]] += 1
            U[d_idx][r[3]] += 1
            U[a_idx][r[4]] += 1; U[b_idx][r[4]] += 1
            U[c_idx][r[5]] += 1; U[a_idx][r[5]] -= 1
            U[b_idx][r[6]] += 1; U[d_idx][r[6]] -= 1

            # V
            V[e_idx][r[0]] += 1; V[h_idx][r[0]] += 1
            V[e_idx][r[1]] += 1
            V[f_idx][r[2]] += 1; V[h_idx][r[2]] -= 1
            V[g_idx][r[3]] += 1; V[e_idx][r[3]] -= 1
            V[h_idx][r[4]] += 1
            V[e_idx][r[5]] += 1; V[f_idx][r[5]] += 1
            V[g_idx][r[6]] += 1; V[h_idx][r[6]] += 1

            # W
            W[k_offset * n + 0][r[0]] += 1
            W[k_offset * n + 0][r[3]] += 1
            W[k_offset * n + 0][r[4]] -= 1
            W[k_offset * n + 0][r[6]] += 1

            W[(k_offset + 1) * n + 0][r[2]] += 1
            W[(k_offset + 1) * n + 0][r[4]] += 1

            W[k_offset * n + 1][r[1]] += 1
            W[k_offset * n + 1][r[3]] += 1

            W[(k_offset + 1) * n + 1][r[0]] += 1
            W[(k_offset + 1) * n + 1][r[1]] -= 1
            W[(k_offset + 1) * n + 1][r[2]] += 1
            W[(k_offset + 1) * n + 1][r[5]] += 1

        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen_block(0, 0, 0)
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_strassen_block(7, 0, 2)
        # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (schoolbook 4 terms)
        U[0][14] = 1; V[4][14] = 1; W[8][14] = 1
        U[1][15] = 1; V[9][15] = 1; W[8][15] = 1
        U[3][16] = 1; V[4][16] = 1; W[9][16] = 1
        U[4][17] = 1; V[9][17] = 1; W[9][17] = 1
        # Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (schoolbook 10 terms)
        for j in range(2):
            for k in range(5):
                col = 18 + j * 5 + k
                U[j * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + j][col] = 1

        return {"U": U, "V": V, "W": W}

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

    def validate(U, V, W):
        """Exact validation using Fractions."""
        rank = len(U[0])
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = Fraction(0)
                    for r in range(rank):
                        lhs += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                    if lhs != 1:
                        return False
        # Check cross terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) == (i2, j2, k2):
                                    continue
                                lhs = Fraction(0)
                                for r in range(rank):
                                    lhs += U_frac[i*m+j][r] * V_frac[j2*p+k2][r] * W_frac[k*n+i2][r]
                                if lhs != 0:
                                    return False
        return True

    def fast_validate(U, V, W, num_tests=20):
        """Fast probabilistic validation using random matrix evaluations."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for _ in range(num_tests):
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_expected = A @ B

            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_col = U_arr[:, r]
                v_col = V_arr[:, r]
                w_col = W_arr[:, r]
                # u_col has n*m entries: index i*m+j
                # v_col has m*p entries: index j*p+k
                # w_col has p*n entries: index k*n+i
                a_vec = u_col.reshape(n, m)
                b_vec = v_col.reshape(m, p)
                w_vec = w_col.reshape(p, n)
                # Term r contributes: sum over i,j,k of a_vec[i,j]*b_vec[j,k]*w_vec[k,i] * A[i,j]*B[j,k]
                # This equals: (A * a_vec) dot-like with (B * b_vec) weighted by w_vec
                # More precisely: sum_{i,j,k} A[i,j]*B[j,k]*a_vec[i,j]*b_vec[j,k]*w_vec[k,i]
                # = sum_i sum_k w_vec[k,i] * (sum_j A[i,j]*a_vec[i,j]*B[j,k]*b_vec[j,k])
                # = sum_i sum_k w_vec[k,i] * (A * a_vec)[i,:] dot (B * b_vec)[:,k]
                # Actually simpler:
                # = sum_{i,j,k} (A[i,j]*a_vec[i,j]) * (B[j,k]*b_vec[j,k]) * w_vec[k,i]
                # Let X = A * a_vec (element-wise), Y = B * b_vec (element-wise)
                # Then term = sum_{i,j,k} X[i,j] * Y[j,k] * w_vec[k,i]
                # = sum_i sum_k w_vec[k,i] * sum_j X[i,j]*Y[j,k]
                # = sum_i sum_k w_vec[k,i] * (X @ Y)[i,k]
                X = A * a_vec
                Y = B * b_vec
                XY = X @ Y
                C_computed += np.sum(w_vec * XY.T)  # w_vec[k,i] * XY[i,k] -> w_vec * XY.T

            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def try_rank27_structured(max_attempts=100000):
        """
        Try to find rank-27 by starting from rank-28 structure and attempting
        to merge/eliminate one term using small integer perturbations.
        Uses polynomial identity testing for speed.
        """
        base = get_rank28()
        U_base = [row[:] for row in base['U']]
        V_base = [row[:] for row in base['V']]
        W_base = [row[:] for row in base['W']]

        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None

            U = [row[:] for row in U_base]
            V = [row[:] for row in V_base]
            W = [row[:] for row in W_base]

            # Try to merge two columns by adding a small perturbation
            # Pick two random columns and try to combine them
            if attempt < 50000:
                # Strategy: pick a pair of columns, add small integers to try to make one redundant
                c1 = np.random.randint(0, 28)
                c2 = np.random.randint(0, 28)
                if c1 == c2:
                    continue

                # Add small perturbation to one column
                delta = np.random.choice([-1, 0, 1], size=28)
                for i in range(28):
                    U[i][c1] += delta[i] if i < n*m else 0
                    V[i][c1] += delta[i] if i < m*p else 0
                    W[i][c1] += delta[i] if i < p*n else 0

                # Try to eliminate column c2 by adjusting
                # Check if the new configuration is valid
                if fast_validate(U, V, W, num_tests=5):
                    if validate(U, V, W):
                        # Remove the redundant column
                        new_U = []
                        new_V = []
                        new_W = []
                        for i in range(n*m):
                            new_U.append(U[i][:c2] + U[i][c2+1:])
                        for i in range(m*p):
                            new_V.append(V[i][:c2] + V[i][c2+1:])
                        for i in range(p*n):
                            new_W.append(W[i][:c2] + W[i][c2+1:])
                        return {"U": new_U, "V": new_V, "W": new_W}
            else:
                # Strategy: random search with structured initialization
                # Start with Strassen-like structure but with random small integers
                U_rand = np.zeros((n*m, 27), dtype=int)
                V_rand = np.zeros((m*p, 27), dtype=int)

                # Fill with Strassen pattern + noise
                for r in range(27):
                    # Small random values
                    for i in range(n*m):
                        U_rand[i, r] = np.random.choice([-1, 0, 0, 0, 1])
                    for j in range(m*p):
                        V_rand[j, r] = np.random.choice([-1, 0, 0, 0, 1])

                # Compute W from the constraint: for each (i,k), sum_j U[i,j,r]*V[j,k,r] * W[k,i,r] = delta
                W_rand = np.zeros((p*n, 27), dtype=float)
                valid = True
                for i in range(n):
                    for k in range(p):
                        coeffs = np.zeros(27)
                        for j in range(m):
                            coeffs += U_rand[i*m+j, :] * V_rand[j*p+k, :]
                        # Need sum_r coeffs[r] * W[k*n+i, r] = 1
                        # And for all other (i',k'): sum_r coeffs[r] * W[k'*n+i', r] = 0
                        # This is a system of linear equations
                        # coeffs dot W_row = delta
                        # Try to solve: if gcd(coeffs) == 1, we can find integer W
                        non_zero = [abs(c) for c in coeffs if c != 0]
                        if non_zero:
                            g = reduce(gcd, non_zero)
                            if g == 1:
                                # Find w such that coeffs dot w = 1
                                # Use extended GCD on two non-zero elements
                                nz_idx = [r for r in range(27) if coeffs[r] != 0]
                                if len(nz_idx) >= 2:
                                    a, b = nz_idx[0], nz_idx[1]
                                    def egcd(a, b):
                                        if b == 0:
                                            return (a, 1, 0)
                                        else:
                                            g, x, y = egcd(b, a % b)
                                            return (g, y, x - (a // b) * y)
                                    g_ab, x, y = egcd(abs(int(coeffs[a])), abs(int(coeffs[b])))
                                    if g_ab == 1:
                                        w_vec = np.zeros(27)
                                        w_vec[a] = x if coeffs[a] > 0 else -x
                                        w_vec[b] = y if coeffs[b] > 0 else -y
                                        # Verify
                                        if np.dot(coeffs, w_vec) == 1:
                                            W_rand[k*n+i, :] = w_vec
                                        else:
                                            valid = False
                                            break
                                    else:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = W_rand.tolist()
                    if fast_validate(U_list, V_list, W_list, num_tests=10):
                        if validate(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try rank-27 search
    result = try_rank27_structured(max_attempts=50000)
    if result is not None:
        return result

    # Fallback to rank-28
    return get_rank28()

