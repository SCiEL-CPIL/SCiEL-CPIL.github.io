import numpy as np
import time
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _strassen_50(n, m, p):
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
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

        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)

    if col != rank:
        return None
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # First, get the rank-50 baseline
    baseline = _strassen_50(n, m, p)
    if baseline is None:
        return _schoolbook(n, m, p)

    # Try to improve via LP-based sparse tensor decomposition
    # The idea: model the tensor decomposition as a linear programming problem
    # where we select a subset of pre-generated rank-1 terms
    # Each term has at most 3 non-zero entries per factor (sparse pattern)

    best_result = baseline
    best_rank = 50

    # Generate a pool of sparse rank-1 candidates
    # Each candidate is (u_pattern, v_pattern, w_pattern) where each pattern
    # is a vector with at most 3 non-zero entries
    # We'll use the current rank-50 solution as a warm start

    U50 = np.array(baseline["U"], dtype=float)
    V50 = np.array(baseline["V"], dtype=float)
    W50 = np.array(baseline["W"], dtype=float)

    # Extract non-zero entries from the current solution as candidate patterns
    candidate_terms = []
    for r in range(50):
        u_nz = np.nonzero(U50[:, r])[0]
        v_nz = np.nonzero(V50[:, r])[0]
        w_nz = np.nonzero(W50[:, r])[0]
        if len(u_nz) <= 3 and len(v_nz) <= 3 and len(w_nz) <= 3:
            term = (U50[:, r].copy(), V50[:, r].copy(), W50[:, r].copy())
            candidate_terms.append(term)

    # Try to find a lower-rank decomposition using LP relaxation
    # This is a heuristic: we try to eliminate some terms and see if the rest still works

    # Sort terms by "importance" (e.g., norm of the term)
    term_norms = []
    for term in candidate_terms:
        norm = np.linalg.norm(term[0]) * np.linalg.norm(term[1]) * np.linalg.norm(term[2])
        term_norms.append(norm)

    # Try removing terms one at a time and check if the decomposition still holds
    for trial_remove in range(1, 4):
        if time.time() > deadline:
            break

        # Try removing the least important terms
        indices = list(range(len(candidate_terms)))
        indices.sort(key=lambda i: term_norms[i])

        for remove_idx in indices[:trial_remove]:
            # Build decomposition without this term
            U_test = np.zeros((n * m, len(candidate_terms) - 1), dtype=float)
            V_test = np.zeros((m * p, len(candidate_terms) - 1), dtype=float)
            W_test = np.zeros((p * n, len(candidate_terms) - 1), dtype=float)

            col = 0
            for i, term in enumerate(candidate_terms):
                if i == remove_idx:
                    continue
                U_test[:, col] = term[0]
                V_test[:, col] = term[1]
                W_test[:, col] = term[2]
                col += 1

            if _verify(U_test, V_test, W_test, n, m, p):
                new_rank = len(candidate_terms) - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}

        # Try removing pairs of terms
        if trial_remove >= 2:
            for remove_idx1 in indices[:2]:
                for remove_idx2 in indices[:2]:
                    if remove_idx1 == remove_idx2:
                        continue
                    if time.time() > deadline:
                        break

                    U_test = np.zeros((n * m, len(candidate_terms) - 2), dtype=float)
                    V_test = np.zeros((m * p, len(candidate_terms) - 2), dtype=float)
                    W_test = np.zeros((p * n, len(candidate_terms) - 2), dtype=float)

                    col = 0
                    for i, term in enumerate(candidate_terms):
                        if i == remove_idx1 or i == remove_idx2:
                            continue
                        U_test[:, col] = term[0]
                        V_test[:, col] = term[1]
                        W_test[:, col] = term[2]
                        col += 1

                    if _verify(U_test, V_test, W_test, n, m, p):
                        new_rank = len(candidate_terms) - 2
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}

    return best_result
