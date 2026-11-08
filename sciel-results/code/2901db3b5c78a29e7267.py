import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    # Fast vectorized verification
    def verify_fast(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        """Vectorized verification using numpy"""
        rank = U_mat.shape[1]
        U_r = U_mat.reshape(n_val, m_val, rank)
        V_r = V_mat.reshape(m_val, p_val, rank)
        W_r = W_mat.reshape(p_val, n_val, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-10)

    # Verify the rank-17 solution is correct
    if not verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Precompute reshaped arrays and term contributions for fast residual checks
    rank = 17
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T_full = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)

    # Precompute each term's contribution
    T_terms = np.zeros((rank, n, m, p))
    for r in range(rank):
        T_terms[r] = np.einsum('ij,jk,ki->ijk', U_r[:, :, r], V_r[:, :, r], W_r[:, :, r])

    best_rank = 17
    best_solution = (U.tolist(), V.tolist(), W.tolist())

    # Score terms by sparsity (fewer non-zero entries = easier to absorb)
    term_sparsity = []
    for r in range(17):
        nz_u = np.count_nonzero(U[:, r])
        nz_v = np.count_nonzero(V[:, r])
        nz_w = np.count_nonzero(W[:, r])
        term_sparsity.append(nz_u + nz_v + nz_w)

    # Sort terms by sparsity (try removing sparse terms first)
    sorted_terms = sorted(range(17), key=lambda r: term_sparsity[r])

    for term_to_remove in sorted_terms:
        if time.time() > deadline - 10:
            break

        # Check if the removed term's contribution is zero (redundant term)
        R = T_full - T_terms[term_to_remove]
        if np.allclose(R, 0.0, atol=1e-10):
            U_new = np.column_stack([U[:, r] for r in range(17) if r != term_to_remove])
            V_new = np.column_stack([V[:, r] for r in range(17) if r != term_to_remove])
            W_new = np.column_stack([W[:, r] for r in range(17) if r != term_to_remove])
            if verify_fast(U_new, V_new, W_new, n, m, p):
                best_rank = 16
                best_solution = (U_new.tolist(), V_new.tolist(), W_new.tolist())
                break

        # Try single-entry perturbations in remaining terms to absorb the removed term
        found = False
        for r in range(17):
            if r == term_to_remove or time.time() > deadline - 5:
                continue

            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                if time.time() > deadline - 5:
                    break

                # Try perturbing U entries for term r
                for idx_u in np.nonzero(U[:, r])[0]:
                    U_try = U.copy()
                    U_try[idx_u, r] += delta
                    U_try_r = U_try.reshape(n, m, rank)
                    T_try = np.einsum('ijr,jkr,kir->ijk', U_try_r, V_r, W_r)
                    if np.allclose(T_try, 1.0, atol=1e-10):
                        U_new = np.column_stack([U_try[:, rr] for rr in range(17) if rr != term_to_remove])
                        V_new = np.column_stack([V[:, rr] for rr in range(17) if rr != term_to_remove])
                        W_new = np.column_stack([W[:, rr] for rr in range(17) if rr != term_to_remove])
                        if verify_fast(U_new, V_new, W_new, n, m, p):
                            best_rank = 16
                            best_solution = (U_new.tolist(), V_new.tolist(), W_new.tolist())
                            found = True
                            break
                if found:
                    break

                if not found:
                    # Try perturbing V entries for term r
                    for idx_v in np.nonzero(V[:, r])[0]:
                        V_try = V.copy()
                        V_try[idx_v, r] += delta
                        V_try_r = V_try.reshape(m, p, rank)
                        T_try = np.einsum('ijr,jkr,kir->ijk', U_r, V_try_r, W_r)
                        if np.allclose(T_try, 1.0, atol=1e-10):
                            U_new = np.column_stack([U[:, rr] for rr in range(17) if rr != term_to_remove])
                            V_new = np.column_stack([V_try[:, rr] for rr in range(17) if rr != term_to_remove])
                            W_new = np.column_stack([W[:, rr] for rr in range(17) if rr != term_to_remove])
                            if verify_fast(U_new, V_new, W_new, n, m, p):
                                best_rank = 16
                                best_solution = (U_new.tolist(), V_new.tolist(), W_new.tolist())
                                found = True
                                break
                if found:
                    break

                if not found:
                    # Try perturbing W entries for term r
                    for idx_w in np.nonzero(W[:, r])[0]:
                        W_try = W.copy()
                        W_try[idx_w, r] += delta
                        W_try_r = W_try.reshape(p, n, rank)
                        T_try = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_try_r)
                        if np.allclose(T_try, 1.0, atol=1e-10):
                            U_new = np.column_stack([U[:, rr] for rr in range(17) if rr != term_to_remove])
                            V_new = np.column_stack([V[:, rr] for rr in range(17) if rr != term_to_remove])
                            W_new = np.column_stack([W_try[:, rr] for rr in range(17) if rr != term_to_remove])
                            if verify_fast(U_new, V_new, W_new, n, m, p):
                                best_rank = 16
                                best_solution = (U_new.tolist(), V_new.tolist(), W_new.tolist())
                                found = True
                                break
                if found:
                    break

            if found:
                break

    if best_rank == 16:
        U_out, V_out, W_out = best_solution
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
