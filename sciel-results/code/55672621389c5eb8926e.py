import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1

        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1

        return U, V, W, rank

    def get_term_contributions(U, V, W, rank):
        """Get the contribution vector of each term across all (i,j,k)."""
        # For term r, contribution at (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We build a matrix where each column is one term's full contribution vector
        n_terms = rank
        n_entries = n * m * p
        contribs = np.zeros((n_entries, n_terms))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(n_terms):
                        contribs[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    idx += 1
        return contribs

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms using linear algebra."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank

        changed = True
        while changed and time_left() > 10:
            changed = False
            U_arr = np.array(U_list, dtype=float)
            V_arr = np.array(V_list, dtype=float)
            W_arr = np.array(W_list, dtype=float)

            # Get contribution vectors
            contribs = get_term_contributions(U_arr, V_arr, W_arr, cur_rank)

            # Try each term: check if its column is in the span of the others
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 10:
                    break

                # Check if removing term_idx still gives valid decomposition
                # Method 1: Direct check - is the term's contribution zero?
                term_contrib = contribs[:, term_idx]
                if np.allclose(term_contrib, 0, atol=1e-12):
                    # Term is redundant, remove it
                    U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                    cur_rank -= 1
                    changed = True
                    break

                # Method 2: Check if term is a linear combination of others
                # Build matrix of other terms' contributions
                if cur_rank > 1:
                    other_indices = [i for i in range(cur_rank) if i != term_idx]
                    other_contribs = contribs[:, other_indices]
                    # Check if term_contrib is in the column space of other_contribs
                    try:
                        # Use least squares to see if we can represent term_contrib
                        sol, residuals, rank_sv, sv = np.linalg.lstsq(other_contribs, term_contrib, rcond=1e-10)
                        # If the residual is small, the term is a linear combination
                        if len(residuals) > 0 and residuals[0] < 1e-10:
                            # Verify by actually removing and checking
                            U_test = [row[:] for row in U_list]
                            V_test = [row[:] for row in V_list]
                            W_test = [row[:] for row in W_list]
                            for i in range(n*m):
                                U_test[i][term_idx] = 0
                            for j in range(m*p):
                                V_test[j][term_idx] = 0
                            for k in range(p*n):
                                W_test[k][term_idx] = 0
                            if verify(U_test, V_test, W_test, cur_rank):
                                U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                                V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                                W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                                cur_rank -= 1
                                changed = True
                                break
                    except np.linalg.LinAlgError:
                        pass

        return U_list, V_list, W_list, cur_rank

    def try_pairwise_merge(U, V, W, rank):
        """Try to find pairs of terms that can be merged into one."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank

        # For each pair of terms, check if their combined contribution can be
        # represented as a single rank-1 term
        # This is more complex, so we'll skip for now and rely on elimination

        return U_list, V_list, W_list, cur_rank

    # Try different Strassen configurations
    # j_pairs: pairs of j indices (from 0..4)
    # k_pairs: pairs of k indices (from 0..5)
    configurations = [
        # 2 j-pairs, 3 k-pairs: rank = 7*6 + 2*1*6 = 42 + 12 = 54
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(3,4),(2,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(3,4)], [(0,2),(1,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(1,2),(3,4)], [(0,2),(1,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,4),(3,5)]),
        ([(0,2),(1,3)], [(0,2),(1,4),(3,5)]),
        # 1 j-pair, 3 k-pairs: rank = 7*3 + 2*3*6 = 21 + 36 = 57
        ([(0,1)], [(0,1),(2,3),(4,5)]),
        ([(0,1)], [(0,1),(2,4),(3,5)]),
        ([(0,1)], [(0,2),(1,3),(4,5)]),
        ([(2,3)], [(0,1),(2,3),(4,5)]),
        ([(2,3)], [(0,1),(2,4),(3,5)]),
        ([(3,4)], [(0,1),(2,3),(4,5)]),
        ([(3,4)], [(0,1),(2,4),(3,5)]),
        # 2 j-pairs, 2 k-pairs: rank = 7*4 + 2*1*6 = 28 + 12 = 40
        # But this leaves 1 k unpaired, which needs special handling
        # Actually with 2 k-pairs, we cover 4 k's, leaving 2 k's
        # The remaining k's for each j still need standard products
        # This is more complex, let's skip for now
    ]

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)

        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue

        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
