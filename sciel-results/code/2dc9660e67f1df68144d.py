import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def verify_decomp(U, V, W, n=2, m=2, p=9):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_strassen_rank32():
        """Build the known rank-32 decomposition: 4 Strassen blocks (rank 7) + schoolbook (rank 4)."""
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        return terms

    def terms_to_matrices(terms, n=2, m=2, p=9):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    def try_reduce_rank(terms, max_reductions=4):
        """
        Try to remove terms by expressing them as linear combinations of others.
        For each term t, check if its contribution can be cancelled by adjusting
        other terms. We use a greedy approach: try removing each term and see if
        the remaining terms can be adjusted (by adding small corrections) to still
        satisfy the tensor equation.
        
        More practically: try removing pairs of terms that have overlapping structure.
        """
        n, m, p = 2, 2, 9
        current_terms = list(terms)
        rank = len(current_terms)
        
        # Build the tensor from current terms
        def compute_tensor(term_list):
            R = len(term_list)
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for t, (Ud, Vd, Wd) in enumerate(term_list):
                for idx, val in Ud.items():
                    U[idx, t] = val
                for idx, val in Vd.items():
                    V[idx, t] = val
                for idx, val in Wd.items():
                    W[idx, t] = val
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p))
        
        # Try single-term removal: for each term, remove it and check if the
        # residual can be expressed as a linear combination of remaining terms.
        # This is equivalent to checking if the removed term's tensor contribution
        # lies in the span of the remaining terms' tensor contributions.
        
        # For efficiency, precompute each term's tensor contribution
        def term_tensor(Ud, Vd, Wd):
            t = np.zeros((n, m, p))
            for i_u, u_val in Ud.items():
                for j_v, v_val in Vd.items():
                    for k_w, w_val in Wd.items():
                        # U index i_u = i*m+j, V index j_v = j*p+k, W index k_w = k*n+i
                        i, j = divmod(i_u, m)
                        j2, k = divmod(j_v, p)
                        k2, i2 = divmod(k_w, n)
                        if j == j2 and k == k2 and i == i2:
                            t[i, j, k] += u_val * v_val * w_val
            return t
        
        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
        
        # Try removing one term at a time
        for remove_idx in range(rank):
            if time.time() > deadline:
                break
            remaining = [t for t in range(rank) if t != remove_idx]
            # Check if term_tensors[remove_idx] is in the span of the remaining term tensors
            # Flatten and use SVD
            removed_flat = term_tensors[remove_idx].flatten()
            remaining_matrix = np.array([term_tensors[r].flatten() for r in remaining])
            if remaining_matrix.shape[0] == 0:
                continue
            # Check if removed_flat is in the column space of remaining_matrix
            # Use least squares: minimize ||remaining_matrix @ x - removed_flat||
            try:
                x, residuals, rank_r, sv = np.linalg.lstsq(remaining_matrix, removed_flat, rcond=1e-10)
                reconstruction = remaining_matrix @ x
                if np.allclose(reconstruction, removed_flat, atol=1e-8):
                    # The removed term is redundant! Remove it.
                    new_terms = [current_terms[t] for t in remaining]
                    # Verify
                    U, V, W = terms_to_matrices(new_terms)
                    if verify_decomp(U, V, W):
                        current_terms = new_terms
                        rank = len(current_terms)
                        # Recompute term_tensors
                        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
                        # Try again (might find more redundancies)
                        break
            except:
                continue
        
        # Try pair removal: remove two terms simultaneously
        if time.time() > deadline - 10:
            pass
        else:
            rank = len(current_terms)
            term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
            for i in range(rank):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, rank):
                    if time.time() > deadline - 5:
                        break
                    remaining = [t for t in range(rank) if t != i and t != j]
                    if len(remaining) < 2:
                        continue
                    combined_flat = (term_tensors[i] + term_tensors[j]).flatten()
                    remaining_matrix = np.array([term_tensors[r].flatten() for r in remaining])
                    try:
                        x, residuals, rank_r, sv = np.linalg.lstsq(remaining_matrix, combined_flat, rcond=1e-10)
                        reconstruction = remaining_matrix @ x
                        if np.allclose(reconstruction, combined_flat, atol=1e-8):
                            new_terms = [current_terms[t] for t in remaining]
                            U, V, W = terms_to_matrices(new_terms)
                            if verify_decomp(U, V, W):
                                current_terms = new_terms
                                break
                    except:
                        continue
                if time.time() > deadline - 5:
                    break
        
        return current_terms

    # Build the base rank-32 decomposition
    terms = build_strassen_rank32()
    
    # Try to reduce the rank
    reduced_terms = try_reduce_rank(terms)
    
    if len(reduced_terms) < len(terms):
        terms = reduced_terms
    
    # Verify final result
    U, V, W = terms_to_matrices(terms)
    if not verify_decomp(U, V, W):
        # Ultimate fallback: schoolbook rank 36
        terms36 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms36.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        U, V, W = terms_to_matrices(terms36)
    
    return {"U": U, "V": V, "W": W}
