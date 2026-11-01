import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    deadline = time.time() + 140  # safety margin before 150s limit

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff

    def strassen_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx

            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff

            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = b_row_offset * 2 + r
                g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff

            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_r = r
                g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff

            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    # Build the standard rank-28 decomposition
    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)

    # Verify the rank-28 solution works
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), "Rank-28 verification failed"

    # Attempt to find a rank-27 decomposition by merging pairs of terms
    # The idea: if two terms have compatible structures, we might be able to
    # combine them into one term, reducing rank by 1.
    
    # For each pair of columns, check if we can merge them
    # This is a heuristic search - we try various merging strategies
    
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    # Try merging pairs of terms
    # A merge is valid if the merged term still correctly computes the product
    # We'll try simple linear combinations
    
    # Precompute the contribution of each term to the output tensor
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # For the merge to work, the sum of contributions must match
    
    # Try merging each pair (a, b) where a < b
    # The merged column would be U[:,a] + U[:,b], V[:,a] + V[:,b], W[:,a] + W[:,b]
    # This works if U[:,a]*V[:,a]*W[:,a] + U[:,b]*V[:,b]*W[:,b] = 
    #   (U[:,a]+U[:,b])*(V[:,a]+V[:,b])*(W[:,a]+W[:,b])
    # which expands to:
    #   UVW_a + UVW_b = UVW_a + UVW_b + U_a*V_b*W_a + U_a*V_a*W_b + U_b*V_a*W_a + U_b*V_b*W_a + U_a*V_b*W_b + U_b*V_a*W_b + U_b*V_b*W_b
    # So we need: U_a*V_b*W_a + U_a*V_a*W_b + U_b*V_a*W_a + U_b*V_b*W_a + U_a*V_b*W_b + U_b*V_a*W_b + U_b*V_b*W_b = 0
    # This is very restrictive. Let's just try it and verify.
    
    for a in range(rank):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, rank):
            # Try merging columns a and b
            U_merged = U.copy()
            V_merged = V.copy()
            W_merged = W.copy()
            
            # Merge: new column a = old a + old b, remove column b
            U_merged[:, a] = U[:, a] + U[:, b]
            V_merged[:, a] = V[:, a] + V[:, b]
            W_merged[:, a] = W[:, a] + W[:, b]
            
            # Remove column b
            U_merged = np.delete(U_merged, b, axis=1)
            V_merged = np.delete(V_merged, b, axis=1)
            W_merged = np.delete(W_merged, b, axis=1)
            
            new_rank = rank - 1
            
            # Verify
            Cf2 = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for cc in range(new_rank):
                        for jj in range(m):
                            acc += U_merged[ii * m + jj, cc] * V_merged[jj * p + kk, cc] * W_merged[kk * n + ii, cc]
                    Cf2[ii, kk] = acc
            
            if np.allclose(C, Cf2):
                best_U = U_merged
                best_V = V_merged
                best_W = W_merged
                best_rank = new_rank
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no simple merge worked, return the rank-28 solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])

