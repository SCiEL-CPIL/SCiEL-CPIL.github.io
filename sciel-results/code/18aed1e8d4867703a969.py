"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 140
    
    def verify(U, V, W, r):
        """Check if U,V,W of rank r correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T_reconstructed = np.zeros((n, m, p))
        for c in range(r):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)
    
    # Block decomposition:
    # i: {0,1}, {2} -> sizes 2, 1
    # j: {0,1}, {2,3} -> sizes 2, 2
    # k: {0,1}, {2,3}, {4,5} -> sizes 2, 2, 2
    
    # 6 blocks of type (2,2,2) using Strassen rank-7
    # 6 blocks of type (1,2,2) using schoolbook rank-4
    # Total rank: 6*7 + 6*4 = 42 + 24 = 66
    
    total_rank = 66
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
    # Global: i = i0 + i', j = j0 + j', k = k0 + k'
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Each entry is (local_i, local_j) for U, (local_j, local_k) for V, (local_k, local_i) for W
        ([((0, 0), 1), ((1, 1), 1)],
         [((0, 0), 1), ((1, 1), 1)],
         [((0, 0), 1), ((1, 1), 1)]),
        ([((1, 0), 1), ((1, 1), 1)],
         [((0, 0), 1)],
         [((0, 1), 1)]),
        ([((0, 0), 1)],
         [((1, 0), 1)],
         [((0, 0), 1)]),
        ([((1, 1), 1)],
         [((1, 1), 1)],
         [((1, 1), 1)]),
        ([((0, 0), 1), ((0, 1), 1)],
         [((1, 1), 1)],
         [((1, 0), 1)]),
        ([((1, 0), 1), ((0, 0), -1)],
         [((0, 0), 1), ((0, 1), 1)],
         [((0, 1), 1)]),
        ([((0, 1), 1), ((1, 1), -1)],
         [((1, 0), 1), ((1, 1), 1)],
         [((1, 0), 1)]),
    ]
    
    # Schoolbook 1x2x2 decomposition (4 terms)
    # Local: i'=0, j' in {0,1}, k' in {0,1}
    schoolbook_122_terms = [
        ([((0, 0), 1)], [((0, 0), 1)], [((0, 0), 1)]),
        ([((0, 1), 1)], [((1, 0), 1)], [((0, 0), 1)]),
        ([((0, 0), 1)], [((0, 1), 1)], [((1, 0), 1)]),
        ([((0, 1), 1)], [((1, 1), 1)], [((1, 0), 1)]),
    ]
    
    # i partitions: (0, [0,1]), (1, [2])
    i_parts = [(0, [0, 1]), (1, [2])]
    # j partitions: (0, [0,1]), (1, [2,3])
    j_parts = [(0, [0, 1]), (1, [2, 3])]
    # k partitions: (0, [0,1]), (1, [2,3]), (2, [4,5])
    k_parts = [(0, [0, 1]), (1, [2, 3]), (2, [4, 5])]
    
    for i_idx, i_vals in i_parts:
        i0 = i_vals[0]
        i_size = len(i_vals)
        for j_idx, j_vals in j_parts:
            j0 = j_vals[0]
            j_size = len(j_vals)
            for k_idx, k_vals in k_parts:
                k0 = k_vals[0]
                k_size = len(k_vals)
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Strassen 2x2x2
                    for term in strassen_terms:
                        u_entries, v_entries, w_entries = term
                        for (li, lj), val in u_entries:
                            gi = i0 + li
                            gj = j0 + lj
                            U[gi * m + gj, col] = val
                        for (lj, lk), val in v_entries:
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk, col] = val
                        for (lk, li), val in w_entries:
                            gk = k0 + lk
                            gi = i0 + li
                            W[gk * n + gi, col] = val
                        col += 1
                elif i_size == 1 and j_size == 2 and k_size == 2:
                    # Schoolbook 1x2x2
                    for term in schoolbook_122_terms:
                        u_entries, v_entries, w_entries = term
                        for (li, lj), val in u_entries:
                            gi = i0 + li
                            gj = j0 + lj
                            U[gi * m + gj, col] = val
                        for (lj, lk), val in v_entries:
                            gj = j0 + lj
                            gk = k0 + lk
                            V[gj * p + gk, col] = val
                        for (lk, li), val in w_entries:
                            gk = k0 + lk
                            gi = i0 + li
                            W[gk * n + gi, col] = val
                        col += 1
    
    assert col == total_rank, f"Expected {total_rank} terms, got {col}"
    
    # Verify
    if not verify(U, V, W, total_rank):
        # Fall back to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, c] = 1
                    V_school[j * p + k, c] = 1
                    W_school[k * n + i, c] = 1
                    c += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Convert to lists (entries are already integers)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
