import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]

    # Build the base decomposition: 5 Strassen blocks (rank 35) + 20 schoolbook columns (rank 20) = rank 55
    base_rank = 55
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]

    col = 0
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1

    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute all nonzero and zero positions
    all_indices = np.argwhere(np.ones((n * m, m * p, p * n), dtype=bool))
    nz_mask = T[all_indices[:, 0], all_indices[:, 1], all_indices[:, 2]] == 1.0
    nz_indices = all_indices[nz_mask]
    zero_indices = all_indices[~nz_mask]

    def verify_decomposition(U_m, V_m, W_m, check_all_zeros=True):
        """Verify that the decomposition exactly matches the target tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_m[i * m + j] * V_m[j * p + k], W_m[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        if check_all_zeros:
            for idx in range(len(zero_indices)):
                a, b, c = zero_indices[idx]
                lhs = np.dot(U_m[a] * V_m[b], W_m[c])
                if not np.isclose(lhs, 0.0, atol=1e-9):
                    return False
        return True

    def is_redundant_matrix(U_m, V_m, W_m, r):
        """
        Check if column r is redundant using matrix operations.
        A column r is redundant if:
        - For all nonzero positions (a,b,c): sum_{s!=r} U[a,s]*V[b,s]*W[c,s] == 1
        - For all zero positions (a,b,c): sum_{s!=r} U[a,s]*V[b,s]*W[c,s] == 0
        
        This is equivalent to checking that removing column r doesn't change the tensor.
        We check: for each position, the contribution of column r equals the total minus the target.
        """
        rank = U_m.shape[1]
        if rank == 0:
            return False
        
        u_r = U_m[:, r]
        v_r = V_m[:, r]
        w_r = W_m[:, r]
        
        # For nonzero positions: need sum_{s!=r} U[a,s]*V[b,s]*W[c,s] == 1
        # i.e., total - contrib_r == 1, so contrib_r == total - 1
        a_nz = nz_indices[:, 0]
        b_nz = nz_indices[:, 1]
        c_nz = nz_indices[:, 2]
        
        total_nz = np.einsum('ir,ir,ir->i', U_m[a_nz, :], V_m[b_nz, :], W_m[c_nz, :])
        contrib_nz = u_r[a_nz] * v_r[b_nz] * w_r[c_nz]
        
        if not np.allclose(total_nz - contrib_nz, 1.0, atol=1e-9):
            return False
        
        # For zero positions: need sum_{s!=r} U[a,s]*V[b,s]*W[c,s] == 0
        # i.e., total - contrib_r == 0, so contrib_r == total
        a_z = zero_indices[:, 0]
        b_z = zero_indices[:, 1]
        c_z = zero_indices[:, 2]
        
        total_z = np.einsum('ir,ir,ir->i', U_m[a_z, :], V_m[b_z, :], W_m[c_z, :])
        contrib_z = u_r[a_z] * v_r[b_z] * w_r[c_z]
        
        if not np.allclose(total_z - contrib_z, 0.0, atol=1e-9):
            return False
        
        return True

    def is_redundant_subset(U_m, V_m, W_m, r, zero_sample):
        """Check redundancy using a subset of zero positions for speed."""
        rank = U_m.shape[1]
        if rank == 0:
            return False
        
        u_r = U_m[:, r]
        v_r = V_m[:, r]
        w_r = W_m[:, r]
        
        a_nz = nz_indices[:, 0]
        b_nz = nz_indices[:, 1]
        c_nz = nz_indices[:, 2]
        
        total_nz = np.einsum('ir,ir,ir->i', U_m[a_nz, :], V_m[b_nz, :], W_m[c_nz, :])
        contrib_nz = u_r[a_nz] * v_r[b_nz] * w_r[c_nz]
        
        if not np.allclose(total_nz - contrib_nz, 1.0, atol=1e-9):
            return False
        
        a_z = zero_sample[:, 0]
        b_z = zero_sample[:, 1]
        c_z = zero_sample[:, 2]
        
        total_z = np.einsum('ir,ir,ir->i', U_m[a_z, :], V_m[b_z, :], W_m[c_z, :])
        contrib_z = u_r[a_z] * v_r[b_z] * w_r[c_z]
        
        if not np.allclose(total_z - contrib_z, 0.0, atol=1e-9):
            return False
        
        return True

    # Greedy removal of redundant columns - try multiple passes with different orders
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    best_rank = base_rank

    # Pass 1: forward order, full zero check
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    r = 0
    while r < rank:
        if time.time() > deadline - 80:
            break
        if is_redundant_matrix(U_work, V_work, W_work, r):
            U_work = np.delete(U_work, r, axis=1)
            V_work = np.delete(V_work, r, axis=1)
            W_work = np.delete(W_work, r, axis=1)
            rank -= 1
            r = 0  # restart from beginning
        else:
            r += 1

    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_work.copy()
        best_rank = rank

    # Pass 2: reverse order, full zero check
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    r = rank - 1
    while r >= 0:
        if time.time() > deadline - 70:
            break
        if is_redundant_matrix(U_work, V_work, W_work, r):
            U_work = np.delete(U_work, r, axis=1)
            V_work = np.delete(V_work, r, axis=1)
            W_work = np.delete(W_work, r, axis=1)
            rank -= 1
        else:
            r -= 1

    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_work.copy()
        best_rank = rank

    # Pass 3: random order, full zero check
    rng = np.random.RandomState(42)
    rank = base_rank
    U_work = U_np.copy()
    V_work = V_np.copy()
    W_work = W_np.copy()
    order = list(range(rank))
    rng.shuffle(order)
    for r_orig in order:
        if time.time() > deadline - 60:
            break
        if r_orig < rank:
            if is_redundant_matrix(U_work, V_work, W_work, r_orig):
                U_work = np.delete(U_work, r_orig, axis=1)
                V_work = np.delete(V_work, r_orig, axis=1)
                W_work = np.delete(W_work, r_orig, axis=1)
                rank -= 1

    if rank < best_rank:
        best_U = U_work.copy()
        best_V = V_work.copy()
        best_W = W_np.copy()
        best_rank = rank

    # Pass 4: Continue from best, try different random seeds
    for seed in [123, 456, 789, 101, 202]:
        if time.time() > deadline - 20:
            break
        rng2 = np.random.RandomState(seed)
        rank = best_rank
        U_work = best_U.copy()
        V_work = best_V.copy()
        W_work = best_W.copy()
        order = list(range(rank))
        rng2.shuffle(order)
        for r_orig in order:
            if time.time() > deadline - 15:
                break
            if r_orig < rank:
                if is_redundant_matrix(U_work, V_work, W_work, r_orig):
                    U_work = np.delete(U_work, r_orig, axis=1)
                    V_work = np.delete(V_work, r_orig, axis=1)
                    W_work = np.delete(W_work, r_orig, axis=1)
                    rank -= 1

        if rank < best_rank:
            best_U = U_work.copy()
            best_V = V_work.copy()
            best_W = W_work.copy()
            best_rank = rank

    # Final verification with all zeros
    if not verify_decomposition(best_U, best_V, best_W, check_all_zeros=True):
        # Fallback to base decomposition
        best_U = U_np
        best_V = V_np
        best_W = W_np
        best_rank = base_rank

    # Convert to lists with integer/half-integer entries
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2

    return {"U": U_out, "V": V_out, "W": W_out}
