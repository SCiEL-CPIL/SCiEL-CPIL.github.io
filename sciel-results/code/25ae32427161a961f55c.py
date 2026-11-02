import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    def verify(U, V, W):
        R = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    # For the target entry, we already checked. Check all others.
        # More efficient: check all entries
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    # Determine if this is a target entry
                    i = u_idx // m
                    j = u_idx % m
                    j2 = v_idx // p
                    k = v_idx % p
                    k2 = w_idx // n
                    i2 = w_idx % n
                    if i == i2 and j == j2 and j == k2 // 1:  # this check is wrong
                        continue
                    # Actually, the target entry for (i,j,k) has u=i*m+j, v=j*p+k, w=k*n+i
                    # So u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i means i2=i, j2=j, k2=k
                    # The entry is a target if i2==i and j2==j and the k from v matches k from w
                    # v_idx = j*p+k => k = v_idx % p
                    # w_idx = k*n+i => k = w_idx // n
                    k_from_v = v_idx % p
                    k_from_w = w_idx // n
                    i_from_u = u_idx // m
                    j_from_u = u_idx % m
                    j_from_v = v_idx // p
                    i_from_w = w_idx % n
                    if i_from_u == i_from_w and j_from_u == j_from_v and k_from_v == k_from_w:
                        # This is a target entry, skip (already verified)
                        continue
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val) > 1e-10:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return U, V, W

    # Try 39-rank hybrid: Strassen for 2x2*2x2 blocks + schoolbook for rest
    U_rows = []
    V_rows = []
    W_rows = []

    def add_term(u_vec, v_vec, w_vec):
        U_rows.append(u_vec)
        V_rows.append(v_vec)
        W_rows.append(w_vec)

    # Strassen block for A[:,0:2] @ B[0:2, k0:k1+1]
    def add_strassen_block(j0, j1, k0, k1):
        # 7 rank-1 terms for 2x2*2x2
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]),
            ([0, 0, 0, 1], [0, 1, -1, 0], [0, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),
            ([0, -1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]),
            ([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, 1]),
        ]
        a_idx = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
        b_idx = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
        c_idx = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
        for u, v, w in terms:
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            for a in range(4):
                if u[a] != 0:
                    U_vec[a_idx[a]] = u[a]
            for b in range(4):
                if v[b] != 0:
                    V_vec[b_idx[b]] = v[b]
            for c in range(4):
                if w[c] != 0:
                    W_vec[c_idx[c]] = w[c]
            add_term(U_vec, V_vec, W_vec)

    # Add 3 Strassen blocks for k pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 1, 0, 1)
    add_strassen_block(0, 1, 2, 3)
    add_strassen_block(0, 1, 4, 5)

    # Schoolbook for A[:,0:2] @ B[0:2, 6] (2x2 * 2x1, rank 4)
    for i in range(n):
        for j in [0, 1]:
            u_idx = i * m + j
            v_idx = j * p + 6
            w_idx = 6 * n + i
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            U_vec[u_idx] = 1
            V_vec[v_idx] = 1
            W_vec[w_idx] = 1
            add_term(U_vec, V_vec, W_vec)

    # Schoolbook for A[:,2] @ B[2,:] (2x1 * 1x7, rank 14)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            U_vec[u_idx] = 1
            V_vec[v_idx] = 1
            W_vec[w_idx] = 1
            add_term(U_vec, V_vec, W_vec)

    U = np.column_stack(U_rows)
    V = np.column_stack(V_rows)
    W = np.column_stack(W_rows)

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
