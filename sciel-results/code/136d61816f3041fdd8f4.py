import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = 2, 3, 9

    # Build the rank-50 decomposition (Strassen on 2x2 blocks, (2+1) split)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def full_decomp(Um, Vm, Wm):
        return np.einsum('ir,jr,kr->ijk', Um, Vm, Wm)

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Verify the base rank-50 is correct
    if not np.allclose(full_decomp(U_np, V_np, W_np), T, atol=1e-9):
        return {"U": U, "V": V, "W": W}

    # Build active pairs: (i, j, k) where T[i,j,k] != 0
    active_ijk = []
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                active_ijk.append((i, j, k))
    
    N_act = len(active_ijk)  # 27

    # Build constraint matrix: M[row, r] = U[i,r]*V[j,r]*W[k,r] for each active (i,j,k)
    M_mat = np.zeros((N_act, rank))
    for idx, (i, j, k) in enumerate(active_ijk):
        for r in range(rank):
            M_mat[idx, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # The target is ones(N_act)
    b_target = np.ones(N_act)
    
    # Verify: M_mat @ ones(rank) should equal b_target
    # (This should hold if the decomposition is correct)
    
    # Rank reduction: find minimum subset S of columns such that
    # b_target is in the column space of M_mat[:, S]
    # i.e., there exists c with M_mat[:, S] @ c = b_target
    
    def can_represent(cols):
        """Check if b_target can be represented as a linear combination of M_mat[:, cols]"""
        if len(cols) == 0:
            return False
        A = M_mat[:, cols]
        try:
            c, residuals, rank_s, sv = np.linalg.lstsq(A, b_target, rcond=1e-12)
            res = np.linalg.norm(A @ c - b_target)
            return res < 1e-10
        except Exception:
            return False
    
    def verify_decomposition(mask):
        """Verify the decomposition with the given mask is correct."""
        U_new = U_np[:, mask]
        V_new = V_np[:, mask]
        W_new = W_np[:, mask]
        decomp = full_decomp(U_new, V_new, W_new)
        return np.allclose(decomp, T, atol=1e-9)
    
    # Greedy rank reduction
    best_mask = np.ones(rank, dtype=bool)
    best_rank_val = rank
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try dropping each remaining column
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            if not best_mask[r]:
                continue
            new_mask = best_mask.copy()
            new_mask[r] = False
            cols = np.where(new_mask)[0]
            if can_represent(cols):
                if verify_decomposition(new_mask):
                    best_mask = new_mask
                    best_rank_val = int(new_mask.sum())
                    improved = True
    
    if best_rank_val < rank:
        U_out = U_np[:, best_mask]
        V_out = V_np[:, best_mask]
        W_out = W_np[:, best_mask]
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return {"U": U, "V": V, "W": W}
