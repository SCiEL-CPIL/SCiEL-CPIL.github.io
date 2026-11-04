import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    rank = 44

    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]

    col = 0

    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col

    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)

    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        u_idx = 8 + j
        U[u_idx][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[u_idx][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def _verify(Ua, Va, Wa):
        T = np.zeros((n, m, p), dtype=float)
        for r in range(Ua.shape[1]):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=1e-8, rtol=0)

    if not _verify(U_arr, V_arr, W_arr):
        # Fallback: schoolbook rank 48
        rank_s = n * m * p
        Us = [[0] * rank_s for _ in range(dim_u)]
        Vs = [[0] * rank_s for _ in range(dim_v)]
        Ws = [[0] * rank_s for _ in range(dim_w)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][c] = 1
                    Vs[j * p + k][c] = 1
                    Ws[k * n + i][c] = 1
                    c += 1
        return {"U": Us, "V": Vs, "W": Ws}

    # Try to reduce rank by merging pairs of terms
    # For each pair (a, b), check if t_a + t_b has tensor rank 1
    # t_a + t_b = (U_a+U_b) ⊗ (V_a+V_b) ⊗ (W_a+W_b) is NOT correct
    # Instead, we need to check if the sum tensor S = t_a + t_b has rank 1
    # A tensor has rank 1 iff it can be written as u⊗v⊗w
    # For a 3D tensor, rank 1 means all slices are proportional

    # Build the target tensor
    T_target = np.ones((n, m, p), dtype=float)

    # Precompute each term's contribution
    terms = []
    for r in range(rank):
        t = np.einsum('i,j,k->ijk', U_arr[:, r].reshape(n,m), 
                       V_arr[:, r].reshape(m,p), 
                       W_arr[:, r].reshape(p,n))
        terms.append(t)

    # Try merging: for each pair (a,b), compute S = terms[a] + terms[b]
    # Check if S has rank 1 by checking if all 2x2 minors of matricizations are zero
    # A simpler check: S has rank 1 iff S[i,j,k] = u[i]*v[j]*w[k] for some u,v,w
    # This is equivalent to: for any i1,i2,j1,j2,k1,k2:
    # S[i1,j1,k1]*S[i2,j2,k2] = S[i1,j2,k1]*S[i2,j1,k2] = ... (6 conditions)
    # But that's expensive. Use SVD on matricizations.

    merged = []
    used = [False] * rank
    best_rank = rank

    for a in range(rank):
        if used[a]:
            continue
        for b in range(a+1, rank):
            if used[b]:
                continue
            S = terms[a] + terms[b]
            # Check if S has rank 1
            # Matricize along first mode: (n) x (m*p)
            S_mat1 = S.reshape(n, m*p)
            # Rank 1 iff all 2x2 minors are zero, or SVD has only 1 nonzero singular value
            try:
                sv = np.linalg.svd(S_mat1, compute_uv=False)
                if len(sv) >= 2 and sv[1] < 1e-10 * sv[0]:
                    # Check second matricization
                    S_mat2 = S.reshape(m, n*p)
                    sv2 = np.linalg.svd(S_mat2, compute_uv=False)
                    if len(sv2) >= 2 and sv2[1] < 1e-10 * sv2[0]:
                        # S appears to have rank 1
                        # Extract u, v, w from S
                        # S = u ⊗ v ⊗ w
                        # u = S_mat1[:, 0] / w[0] if w[0] != 0
                        # This is tricky. Let's just mark them as merged and verify later.
                        merged.append((a, b))
                        used[a] = True
                        used[b] = True
                        break
            except:
                pass

    # If we found merges, rebuild
    if merged:
        new_U = []
        new_V = []
        new_W = []
        for a, b in merged:
            # The merged term is terms[a] + terms[b], but we need to express it as u⊗v⊗w
            # This is complex. Skip for now and just try removing individual terms.
            pass

    # Alternative: try removing each term and see if we can adjust others
    # This is a hard problem. Let's just return the rank-44 result.

    return {"U": U, "V": V, "W": W}
