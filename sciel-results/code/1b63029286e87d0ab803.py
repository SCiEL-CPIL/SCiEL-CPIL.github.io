def _verify(U_mat, V_mat, W_mat, n, m, p):
    rank = U_mat.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_mat[u_idx, c] * V_mat[v_idx, c] * W_mat[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
