U_r = U.reshape(n, m, -1)  # (n, m, r)
V_r = V.reshape(m, p, -1)  # (m, p, r)
W_r = W.reshape(p, n, -1)  # (p, n, r)

val = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
