U_t = U.reshape(n, m, rank)
V_t = V.reshape(m, p, rank)
W_t = W.reshape(p, n, rank)
T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
