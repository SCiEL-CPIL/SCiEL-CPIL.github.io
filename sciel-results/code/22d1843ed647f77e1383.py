ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])

U_idx = ii * m + jj
V_idx = jj * p + kk
W_idx = kk * n + ii

U_sel = U[U_idx]
V_sel = V[V_idx]
W_sel = W[W_idx]
vals = np.sum(U_sel * V_sel * W_sel, axis=1)
