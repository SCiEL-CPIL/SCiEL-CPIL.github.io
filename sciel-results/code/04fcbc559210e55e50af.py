C_computed = np.zeros((n, p))
for i in range(n):
    for k in range(p):
        w_row = W[k*n+i, :]  # shape (rank,)
        for j in range(m):
            u_row = U[i*m+j, :]  # shape (rank,)
            v_row = V[j*p+k, :]  # shape (rank,)
            inner = np.sum(u_row * v_row * w_row)
            C_computed[i, k] += A[i, j] * B[j, k] * inner
