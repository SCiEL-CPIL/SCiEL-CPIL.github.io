<<<<<<< SEARCH
def _verify(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1.0
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - expected) > 1e-6:
                    return False
    return True
=======
def _verify(U, V, W, n, m, p):
    U = np.asarray(U); V = np.asarray(V); W = np.asarray(W)
    rank = U.shape[1]
    # Vectorized verification: compute T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Reshape for efficient computation
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For each r, outer product U[:,r] * V[:,r] * W[:,r] contributes to tensor
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let's compute via: for each r, form matrix A_r[i,j] = U[i*m+j,r], B_r[j,k] = V[j*p+k,r]
    # C_r[i,k] = sum_j A_r[i,j]*B_r[j,k], then T[i,k] = sum_r C_r[i,k]*W[k*n+i,r]
    # Better: T[i,k] = sum_r W[k*n+i,r] * (U_r @ V_r)[i,k] where U_r is n x m, V_r is m x p
    T = np.zeros((n, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r]  # shape (p*n,)
        Cr = Ur @ Vr  # (n, p)
        # T[i,k] += Wr[k*n+i] * Cr[i,k]
        T += Cr * Wr.reshape(p, n).T  # Wr.reshape(p,n) has [k,i] = W[k*n+i]
    # Check T[i,k] == 1 for all i,k
    return np.allclose(T, 1.0, atol=1e-6)
>>>>>>> REPLACE

Now let me also add a post-selection column elimination pass to try to remove any redundant columns:

