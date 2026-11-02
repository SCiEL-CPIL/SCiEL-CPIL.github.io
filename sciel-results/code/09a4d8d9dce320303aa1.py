# Inside construct
# After building U_bs, V_bs, W_bs
# Compute target tensor
T_target = np.ones((n, m, p))

# Compute current tensor T
# T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# Vectorized:
# Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
# T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped) 
# Wait, indices:
# U[i, j, r], V[j, k, r], W[k, i, r] -> sum over r
# T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
# This is a specific tensor contraction.

# Check if T == T_target
if not np.allclose(T, T_target):
    # Fallback
    pass

# Precompute column contributions T_r
# T_r[i, j, k] = U[i, j, r] * V[j, k, r] * W[k, i, r]
# This is an outer product of 3 vectors, permuted.
# u_r (n*m,), v_r (m*p,), w_r (p*n,)
# T_r = u_r[:, None, None] * v_r[None, :, None] * w_r[None, None, :] ?
# Need to reshape carefully.

# Elimination loop
remaining = list(range(rank_bs))
T_current = T.copy()
eliminated = 0
while True:
    found = False
    for r in remaining:
        # Compute T_r
        u = U_bs[:, r].reshape(n, m)
        v = V_bs[:, r].reshape(m, p)
        w = W_bs[:, r].reshape(p, n)
        # T_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        T_r = u[:, :, None] * v[None, :, :] * w[None, None, :] 
        # Wait, w is (p, n). We need w[k, i].
        # So T_r[i, j, k] = u[i,j] * v[j,k] * w[k,i]
        # Let's construct T_r:
        # u[i,j] -> shape (n, m, 1)
        # v[j,k] -> shape (1, m, p)
        # w[k,i] -> shape (p, 1, n) -> need to transpose/permute to (1, n, p)?
        # w is (p, n). w[k, i].
        # We want T_r[i, j, k].
        # Term 1: u[i, j]
        # Term 2: v[j, k]
        # Term 3: w[k, i]
        
        # Let's use einsum or broadcasting.
        # u (n, m)
        # v (m, p)
        # w (p, n)
        
        # T_r[i, j, k] = u[i, j] * v[j, k] * w[k, i]
        # This is not a standard outer product.
        # We can compute it as:
        # T_r = np.zeros((n, m, p))
        # for i in range(n):
        #   for j in range(m):
        #     for k in range(p):
        #       T_r[i,j,k] = u[i,j] * v[j,k] * w[k,i]
        
        # Vectorized:
        # u (n, m, 1)
        # v (1, m, p)
        # w (p, 1, n) -> we need w[k, i] at index [k, i].
        # So w_t = w.T (n, p). w_t[i, k] = w[k, i].
        # So we need w_t[i, k] at index [i, k].
        # So w_t shape (n, p) -> (n, 1, p).
        
        w_t = W_bs[:, r].reshape(p, n).T # (n, p), w_t[i, k] = W[k*n+i, r]
        
        T_r = u[:, :, None] * v[None, :, :] * w_t[:, None, :]
        # Check dimensions:
        # u (n, m, 1)
        # v (1, m, p)
        # w_t (n, 1, p)
        # Product: (n, m, p).
        # Index [i, j, k] gets u[i, j] * v[j, k] * w_t[i, k].
        # w_t[i, k] = W[k*n+i, r]. Correct.
        
        T_test = T_current - T_r
        if np.allclose(T_test, T_target, atol=1e-9):
            # Redundant
            T_current = T_test
            remaining.remove(r)
            eliminated += 1
            found = True
            break
    if not found:
        break
