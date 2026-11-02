T = np.zeros((n,m,p))
for c in range(rank):
    u = U[:,c].reshape(n,m)
    v = V[:,c].reshape(m,p)
    w = W[:,c].reshape(p,n)
    T += np.einsum('ij,jk,ki->ijk', u, v, w)
return np.allclose(T, 1.0, atol=1e-9)
