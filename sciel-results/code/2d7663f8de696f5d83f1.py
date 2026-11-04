T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
return np.allclose(T, 1.0, atol=tol)
