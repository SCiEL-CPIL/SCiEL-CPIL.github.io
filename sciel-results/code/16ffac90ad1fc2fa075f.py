target = np.ones((n, m, p))
return np.allclose(T, target, atol=tol)
