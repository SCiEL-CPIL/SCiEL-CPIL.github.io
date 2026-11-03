s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
if abs(s - 1.0) > tol:
    return False
