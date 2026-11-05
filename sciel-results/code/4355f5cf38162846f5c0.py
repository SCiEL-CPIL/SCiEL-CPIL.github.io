val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
if abs(val - 1.0) > 1e-9:
    return False
