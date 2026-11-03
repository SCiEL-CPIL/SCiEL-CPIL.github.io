for i in range(n):
    for j in range(m):
        for k in range(p):
            val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
            if abs(val - 1.0) > tol:
                return False
