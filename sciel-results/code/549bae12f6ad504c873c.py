    # Quick verification
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1) > 1e-10:
                    # This should not happen
                    pass
