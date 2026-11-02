for i in range(n):
    for j in range(m):
        for k in range(p):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
            if abs(val - 1.0) > 1e-9:
                return False
