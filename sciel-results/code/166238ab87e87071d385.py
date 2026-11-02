for i in range(n):
    for k in range(p):
        for j in range(m):
            for jp in range(m):
                s = 0.0
                for c in range(r):
                    s += U_arr[i*m + j, c] * V_arr[jp*p + k, c] * W_arr[k*n + i, c]
                target = 1.0 if j == jp else 0.0
                if abs(s - target) > tol:
                    return False
