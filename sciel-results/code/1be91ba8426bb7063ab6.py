def verify(U_list, V_list, W_list):
    n, m, p = 3, 4, 6
    R = len(U_list[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
            val = 0
            for r in range(R):
                val += U_list[i*m+k][r] * V_list[k*p+j][r] * W_list[j*n+i][r]
            if abs(val - 1) > 1e-9:
                return False
    return True
