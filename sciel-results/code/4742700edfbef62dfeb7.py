for i in range(n):
    for j in range(m):
        for k in range(p):
            s = 0
            for r in range(len(U_mat[0])):
                s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
            if abs(s - 1) > 1e-9:
                return False
