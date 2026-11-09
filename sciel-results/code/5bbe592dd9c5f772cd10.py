for i in range(n):
    for j in range(m):
        for k in range(p):
            val = 0.0
            for c in range(r):
                val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
            if abs(val - 1.0) > 1e-9:
                return False
return True
