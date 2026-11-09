    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
                C_recon[i,k] += s
    