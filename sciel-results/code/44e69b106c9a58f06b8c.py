<<<<<<< SEARCH
            M_cur = np.zeros((nmp, r_cur))
            for l in range(r_cur):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M_cur[i * m * p + j * p + k, l] = U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
=======
            M_cur = np.einsum('il,jl,kil->ijk', U_cur, V_cur, W_cur)
            M_cur = M_cur.reshape(nmp, r_cur)
>>>>>>> REPLACE
