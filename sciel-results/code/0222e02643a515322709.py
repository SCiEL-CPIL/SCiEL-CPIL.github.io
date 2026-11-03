        for i in range(n):
            u = p[i]
            b_uu = b[u, u]
            a_ii = a[i, i]
            
            D = a[i, :] @ b
            D_u = D[u]
            E = a @ b[u, :]
            S = D[p] - D_u - diag_C + E
            
            for k in range(n):
                if i == k:
                    continue
                
                v = p[k]
                
                a_ki = a[k, i]
                a_ik = a[i, k]
                a_kk = a[k, k]
                
                b_vu = b[v, u]
                b_vv = b[v, v]
                b_uv = b[u, v]
                
                t1 = -(a_ii - a_ki) * (b_vu - b_uu)
                t2 = -(a_ik - a_kk) * (b_vv - b_uv)
                t3 = (a_ik - a_ki) * (b_vu - b_uv)
                t4 = (a_ii - a_kk) * (b_vv - b_uu)
                
                delta_c = S[k] + t1 + t2 + t3 + t4
                # ...
