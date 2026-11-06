                for i in range(len(nu_list)):
                    if conflict: break
                    v1 = nu_list[i]
                    for j in range(i+1, len(nu_list)):
                        if conflict: break
                        v2 = nu_list[j]
                        common_u = nv & v_adj[v1] & v_adj[v2]
                        if len(common_u) >= 2:
                            conflict = True
                            break
