            def bk(R, P, X):
                nonlocal best_clique, best_size
                
                if not P and not X:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = R[:]
                    return
                
                if len(R) + popcount(P) <= best_size:
                    return
                
                if not P:
                    return

                # Find pivot
                max_deg = -1
                u = None
                temp_P = P
                while temp_P:
                    lsb = temp_P & -temp_P
                    v = lsb.bit_length() - 1
                    temp_P ^= lsb
                    deg = popcount(adj[v] & P)
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                
                P_not_Nu = P & ~adj[u]
                
                while P_not_Nu:
                    lsb = P_not_Nu & -P_not_Nu
                    v = lsb.bit_length() - 1
                    P_not_Nu ^= lsb
                    
                    new_R = R + [v]
                    new_P = P & adj[v]
                    new_X = X & adj[v]
                    
                    bk(new_R, new_P, new_X)
                    
                    P &= ~lsb
                    X |= lsb
                    
                    if len(R) + popcount(P) <= best_size:
                        break
