    def bron_kerbosch_maximal(R, P, X, adj):
        if not P and not X:
            yield R
            return
        
        pivot_set = P | X
        if pivot_set:
            best_u = -1
            best_count = -1
            temp = pivot_set
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                c = popcount(P & adj[u])
                if c > best_count:
                    best_count = c
                    best_u = u
            candidates = P & ~adj[best_u] & ~(1 << best_u) # v must not be u, and not neighbor of u? 
            # Wait, pivot rule: iterate over P \ N(u). 
            # If u is in P, N(u) includes neighbors. u is not a neighbor of itself.
            # So P \ N(u) is correct.
            # But if u is in X, it's not in P, so it won't be in candidates anyway.
            # So candidates = P & ~adj[best_u] is correct.
        else:
            candidates = P

        while candidates:
            v = (candidates & -candidates).bit_length() - 1
            candidates &= candidates - 1
            
            new_R = R | (1 << v)
            # P_new = P \ {v} \ N(v)
            new_P = P & ~adj[v] & ~(1 << v)
            # X_new = X \ {v} \ N(v)
            new_X = X & ~adj[v] & ~(1 << v)
            
            for clique in bron_kerbosch_maximal(new_R, new_P, new_X, adj):
                yield clique
            
            P &= ~(1 << v)
            X |= (1 << v)
