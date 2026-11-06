    def bron_kerbosch_best(R, P, X, adj, best_size, best_clique):
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = set(R)
            return best_size, best_clique
        
        # Pruning
        if len(R) + len(P) <= best_size:
            return best_size, best_clique

        # pivot selection
        pivot_set = P | X
        if pivot_set:
            best_u = None
            best_count = -1
            for u in pivot_set:
                c = len(P & adj.get(u, set()))
                if c > best_count:
                    best_count = c
                    best_u = u
            candidates = P - adj.get(best_u, set())
        else:
            candidates = P
            
        for v in list(candidates):
            new_R = R | {v}
            new_P = P & adj.get(v, set())
            new_X = X & adj.get(v, set())
            best_size, best_clique = bron_kerbosch_best(new_R, new_P, new_X, adj, best_size, best_clique)
            P.remove(v)
            X.add(v)
            
        return best_size, best_clique
