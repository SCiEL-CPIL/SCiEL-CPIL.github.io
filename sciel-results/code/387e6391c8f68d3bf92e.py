def construct():
    import random
    import numpy as np
    from itertools import combinations
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp

    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_maximal_cliques(n, edges):
        adjacency = [set() for _ in range(n)]
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
            
        maximal_cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Pivot selection
            pivot = None
            max_deg = -1
            for u in P | X:
                deg = len(adjacency[u] & P)
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
            
            if pivot is None:
                pivot = next(iter(P | X)) if (P | X) else None
                if pivot is None:
                    return

            for v in list(P - adjacency[pivot]):
                R_new = R | {v}
                P_new = P & adjacency[v]
                X_new = X & adjacency[v]
                bron_kerbosch(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)

        bron_kerbosch(set(), set(range(n)), set())
        return maximal_cliques

    def greedy_partition(n, edges):
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
            
            best_clique = []
            best_edges_count = 0
            
            for start in sorted_verts[:10]:
                if deg[start] == 0:
                    continue
                
                candidates = set()
                for u, v in remaining_edges:
                    if u == start:
                        candidates.add(v)
                    elif v == start:
                        candidates.add(u)
                
                if not candidates:
                    continue
                
                clique = [start]
                current_candidates = candidates
                
                while current_candidates:
                    best_v = max(current_candidates, key=lambda x: deg[x])
                    clique.append(best_v)
                    
                    new_candidates = set()
                    for u, v in remaining_edges:
                        if u == best_v:
                            if v in current_candidates:
                                new_candidates.add(v)
                        elif v == best_v:
                            if u in current_candidates:
                                new_candidates.add(u)
                    
                    current_candidates = new_candidates
                
                clique_set = set(clique)
                c_edges = set()
                valid = True
                for u, v in combinations(clique, 2):
                    e = (min(u,v), max(u,v))
                    if e not in remaining_edges:
                        valid = False
                        break
                    c_edges.add(e)
                
                if valid and len(c_edges) > best_edges_count:
                    best_edges_count = len(c_edges)
                    best_clique = clique
            
            if not best_clique:
                e = next(iter(remaining_edges))
                cliques.append([e[0], e[1]])
                remaining_edges.remove(e)
            else:
                for u, v in combinations(best_clique, 2):
                    e = (min(u,v), max(u,v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                cliques.append(sorted(best_clique))
        
        return cliques

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        if not edges:
            return []
        
        try:
            cliques = get_maximal_cliques(n, edges)
            if not cliques:
                return greedy_partition(n, edges)
            
            # Limit number of cliques for ILP
            # If too many, ILP might be slow. 
            # We can sort by size and take top K, or just try all.
            # Let's try all.
            
            clique_list = list(cliques)
            num_cliques = len(clique_list)
            edge_list = list(edges)
            num_edges = len(edge_list)
            
            rows = []
            cols = []
            data = []
            
            for j, clique in enumerate(clique_list):
                clique_set = set(clique)
                for i, (u, v) in enumerate(edge_list):
                    if u in clique_set and v in clique_set:
                        rows.append(i)
                        cols.append(j)
                        data.append(1)
            
            A = sp.csr_matrix((data, (rows, cols)), shape=(num_edges, num_cliques))
            c = np.ones(num_cliques)
            lb = np.ones(num_edges)
            ub = np.ones(num_edges)
            
            constraints = LinearConstraint(A, lb, ub)
            bounds = Bounds(0, 1)
            integrality = np.ones(num_cliques)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                selected = [i for i in range(num_cliques) if res.x[i] > 0.5]
                result = [sorted(list(clique_list[i])) for i in selected]
                
                # Verify
                covered = set()
                valid = True
                for clique in result:
                    c_set = set(clique)
                    for u, v in combinations(c_set, 2):
                        e = (min(u,v), max(u,v))
                        if e not in edges or e in covered:
                            valid = False
                            break
                        covered.add(e)
                    if not valid:
                        break
                
                if valid and covered == edges:
                    return result
        except Exception:
            pass
            
        return greedy_partition(n, edges)

    return [solve_instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
