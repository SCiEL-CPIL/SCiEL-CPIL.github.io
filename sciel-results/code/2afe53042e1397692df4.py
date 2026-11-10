def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # For each independent set vertex v, N(v) = neighbors in [0,k)
        neighbor_sets = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if (u, v) in edge_set or (v, u) in edge_set]
            neighbor_sets[v] = set(neighbors)
        
        # Candidate cliques from independent set vertices
        candidate_cliques = []
        for v in range(k, n):
            S = neighbor_sets[v]
            if len(S) >= 1:
                clique = sorted(list(S) + [v])
                candidate_cliques.append(clique)
        
        uncovered = set(edge_set)
        selected_cliques = []
        
        # Sort by number of edges they cover (potential)
        def count_coverable(clique):
            cnt = 0
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    if e in uncovered:
                        cnt += 1
            return cnt
        
        # Greedy: repeatedly pick the clique that covers the most uncovered edges
        # and doesn't include any already-covered edge
        candidates = list(candidate_cliques)
        while candidates and uncovered:
            best_clique = None
            best_count = 0
            for clique in candidates:
                # Check if clique has any covered edge
                has_covered = False
                cnt = 0
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in uncovered:
                            cnt += 1
                        else:
                            # Edge is already covered or not an edge
                            # If it's not in edge_set, it's not an edge, that's fine
                            # But if it's in edge_set but not in uncovered, it's covered
                            if e in edge_set:
                                has_covered = True
                                break
                    if has_covered:
                        break
                if has_covered:
                    continue
                if cnt > best_count:
                    best_count = cnt
                    best_clique = clique
            
            if best_clique is None:
                break
            
            selected_cliques.append(best_clique)
            # Remove covered edges
            for i in range(len(best_clique)):
                for j in range(i+1, len(best_clique)):
                    e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                    uncovered.discard(e)
            candidates.remove(best_clique)
        
        # Remaining edges: add as 2-cliques
        for e in sorted(uncovered):
            selected_cliques.append(list(e))
        
        return [list(c) for c in selected_cliques]
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    return [clique_partition(*spec) for spec in specs]
