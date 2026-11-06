    def solve_transversal(n, adj, cliques):
        if not cliques:
            return []
        
        # Use branch and bound
        sol = branch_and_bound(n, adj, cliques, time_limit=5.0)
        
        # Verify and return
        if not sol:
            return []
        
        # Final local search polish
        selected = set(sol)
        vertex_cliques = {v: [] for v in range(n)}
        for i, C in enumerate(cliques):
            for v in C:
                vertex_cliques[v].append(i)
        
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                safe = True
                for ci in vertex_cliques[v]:
                    C = cliques[ci]
                    has_other = False
                    for u in C:
                        if u != v and u in selected:
                            has_other = True
                            break
                    if not has_other:
                        safe = False
                        break
                if safe:
                    selected.remove(v)
                    changed = True
        
        return sorted(selected)
    