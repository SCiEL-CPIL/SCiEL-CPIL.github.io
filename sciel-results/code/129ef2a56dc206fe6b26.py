def _can_merge(clique1, clique2, adj):
    for u in clique1:
        for v in clique2:
            if u == v:
                continue
            if not (adj[u] & (1 << v)):
                return False
    return True

# In solve_instance:
# ...
if _can_merge(cliques[i], cliques[j], adj):
    cliques[i] = list(set(cliques[i]) | set(cliques[j]))
    cliques.pop(j)
    changed = True
    break
