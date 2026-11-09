<<<<<<< SEARCH
def _build_adj(n, edges):
    """Return adjacency as a set of frozensets for fast membership and a list of edge tuples."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    return adj, edge_set
=======
def _build_adj(n, edges):
    """Return adjacency as a list of sets for fast membership."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj
>>>>>>> REPLACE
