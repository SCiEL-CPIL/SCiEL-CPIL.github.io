import random
import itertools

def is_valid(edges, n=15):
    # Build adjacency list and degree
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Check all paths of length 3
    # s-u-v-t
    # Iterate over all edges (u,v)
    for u, v in edges:
        # s in adj[u] \ {v}
        # t in adj[v] \ {u}
        # s != t
        for s in adj[u]:
            if s == v: continue
            for t in adj[v]:
                if t == u: continue
                if s == t: continue
                if degrees[s] == degrees[t]:
                    return False
    return True

def construct():
    n = 15
    # Start with K_{7,8}
    a_size = 7
    b_size = 8
    partition_a = list(range(0, a_size))
    partition_b = list(range(a_size, n))
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    # List of potential extra edges (within A or within B)
    extra_edges = []
    for i in range(len(partition_a)):
        for j in range(i+1, len(partition_a)):
            extra_edges.append([partition_a[i], partition_a[j]])
    for i in range(len(partition_b)):
        for j in range(i+1, len(partition_b)):
            extra_edges.append([partition_b[i], partition_b[j]])
            
    # Try to add extra edges
    # We want to maximize count, so try all and keep if valid?
    # But adding one might make another invalid.
    # Greedy: try to add each extra edge if it keeps the graph valid.
    # Order matters. Randomize?
    
    current_edges = edges
    added = []
    
    # Shuffle extra edges to avoid bias
    random.seed(42)
    random.shuffle(extra_edges)
    
    for e in extra_edges:
        if e in current_edges: continue
        new_edges = current_edges + [e]
        if is_valid(new_edges, n):
            current_edges = new_edges
            added.append(e)
            
    return current_edges
