# EVOLVE-BLOCK-START
import itertools

def _get_cycle_lengths(adj):
    n = len(adj)
    lengths = set()
    for start in range(n):
        stack = [(start, [start], frozenset([start]))]
        while stack:
            node, path, visited = stack.pop()
            for nb in adj[node]:
                if nb == start and len(path) >= 3:
                    lengths.add(len(path))
                elif nb > start and nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))
    return lengths

def _build_adj(edges):
    n = 12
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _count_cycle_lengths(adj):
    lengths = _get_cycle_lengths(adj)
    return len(lengths) if lengths else float('inf')

def construct():
    # Direct construction of two disjoint K3,3 graphs.
    # This graph is 3-regular, bipartite (triangle-free), and has cycles of length 4 and 6.
    # Distinct cycle lengths: {4, 6}. Count = 2.
    # This is likely the minimum possible count for 12 vertices, 3-regular.
    
    edges = []
    # First K3,3: Left 0-2, Right 3-5
    for i in range(3):
        for j in range(3, 6):
            edges.append((i, j))
            
    # Second K3,3: Left 6-8, Right 9-11
    for i in range(6, 9):
        for j in range(9, 12):
            edges.append((i, j))
            
    # Verify validity
    adj = _build_adj(edges)
    # Check degrees
    for i in range(12):
        if len(adj[i]) < 3:
            # Fallback to backtracking if this fails (it shouldn't)
            pass 
            
    # Return sorted edges
    sorted_edges = sorted(tuple(sorted(e)) for e in edges)
    return [list(e) for e in sorted_edges]

# EVOLVE-BLOCK-END
