import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        if time.time() > deadline:
            return
        if not P and not X:
            if len(R) > len(max_clique):
                max_clique.clear()
                max_clique.extend(R)
                max_clique.sort()
            return
        
        # Pivot selection
        if P:
            pivot_candidates = P.union(X)
        else:
            pivot_candidates = X
            
        if pivot_candidates:
            # Choose pivot with max degree in P
            pivot = max(pivot_candidates, key=lambda u: len(adj[u].intersection(P)))
            candidates = P.difference(adj[pivot])
        else:
            candidates = P.copy()
            
        for v in list(candidates):
            bron_kerbosch(R + [v], P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
            
    bron_kerbosch([], set(range(n)), set())
    
    if not max_clique:
        # Fallback
        if edges:
            u, v = edges[0]
            return {'branches': [u, v], 'paths': [[u, v]]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]} # Should not happen for n>=2
            
    branches = list(max_clique)
    paths = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            paths.append([branches[i], branches[j]])
            
    return {'branches': branches, 'paths': paths}
