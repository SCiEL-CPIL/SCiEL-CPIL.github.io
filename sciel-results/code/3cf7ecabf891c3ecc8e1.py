   def construct():
       def instance(n, k, p, seed):
           rng = Random(seed)
           edges = set()
           # Clique edges
           for u in range(k):
               for v in range(u + 1, k):
                   edges.add((u, v))
           # Cross edges
           for v in range(k, n):
               for u in range(k):
                   if rng.random() < p:
                       edges.add((u, v))
           return list(edges)

       def solve(n, k, p, seed):
           edges = instance(n, k, p, seed)
           
           # Build adjacency for I vertices
           # adj[v] = set of u in C connected to v
           adj = {v: set() for v in range(k, n)}
           
           for u, v in edges:
               if u < k and v >= k:
                   # u in C, v in I
                   adj[v].add(u)
               elif v < k and u >= k:
                   # v in C, u in I (shouldn't happen based on generation, but safe)
                   adj[u].add(v)
               # C-C edges ignored for adj[I]
           
           cliques = []
           # 1. The main clique C
           if k >= 2:
               cliques.append(list(range(k)))
           
           # 2. Cliques for each v in I with neighbors
           for v in range(k, n):
               if adj[v]:
                   # Sort for determinism, though not strictly required for validity
                   clique_v = list(adj[v]) + [v]
                   cliques.append(clique_v)
                   
           return cliques

       specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
       results = []
       for spec in specs:
           results.append(solve(*spec))
       return results
   