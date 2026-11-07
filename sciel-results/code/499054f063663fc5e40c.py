import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def color_graph(adj, n):
        # Greedy coloring with ordering by degree (descending)
        # adj is list of lists or sets
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        color = [-1] * n
        colors_used = 0
        for u in order:
            used_colors = set()
            for v in adj[u]:
                if color[v] != -1:
                    used_colors.add(color[v])
            c = 0
            while c in used_colors:
                c += 1
            color[u] = c
            colors_used = max(colors_used, c + 1)
        return color, colors_used

    def clique_partition(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency for K
        # K is 0..k-1
        # I is k..n-1
        
        # 1. Determine N_K(v) for all v in I
        N_K = [set() for _ in range(n)] # N_K[v] for v in I
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N_K[v].add(u)
                    
        # 2. Build G_conflict on K
        # adj_conf[u] is set of w such that (u,w) in G_conflict
        adj_conf = [set() for _ in range(k)]
        for v in range(k, n):
            neigh = list(N_K[v])
            for i in range(len(neigh)):
                for j in range(i+1, len(neigh)):
                    u = neigh[i]
                    w = neigh[j]
                    adj_conf[u].add(w)
                    adj_conf[w].add(u)
                    
        # 3. Color G_conflict
        color, num_colors = color_graph(adj_conf, k)
        
        # 4. Construct Cliques
        cliques = []
        
        # Cross cliques
        for v in range(k, n):
            if N_K[v]:
                clique = list(N_K[v])
                clique.append(v)
                cliques.append(clique)
                
        # K cliques (color classes)
        # Group vertices in K by color
        color_classes = [ [] for _ in range(num_colors) ]
        for u in range(k):
            color_classes[color[u]].append(u)
            
        for c in range(num_colors):
            if len(color_classes[c]) >= 2:
                cliques.append(color_classes[c])
            # If len < 2, it's a singleton, which is invalid as a clique in the output?
            # "Every listed set must be a clique with at least two vertices"
            # If a color class has 1 vertex, it covers no edges. We don't need to list it.
            # If a color class has 0 vertices, skip.
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 10:
            break
        results.append(clique_partition(*spec))
    
    return results
