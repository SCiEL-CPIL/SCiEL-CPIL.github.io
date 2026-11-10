"""Clique partition of split graphs via structural decomposition."""
import time
import random


def _build_graph(n, k, p, seed):
    rng = random.Random(seed)
    K = list(range(k))
    I = list(range(k, n))
    adj = [set() for _ in range(n)]
    for i in range(k):
        for j in range(i + 1, k):
            adj[i].add(j)
            adj[j].add(i)
    for u in I:
        for v in K:
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return K, I, adj


def _cover_k_edges(K, adj, rng, max_iters=5000):
    """Cover all edges within K using cliques (size>=2), minimize count."""
    k = len(K)
    if k < 2:
        return []
    # Build adjacency within K
    k_adj = [set() for _ in range(k)]
    for i in range(k):
        for j in K:
            if j > i and j in adj[K[i]]:
                k_adj[i].add(K.index(j))
                k_adj[K.index(j)].add(i)
    # Simpler: work with indices
    k_adj = [set() for _ in range(k)]
    for i in range(k):
        for j in range(i+1, k):
            if K[j] in adj[K[i]]:
                k_adj[i].add(j)
                k_adj[j].add(i)
    
    all_edges = set()
    for i in range(k):
        for j in k_adj[i]:
            if i < j:
                all_edges.add((i, j))
    
    best = None
    best_count = len(all_edges)  # worst case: all 2-cliques
    
    for it in range(max_iters):
        edge_list = list(all_edges)
        rng.shuffle(edge_list)
        uncovered = set(edge_list)
        cliques = []
        for (i, j) in edge_list:
            if (i, j) not in uncovered:
                continue
            # Build maximal clique containing i,j using only uncovered edges
            cand = {i, j}
            for w in range(k):
                if w in cand:
                    continue
                ok = True
                for x in cand:
                    if (min(x,w), max(x,w)) not in uncovered:
                        ok = False
                        break
                if ok:
                    cand.add(w)
            cliques.append(list(cand))
            for a in range(len(cand)):
                for b in range(a+1, len(cand)):
                    uncovered.discard((min(cand[a], cand[b]), max(cand[a], cand[b])))
        if len(cliques) < best_count:
            best_count = len(cliques)
            best = [list(c) for c in cliques]
            if best_count == 1:
                break
    
    return best if best else [[i, j] for (i, j) in all_edges]


def construct():
    deadline = time.time() + 110
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed * 1000 + 7)
        K, I, adj = _build_graph(n, k, p, seed)
        cliques = []
        
        # Part 1: cover edges within K
        k_cliques = _cover_k_edges(K, adj, rng)
        for c in k_cliques:
            cliques.append([K[i] for i in c])
        
        # Part 2: one star per K-vertex with I-neighbors
        for v in K:
            nbrs = [u for u in I if u in adj[v]]
            if len(nbrs) >= 1:
                cliques.append([v] + nbrs)
        
        # Validate
        validated = []
        for c in cliques:
            c = [int(x) for x in c]
            if len(c) >= 2 and len(set(c)) == len(c):
                validated.append(c)
        results.append(validated)
        
        if time.time() > deadline:
            break
    return results
