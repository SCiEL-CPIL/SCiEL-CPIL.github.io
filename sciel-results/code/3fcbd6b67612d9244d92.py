"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def _check_validity_fast(adj, n):
    """Check if no simple path s-u-v-t has deg(s)==deg(t)."""
    deg = [0]*n
    for i in range(n):
        deg[i] = len(adj[i])
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def _build_k78(n):
    """Build K_{7,8}: vertices 0-6 in part A, 7-14 in part B. All cross edges."""
    adj = [set() for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
    return adj

def _try_from_k78(n, seed=42):
    """Start from K_{7,8} and greedily add intra-part edges."""
    rng = random.Random(seed)
    adj = _build_k78(n)
    
    # Generate all possible intra-part edges
    intra_edges = []
    for i in range(7):
        for j in range(i+1, 7):
            intra_edges.append((i, j))
    for i in range(7, 15):
        for j in range(i+1, 15):
            intra_edges.append((i, j))
    
    rng.shuffle(intra_edges)
    
    improved = True
    while improved:
        improved = False
        for i, j in intra_edges:
            if j not in adj[i]:
                adj[i].add(j)
                adj[j].add(i)
                if _check_validity_fast(adj, n):
                    improved = True
                else:
                    adj[i].discard(j)
                    adj[j].discard(i)
    
    return [[i, j] for i in range(n) for j in adj[i] if j > i]

def _try_random_removal(n, seed=42):
    """Start from complete graph, remove edges to fix violations, then add back."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Remove edges to eliminate violations
    for iteration in range(300):
        deg = [len(adj[i]) for i in range(n)]
        edge_viol = {}
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            key = (min(u,v), max(u,v))
                            edge_viol[key] = edge_viol.get(key, 0) + 1
        if not edge_viol:
            break
        items = list(edge_viol.items())
        items.sort(key=lambda x: -x[1])
        top_k = min(3, len(items))
        chosen = items[rng.randint(0, top_k-1)][0]
        u, v = chosen
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Greedily add back edges
    improved = True
    while improved:
        improved = False
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        rng.shuffle(missing)
        for i, j in missing:
            adj[i].add(j)
            adj[j].add(i)
            if _check_validity_fast(adj, n):
                improved = True
            else:
                adj[i].discard(j)
                adj[j].discard(i)
    
    return [[i, j] for i in range(n) for j in adj[i] if j > i]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_edges = None
    best_count = 0
    
    # Strategy 1: Start from K_{7,8} (provably valid, 56 edges) and add intra-part edges
    for seed in range(20):
        if time.time() > deadline:
            break
        edges = _try_from_k78(P['n'], seed=seed)
        if edges and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Strategy 2: Random removal from K_15
    for seed in range(30):
        if time.time() > deadline:
            break
        edges = _try_random_removal(P['n'], seed=seed)
        if edges and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Fallback to K_{7,8} if nothing works
    if best_edges is None:
        adj = _build_k78(P['n'])
        best_edges = [[i, j] for i in range(P['n']) for j in adj[i] if j > i]
    
    # Convert to proper format
    result = []
    for e in best_edges:
        u = int(e[0])
        v = int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
