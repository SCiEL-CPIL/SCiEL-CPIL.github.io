import time

P = {'n': 15}

def construct():
    n = P['n']
    deadline = time.time() + 50  # safety margin
    
    # Start with empty graph
    adj = [[False]*n for _ in range(n)]
    edges = []
    
    def get_degree(v):
        return sum(adj[v])
    
    def has_conflict(s, t):
        """Check if s and t have a 3-edge path s-u-v-t with all distinct"""
        if s == t:
            return False
        ns = [u for u in range(n) if adj[s][u]]
        nt = [v for v in range(n) if adj[t][v]]
        for u in ns:
            if u == t:
                continue
            for v in nt:
                if v == s or v == u:
                    continue
                if adj[u][v]:
                    return True
        return False
    
    def is_valid():
        """Check all pairs for the constraint"""
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                if get_degree(s) == get_degree(t):
                    if has_conflict(s, t):
                        return False
        return True
    
    def try_add_edge(u, v):
        if adj[u][v] or u == v:
            return False
        adj[u][v] = True
        adj[v][u] = True
        edges.append([u, v])
        if not is_valid():
            adj[u][v] = False
            adj[v][u] = False
            edges.pop()
            return False
        return True
    
    # Try all possible edges in a smart order
    # Prioritize edges that create degree diversity
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Shuffle for variety, but try systematically
    # Use greedy: try to add edges, prefer those connecting low-degree to high-degree
    added = set()
    for iteration in range(10):
        if time.time() > deadline:
            break
        progress = True
        while progress:
            if time.time() > deadline:
                return [[int(u), int(v)] for u, v in edges]
            progress = False
            # Sort candidate edges by some heuristic
            candidates = [(u, v) for u, v in all_pairs if not adj[u][v]]
            # Heuristic: prefer edges that maximize degree difference
            candidates.sort(key=lambda uv: -(abs(get_degree(uv[0]) - get_degree(uv[1]))))
            for u, v in candidates:
                if adj[u][v]:
                    continue
                if try_add_edge(u, v):
                    progress = True
                    break
    
    return [[int(u), int(v)] for u, v in edges]
