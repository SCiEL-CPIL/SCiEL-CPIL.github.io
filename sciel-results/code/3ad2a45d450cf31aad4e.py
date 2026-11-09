import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    n = 15
    # Partition: A = 0..6 (size 7), B = 7..14 (size 8)
    A = list(range(7))
    B = list(range(7, 15))
    
    # Start with complete bipartite K_{7,8}
    edges = set()
    for a in A:
        for b in B:
            edges.add((min(a,b), max(a,b)))
    
    def get_degrees():
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def is_valid():
        deg = get_degrees()
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check every 3-edge path s-u-v-t
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Try adding intra-partition edges greedily
    # Candidate edges: all pairs within A and within B not already present
    candidates = []
    for i, j in itertools.combinations(A, 2):
        candidates.append((i, j))
    for i, j in itertools.combinations(B, 2):
        candidates.append((i, j))
    
    # Shuffle candidates in a deterministic but varied order
    import random
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    best_edges = set(edges)
    best_count = len(edges)
    
    for i, j in candidates:
        if time.time() > DEADLINE:
            break
        e = (min(i, j), max(i, j))
        if e in edges:
            continue
        edges.add(e)
        if is_valid():
            best_edges = set(edges)
            best_count = len(edges)
        else:
            edges.discard(e)
    
    # Convert to list format
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
