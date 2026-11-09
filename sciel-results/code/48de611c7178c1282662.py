"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 15}

def _check_validity(adj, n):
    """Check if no simple path s-u-v-t has deg(s)==deg(t)."""
    deg = [len(adj[i]) for i in range(n)]
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
                        return False
    return True

def _count_violations(adj, n):
    """Count number of violating paths."""
    deg = [len(adj[i]) for i in range(n)]
    count = 0
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
                        count += 1
    return count

def _try_build_random(n, seed=42):
    """Try to build a dense valid graph via randomized local search."""
    rng = random.Random(seed)
    # Start with complete graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Remove edges to fix violations - use a smarter strategy
    # Prioritize removing edges that appear in many violations
    max_iter = 3000
    it = 0
    while it < max_iter:
        it += 1
        deg = [len(adj[i]) for i in range(n)]
        # Count how many violations each edge participates in
        edge_violations = {}
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
                            key = (u, v)
                            edge_violations[key] = edge_violations.get(key, 0) + 1
        
        if not edge_violations:
            break
        
        # Remove the edge with most violations
        worst_edge = max(edge_violations, key=edge_violations.get)
        u, v = worst_edge
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Now try to add back edges - use a faster incremental check
    improved = True
    while improved:
        improved = False
        deg = [len(adj[i]) for i in range(n)]
        # Try adding edges in random order to escape local optima
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
        rng.shuffle(missing)
        for i, j in missing:
            adj[i].add(j)
            adj[j].add(i)
            if _check_validity(adj, n):
                improved = True
                deg = [len(adj[k]) for k in range(n)]
            else:
                adj[i].discard(j)
                adj[j].discard(i)
    
    return [[i, j] for i in range(n) for j in adj[i] if j > i]

def _try_build_structured(n):
    """Try a structured construction with diverse degrees."""
    # Create a structure where degrees are as distinct as possible
    # to minimize equal-degree endpoint conflicts
    adj = [set() for _ in range(n)]
    
    # Assign target degrees: 0, 1, 2, ..., n-1 (or close to it)
    # Use a threshold-based construction
    # Vertex i has degree approximately i
    for i in range(n):
        for j in range(i+1, n):
            # Connect if j's index is within i's "range"
            # This creates a triangular-like structure with varying degrees
            if j <= i + 1 + (i * i) // n:
                adj[i].add(j)
                adj[j].add(i)
    
    # Check validity and try to improve
    if _check_validity(adj, n):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        adj[i].add(j)
                        adj[j].add(i)
                        if _check_validity(adj, n):
                            improved = True
                        else:
                            adj[i].discard(j)
                            adj[j].discard(i)
        return [[i, j] for i in range(n) for j in adj[i] if j > i]
    
    # Try another structured pattern: bipartite-like with unequal sides
    adj2 = [set() for _ in range(n)]
    mid = n // 2
    for i in range(mid):
        for j in range(mid, n):
            adj2[i].add(j)
            adj2[j].add(i)
    
    if _check_validity(adj2, n):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj2[i]:
                        adj2[i].add(j)
                        adj2[j].add(i)
                        if _check_validity(adj2, n):
                            improved = True
                        else:
                            adj2[i].discard(j)
                            adj2[j].discard(i)
        return [[i, j] for i in range(n) for j in adj2[i] if j > i]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s kill limit)
    
    best_edges = None
    best_count = 0
    
    # Try structured construction first
    edges = _try_build_structured(P['n'])
    if edges is not None:
        best_edges = edges
        best_count = len(edges)
    
    # Try multiple random restarts with more seeds
    for seed in range(80):
        if time.time() > deadline:
            break
        edges = _try_build_random(P['n'], seed=seed)
        if edges and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Fallback to star if nothing works
    if best_edges is None:
        best_edges = [[0, v] for v in range(1, P['n'])]
    
    # Convert to proper format
    result = []
    for e in best_edges:
        u = int(e[0])
        v = int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
