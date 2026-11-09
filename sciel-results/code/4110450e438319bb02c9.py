"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 15}

def _check_validity_fast(adj, n):
    """Check if no simple path s-u-v-t has deg(s)==deg(t). Fast version."""
    deg = [0]*n
    for i in range(n):
        deg[i] = len(adj[i])
    # Group vertices by degree
    deg_map = {}
    for i in range(n):
        deg_map.setdefault(deg[i], set()).add(i)
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # For each s in adj[u] (s != v), check if any t in adj[v] (t != u, t != s) has same degree as s
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                # Check if any t in adj[v] has degree ds, t != u, t != s
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[t] == ds:
                        return False
    return True

def _fast_add_check(adj, n, i, j):
    """Check if adding edge (i,j) would create a violation. O(deg(i)*deg(j) + deg(i) + deg(j))"""
    deg = [len(adj[k]) for k in range(n)]
    deg[i] += 1
    deg[j] += 1
    
    # A violation is a path s-u-v-t where deg(s)==deg(t)
    # Adding edge (i,j) creates new paths where (i,j) is one of the edges
    # Case 1: path s-i-j-t: s in adj[i]\{j}, t in adj[j]\{i}, s!=t, deg[s]==deg[t]
    for s in adj[i]:
        if s == j:
            continue
        ds = deg[s]
        for t in adj[j]:
            if t == i or t == s:
                continue
            if deg[t] == ds:
                return False
    # Case 2: path s-j-i-t: s in adj[j]\{i}, t in adj[i]\{j}, s!=t, deg[s]==deg[t]
    for s in adj[j]:
        if s == i:
            continue
        ds = deg[s]
        for t in adj[i]:
            if t == j or t == s:
                continue
            if deg[t] == ds:
                return False
    return True

def _try_build_structured(n):
    """Build a threshold graph with distinct degrees, then greedily add edges."""
    # Threshold graph: vertex i connects to all vertices j > i where j <= threshold[i]
    # This gives distinct degrees if thresholds are chosen properly
    # Simplest: vertex i connects to vertices i+1, i+2, ..., n-1 -> degrees n-1-i
    # That's the complete graph minus a triangle-free complement... actually it's a "nesting" graph
    # deg(i) = n-1-i, so degrees are 14,13,12,...,0 - all distinct!
    
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # This is the complete graph! Degrees are all n-1=14. That's bad.
    # Let me try: vertex i connects to vertices j where j > i and (j-i) <= i+1
    # Actually, let's use: connect i to j if j > i and j <= i + (n-1-i)
    # That's still complete.
    
    # Better: use a "split" construction.
    # Put vertices in order. Vertex i connects to vertices j > i only if some condition.
    # To get distinct degrees d_i = n-1-i, vertex i must have exactly n-1-i neighbors.
    # Since there are n-1-i vertices with index > i, vertex i must connect to ALL of them.
    # But then vertex i also has neighbors with index < i.
    # For vertex 0: degree 14, must connect to all 14 others.
    # For vertex 1: degree 13, must NOT connect to exactly 1 of {0,2,3,...,14}.
    #   It already connects to 0 (from vertex 0's requirement). So it must not connect to one of {2,...,14}.
    # For vertex 2: degree 12, must NOT connect to 2 of {0,1,3,...,14}.
    #   It connects to 0 (required) and 1 (if vertex 1 chose to connect). 
    # This is getting complex. Let me just use the "co-chain" / threshold graph.
    
    # Threshold graph construction: 
    # Start with isolated vertices. Add vertices one by one, each new vertex is either
    # connected to all previous (dominating) or to none (isolated).
    # This gives distinct degrees.
    
    # Let's try: add vertex 0 (deg 0), then add vertices 1..14 all as dominating.
    # Vertex 1 connects to 0: deg(1)=1, deg(0)=1. Not distinct.
    
    # Alternative: just try the "almost complete" graph and remove edges to make degrees distinct.
    # Start with complete graph (all deg 14). Remove edges to make degrees 14,13,12,...,0.
    # Vertex 0: keep all 14 edges. deg=14.
    # Vertex 1: remove 1 edge. Remove (1,2). deg=13.
    # Vertex 2: remove 2 edges. Already lost (1,2). Remove one more, say (2,3). deg=12.
    # Vertex 3: remove 3 edges. Already lost (1,2) doesn't involve 3. Lost (2,3). Remove 2 more.
    # This is a "graphical sequence" problem. Let me just use a known construction.
    
    # Actually, the simplest valid dense graph: use degrees that are all distinct.
    # The maximum number of edges with all-distinct degrees on n vertices is bounded.
    # Sum of degrees = 2*E. If degrees are 0,1,...,14, sum=105, E=52.5 -> not integer.
    # Degrees 1,2,...,14 plus one more: sum = 105 + d. Must be even.
    # Let's try degrees: 0,1,2,...,13,14 -> sum=105 (odd, impossible)
    # Degrees: 1,2,...,14,14 -> sum=106, E=53. But two vertices have degree 14.
    # If two vertices have the same degree, we need to check the constraint.
    
    # Let me just use a greedy approach: start with complete graph, remove edges to fix violations.
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Greedy removal: while there's a violation, remove the "worst" edge
    for iteration in range(200):
        deg = [len(adj[i]) for i in range(n)]
        # Find a violating edge
        found = False
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
                            # Remove edge (u,v)
                            adj[u].discard(v)
                            adj[v].discard(u)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    # Now greedily add back edges
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    if _check_validity_fast(adj, n):
                        improved = True
                    else:
                        adj[i].discard(j)
                        adj[j].discard(i)
    
    return [[i, j] for i in range(n) for j in adj[i] if j > i]

def _try_build_random(n, seed=42):
    """Randomized local search: start from complete, remove edges, then add back."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
    
    # Phase 1: Remove edges to eliminate violations
    for iteration in range(500):
        deg = [len(adj[i]) for i in range(n)]
        # Count violations per edge
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
        # Remove edge with most violations (with some randomness)
        items = list(edge_viol.items())
        items.sort(key=lambda x: -x[1])
        # Pick from top 3 with probability
        top_k = min(3, len(items))
        chosen = items[rng.randint(0, top_k-1)][0]
        u, v = chosen
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Phase 2: Greedily add back edges
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
    deadline = start_time + 85
    
    best_edges = None
    best_count = 0
    
    # Try structured construction first
    edges = _try_build_structured(P['n'])
    if edges is not None:
        best_edges = edges
        best_count = len(edges)
    
    # Try multiple random restarts
    for seed in range(50):
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
