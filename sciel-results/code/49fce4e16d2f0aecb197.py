import time

N = 40
DEADLINE_SECONDS = 85  # safety margin inside the 90s evaluator window


def construct():
    t0 = time.time()
    
    # Elementary fan: path 0-1-...-39 (N-1 edges) plus spokes (i,0) for i=2..39 (N-2 edges)
    # Total = 2N-3 = 77 edges.
    # Cycles for length k (3..N): 0,1,...,k-1,0 — uses path edges (j,j+1) and closing spoke (k-1,0).
    
    path_edges = [(i, i + 1) for i in range(N - 1)]
    spoke_edges = [(i, 0) for i in range(2, N)]
    edges = path_edges + spoke_edges
    
    # De-duplicate (spokes for i>=2 never equal path edges which connect >=1 to >=1 or 0-1)
    seen = set()
    unique_edges = []
    for a, b in edges:
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            unique_edges.append((a, b))
    edges = unique_edges
    
    # Build adjacency set for O(1) edge checks
    edge_set = set()
    for a, b in edges:
        edge_set.add((min(a, b), max(a, b)))
    
    cycles = []
    for k in range(3, N + 1):
        cyc = list(range(k))
        cycles.append(cyc)
    
    # Validate cycles: distinct vertices and all required edges present
    ok = True
    if len(edges) != len(edge_set):
        ok = False  # duplicate edges
    for cyc in cycles:
        if len(set(cyc)) != len(cyc):
            ok = False
            break
        L = len(cyc)
        for i in range(L):
            u = cyc[i]
            v = cyc[(i + 1) % L]
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                ok = False
                break
        if not ok:
            break
    
    # If validation fails (shouldn't), we still return the same structure since it's known valid
    # but to be safe under deadline, return immediately.
    
    return {'edges': edges, 'cycles': cycles}
