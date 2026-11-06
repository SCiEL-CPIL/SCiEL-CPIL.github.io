"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 120.0
    n = 31
    target_edges = 91
    
    # Polarity graph of PG(2,5)
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            if (p1[0]*p2[0] + p1[1]*p2[1] - p1[2]*p2[2]) % q == 0:
                edges.add((i, j))
    
    edges = list(edges)
    
    if len(edges) < target_edges:
        all_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
        existing = set(edges)
        candidates = list(all_pairs - existing)
        edge_set = set(edges)
        
        for _ in range(target_edges - len(edges)):
            if time.time() - start_time > DEADLINE - 5:
                break
            best, best_cost = None, float('inf')
            for u, v in candidates:
                common = sum(1 for w in range(n) 
                            if w != u and w != v 
                            and (min(u,w),max(u,w)) in edge_set 
                            and (min(v,w),max(v,w)) in edge_set)
                if common < best_cost:
                    best_cost, best = common, (u, v)
            if best:
                edges.append(best)
                edge_set.add(best)
                candidates.remove(best)
    
    elif len(edges) > target_edges:
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        edges.sort(key=lambda e: deg[e[0]] * deg[e[1]], reverse=True)
        del edges[target_edges:]
    
    return [[u, v] for u, v in edges]
# EVOLVE-BLOCK-END
