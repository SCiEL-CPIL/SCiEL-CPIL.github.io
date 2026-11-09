"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(added_edges):
    """Build adjacency list from base edges + added edges. Returns None if duplicate."""
    n = P['n']
    base = set()
    for e in P['edges']:
        base.add((min(e[0], e[1]), max(e[0], e[1])))
    all_edges = set(base)
    for e in added_edges:
        u, v = int(e[0]), int(e[1])
        u, v = min(u, v), max(u, v)
        if (u, v) in all_edges:
            return None
        all_edges.add((u, v))
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check(adj):
    """Check triangle-free, connected, diameter <= 4. Returns (valid, diameter)."""
    n = P['n']
    if adj is None:
        return False, float('inf')
    # Triangle-free: for each vertex, no two neighbors are adjacent
    for u in range(n):
        nbrs = adj[u]
        for v in nbrs:
            for w in adj[v]:
                if w in nbrs and w != u:
                    return False, float('inf')
    # Connected + diameter via BFS from each vertex
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        qi = 0
        while qi < len(q):
            node = q[qi]
            qi += 1
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        if any(d == -1 for d in dist):
            return False, float('inf')
        max_dist = max(max_dist, max(dist))
    return True, max_dist

def _valid(added_edges):
    adj = _build_adj(added_edges)
    valid, _ = _check(adj)
    return valid

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s budget, 10s safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # All valid candidate edges: even-odd pairs not in base (bipartite => triangle-free)
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Strategy: Hub-and-spoke with local search refinement
    # Initial: hub at 12 (center) connected to all odd vertices
    hub = 12
    initial = []
    for v in range(n):
        if v != hub and (v % 2) != (hub % 2) and (min(hub, v), max(hub, v)) not in base_edges:
            initial.append([hub, v])
    
    # Verify initial solution
    best = [list(e) for e in initial]
    best_adj = _build_adj(best)
    best_valid, best_diam = _check(best_adj)
    
    if not best_valid:
        # Fallback: all odd-difference edges
        return [[int(u), int(v)] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
    
    best_len = len(best)
    
    # Local search: iteratively try removing one edge and adding another
    # Also try pure removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Try pure removal of each edge
        for i in range(best_len):
            if (time.time() - start_time) > deadline:
                break
            remaining = [e for j, e in enumerate(best) if j != i]
            if _valid(remaining):
                best = remaining
                best_len = len(best)
                improved = True
                break  # Restart loop with new best
        
        if improved:
            continue
        
        # Try swap: remove one, add another
        for i in range(best_len):
            if (time.time() - start_time) > deadline:
                break
            removed = best[i]
            rem_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best if e is not removed)
            remaining = [e for j, e in enumerate(best) if j != i]
            
            for cand in candidates:
                if (time.time() - start_time) > deadline:
                    break
                if cand in rem_set:
                    continue
                test = remaining + [[cand[0], cand[1]]]
                if _valid(test):
                    best = test
                    best_len = len(best)
                    improved = True
                    break
            if improved:
                break
        
        # Try adding one more edge (if we can later remove)
        if not improved and (time.time() - start_time) < deadline:
            for cand in candidates:
                if (time.time() - start_time) > deadline:
                    break
                cand_set = (min(cand[0], cand[1]), max(cand[0], cand[1]))
                if cand_set in set((min(e[0], e[1]), max(e[0], e[1])) for e in best):
                    continue
                test = best + [[cand[0], cand[1]]]
                if _valid(test):
                    # Check if we can now remove something
                    for j in range(len(test)):
                        remaining2 = [e for k, e in enumerate(test) if k != j]
                        if _valid(remaining2):
                            best = remaining2
                            best_len = len(best)
                            improved = True
                            break
                    if improved:
                        break
    
    # Final verification
    adj_final = _build_adj(best)
    valid, diam = _check(adj_final)
    
    if not valid:
        # Fallback to seed
        best = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
    
    return [[int(u), int(v)] for u, v in best]

# EVOLVE-BLOCK-END
