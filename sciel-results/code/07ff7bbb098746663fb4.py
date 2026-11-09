"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e), max(e)))

    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)

    # All candidate edges (non-path, non-self)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                all_candidates.append((u, v))

    def bfs_distances(adj, src):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if dist[nbr] == -1:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        return dist

    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = bfs_distances(adj, src)
            if any(d == -1 for d in dist):
                return 10**9
            md = max(dist)
            if md > max_dist:
                max_dist = md
        return max_dist

    def is_triangle_free_add(adj, u, v):
        return len(adj[u] & adj[v]) == 0

    def check_valid(added_edges):
        adj = [set(s) for s in adj_base]
        for u, v in added_edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            if not is_triangle_free_add(adj, u, v):
                return False
            adj[u].add(v)
            adj[v].add(u)
        # connected + diameter
        dist = bfs_distances(adj, 0)
        if any(d == -1 for d in dist):
            return False
        return compute_diameter(adj) <= 4

    def greedy_build():
        """Greedy: start with path, repeatedly add edge that most reduces max distance."""
        adj = [set(s) for s in adj_base]
        added = []
        used_edges = set(base_edges)
        
        while compute_diameter(adj) > 4:
            best_edge = None
            best_score = -1
            for (u, v) in all_candidates:
                if (u, v) in used_edges:
                    continue
                if not is_triangle_free_add(adj, u, v):
                    continue
                # Try adding this edge, measure improvement
                adj[u].add(v)
                adj[v].add(u)
                diam = compute_diameter(adj)
                # Count pairs now within distance 4
                adj[u].remove(v)
                adj[v].remove(u)
                
                # Score: lower diameter is better, then more pairs within 4
                score = -diam * 1000
                if diam <= 4:
                    score = 100000
                best_score_candidate = score
                if best_score_candidate > best_score:
                    best_score = best_score_candidate
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            used_edges.add((u, v))
            added.append([u, v])
        
        return added

    def shake_shake_remove(current, max_iter=200):
        """Randomized removal: try removing random edges, keep if still valid."""
        current = list(current)
        rng = random.Random(42)
        improved = True
        while improved and len(current) > 0:
            improved = False
            for _ in range(max_iter):
                if len(current) <= 1:
                    break
                idx = rng.randint(0, len(current) - 1)
                trial = current[:idx] + current[idx+1:]
                if check_valid(trial):
                    current = trial
                    improved = True
                    break
        return current

    def systematic_remove(current):
        """Systematic: try removing each edge in order."""
        current = list(current)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                trial = current[:i] + current[i+1:]
                if check_valid(trial):
                    current = trial
                    improved = True
                    break
        return current

    # Strategy 1: Greedy build + shake
    best_edges = None
    best_count = 10**9

    g = greedy_build()
    if check_valid(g):
        g_shaken = shake_shake_remove(g, max_iter=500)
        g_sys = systematic_remove(g_shaken)
        if len(g_sys) < best_count:
            best_count = len(g_sys)
            best_edges = g_sys

    # Strategy 2: Multiple greedy variants with different candidate orderings
    rng2 = random.Random(123)
    for attempt in range(10):
        if time.time() > deadline - 15:
            break
        shuffled = list(all_candidates)
        rng2.shuffle(shuffled)
        
        adj = [set(s) for s in adj_base]
        added = []
        used_edges = set(base_edges)
        
        while compute_diameter(adj) > 4:
            best_edge = None
            best_diam = 10**9
            for (u, v) in shuffled:
                if (u, v) in used_edges:
                    continue
                if not is_triangle_free_add(adj, u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                diam = compute_diameter(adj)
                adj[u].remove(v)
                adj[v].remove(u)
                if diam < best_diam:
                    best_diam = diam
                    best_edge = (u, v)
                    if diam <= 4:
                        break
            
            if best_edge is None:
                break
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            used_edges.add((u, v))
            added.append([u, v])
        
        if check_valid(added):
            shaken = shake_shake_remove(added, max_iter=300)
            sys_removed = systematic_remove(shaken)
            if len(sys_removed) < best_count:
                best_count = len(sys_removed)
                best_edges = sys_removed

    # Strategy 3: Direct construction with known good patterns
    # For path of 24 nodes, diameter 4: we can use edges that jump by ~5-6
    # Pattern: add edges (0,5),(5,10),(10,15),(15,20),(20,23) plus fillers
    # Let's try: connect every 5th node
    patterns = [
        # Jump-5 pattern
        [[0,5],[5,10],[10,15],[15,20],[20,23],[1,6],[6,11],[11,16],[16,21],[2,7],[7,12],[12,17],[17,22],[3,8],[8,13],[13,18],[18,23],[4,9],[9,14],[14,19],[19,23]],
        # Jump-6 pattern  
        [[0,6],[6,12],[12,18],[18,23],[1,7],[7,13],[13,19],[19,23],[2,8],[8,14],[14,20],[20,23],[3,9],[9,15],[15,21],[21,23],[4,10],[10,16],[16,22],[22,23],[5,11],[11,17],[17,23]],
        # Hub pattern: connect to center
        [[0,12],[1,12],[2,12],[3,12],[4,12],[5,12],[6,12],[7,12],[8,12],[9,12],[10,12],[11,12],[13,12],[14,12],[15,12],[16,12],[17,12],[18,12],[19,12],[20,12],[21,12],[22,12],[23,12]],
    ]
    
    for pat in patterns:
        if time.time() > deadline - 10:
            break
        # Filter to only valid (triangle-free) additions
        adj = [set(s) for s in adj_base]
        valid_pat = []
        for e in pat:
            u, v = e
            if (u, v) in base_edges:
                continue
            if not is_triangle_free_add(adj, u, v):
                continue
            adj[u].add(v)
            adj[v].add(u)
            valid_pat.append([u, v])
        
        if check_valid(valid_pat):
            shaken = shake_shake_remove(valid_pat, max_iter=400)
            sys_removed = systematic_remove(shaken)
            if len(sys_removed) < best_count:
                best_count = len(sys_removed)
                best_edges = sys_removed

    # Strategy 4: Randomized local search from scratch
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            rng3 = random.Random(1000 + restart)
            # Start with a random valid superset
            adj = [set(s) for s in adj_base]
            added = []
            used_edges = set(base_edges)
            
            # Add random bipartite edges until diameter <= 4
            candidates = [e for e in all_candidates if (e[0] % 2) != (e[1] % 2)]
            rng3.shuffle(candidates)
            
            for (u, v) in candidates:
                if compute_diameter(adj) <= 4:
                    break
                if (u, v) in used_edges:
                    continue
                if not is_triangle_free_add(adj, u, v):
                    continue
                adj[u].add(v)
                adj[v].add(u)
                used_edges.add((u, v))
                added.append([u, v])
            
            if check_valid(added):
                # Aggressive shake
                for _ in range(3):
                    added = shake_shake_remove(added, max_iter=500)
                added = systematic_remove(added)
                if len(added) < best_count:
                    best_count = len(added)
                    best_edges = added

    # Final verification
    if best_edges is None:
        # Absolute fallback
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u % 2) != (v % 2) and (u, v) not in base_edges:
                    best_edges.append([u, v])
        best_edges = systematic_remove(best_edges)

    if not check_valid(best_edges):
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u % 2) != (v % 2) and (u, v) not in base_edges:
                    best_edges.append([u, v])

    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
