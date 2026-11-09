"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w in adj[u] and w != v:
                            return False
        return True
    
    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            if -1 in dist:
                return float('inf')
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    def validate(added_edges):
        adj = build_adj(added_edges)
        if not is_triangle_free(adj):
            return None
        diam = compute_diameter(adj)
        if diam > 4:
            return None
        added_set = set()
        for u, v in added_edges:
            key = tuple(sorted((u, v)))
            if key in base_edges:
                return None
            if key in added_set:
                return None
            added_set.add(key)
        return len(added_edges)
    
    # All valid candidate edges: even-odd pairs with |u-v| >= 2
    all_candidates = []
    for u in range(0, n, 2):
        for v in range(1, n, 2):
            if abs(u - v) >= 2:
                key = tuple(sorted((u, v)))
                if key not in base_edges:
                    all_candidates.append(key)
    
    # Start with all candidates (guaranteed valid: bipartite => triangle-free)
    current = set(all_candidates)
    current_count = len(current)
    
    # Greedy removal: try removing each edge, keep if still valid
    # Process in order of least "importance" (edges farthest from center first)
    sorted_candidates = sorted(all_candidates, key=lambda e: -abs(e[0] - e[1]))
    
    for edge in sorted_candidates:
        if time.time() > deadline - 10:
            break
        if edge not in current:
            continue
        trial = current - {edge}
        trial_list = [list(e) for e in trial]
        count = validate(trial_list)
        if count is not None:
            current = trial
            current_count = count
    
    # Second pass: try again (removals may enable further removals)
    for edge in sorted_candidates:
        if time.time() > deadline - 5:
            break
        if edge not in current:
            continue
        trial = current - {edge}
        trial_list = [list(e) for e in trial]
        count = validate(trial_list)
        if count is not None:
            current = trial
            current_count = count
    
    # Try structured patterns that might be better
    patterns = [
        lambda u, v: abs(u - v) == 2,
        lambda u, v: abs(u - v) in (2, 4),
        lambda u, v: abs(u - v) in (2, 6),
        lambda u, v: abs(u - v) in (2, 4, 6),
        lambda u, v: abs(u - v) in (4, 6),
        lambda u, v: abs(u - v) in (2, 3),
        lambda u, v: abs(u - v) in (3, 4),
        lambda u, v: abs(u - v) in (2, 4, 8),
        lambda u, v: abs(u - v) in (2, 6, 10),
    ]
    
    for pred in patterns:
        if time.time() > deadline - 3:
            break
        trial = set()
        for u in range(0, n, 2):
            for v in range(1, n, 2):
                if abs(u - v) >= 2 and pred(u, v):
                    key = tuple(sorted((u, v)))
                    if key not in base_edges:
                        trial.add(key)
        if not trial:
            continue
        trial_list = [list(e) for e in trial]
        count = validate(trial_list)
        if count is not None and count < current_count:
            current = trial
            current_count = count
    
    return [[int(u), int(v)] for u, v in current]

# EVOLVE-BLOCK-END
