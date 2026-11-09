"""Triangle-free diameter-four augmentation via optimized search."""
# EVOLVE-BLOCK-START

import time
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _validate_fast(added_edges):
    """Fast validation: triangle-free (bipartite), connected, diameter <= 4."""
    n = 24
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    added_set = set()
    for u, v in added_edges:
        if not (0 <= u <= 23 and 0 <= v <= 23) or u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in added_set:
            return False
        added_set.add(key)
        if (u % 2) == (v % 2):
            return False
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # BFS from each vertex
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d == -1:
                return False
            if d > max_d:
                max_d = d
                if max_d > 4:
                    return False
    return True


def _all_pairs_dist(added_edges):
    """Compute all-pairs distances. Returns (max_dist, dist_matrix) or (None, None) if disconnected."""
    n = 24
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    dist_matrix = [[-1] * n for _ in range(n)]
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        dist_matrix[start] = dist
        for d in dist:
            if d == -1:
                return None, None
            if d > max_d:
                max_d = d
    return max_d, dist_matrix


def construct():
    """
    Find minimum edges for triangle-free diameter-4 augmentation of P24.
    """
    deadline = time.time() + 80  # 10s safety margin
    n = 24
    
    best_edges = None
    best_count = 99
    
    # Known good solution
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    if _validate_fast(fallback):
        best_edges = [list(e) for e in fallback]
        best_count = 10
    
    # Strategy: Try various hub configurations and greedy construction
    # Then local search to minimize
    
    # Phase 1: Try all pairs of hubs (even, odd) with greedy edge selection
    even_hubs = list(range(0, 24, 2))
    odd_hubs = list(range(1, 24, 2))
    
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    for he in even_hubs:
        if time.time() > deadline:
            break
        for ho in odd_hubs:
            if time.time() > deadline:
                break
            
            # Start with hub-hub edge if not adjacent
            edges = []
            if abs(he - ho) > 1:
                edges.append([he, ho])
            
            # Build adjacency
            adj = [set() for _ in range(n)]
            for u, v in P['edges']:
                adj[u].add(v)
                adj[v].add(u)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Greedily add edges to reduce diameter
            # Candidate edges: he to odd, ho to even
            used_edges = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
            
            while time.time() < deadline:
                # Check current diameter
                max_d, _ = _all_pairs_dist(edges)
                if max_d is None:
                    break
                if max_d <= 4:
                    break
                
                # Find the pair with maximum distance
                # Compute all pairs dist
                adj = [set() for _ in range(n)]
                for u, v in P['edges']:
                    adj[u].add(v)
                    adj[v].add(u)
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Find worst pair
                worst_pair = None
                worst_dist = 0
                for start in range(n):
                    dist = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                queue.append(v)
                    for v in range(start + 1, n):
                        if dist[v] > worst_dist:
                            worst_dist = dist[v]
                            worst_pair = (start, v)
                
                if worst_pair is None:
                    break
                
                # Try all candidate edges and pick the one that most reduces the worst distance
                candidates = []
                for o in range(1, 24, 2):
                    if abs(he - o) > 1:
                        key = (min(he, o), max(he, o))
                        if key not in used_edges and key not in base_edge_set:
                            candidates.append([he, o])
                for e in range(0, 24, 2):
                    if abs(ho - e) > 1:
                        key = (min(ho, e), max(ho, e))
                        if key not in used_edges and key not in base_edge_set:
                            candidates.append([ho, e])
                
                if not candidates:
                    break
                
                best_cand = None
                best_score = -1
                
                for cand in candidates:
                    u, v = cand
                    # Add temporarily
                    adj[u].add(v)
                    adj[v].add(u)
                    
                    # Compute new worst distance (only for pairs involving the worst pair's region)
                    # Actually, just compute the max distance
                    new_max = 0
                    for start in range(n):
                        dist = [-1] * n
                        dist[start] = 0
                        queue = [start]
                        head = 0
                        while head < len(queue):
                            node = queue[head]
                            head += 1
                            for nb in adj[node]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[node] + 1
                                    queue.append(nb)
                        for d in dist:
                            if d > new_max:
                                new_max = d
                    
                    # Remove
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
                    if new_max < best_score or best_score == -1:
                        best_score = new_max
                        best_cand = cand
                
                if best_cand is None:
                    break
                
                edges.append(best_cand)
                used_edges.add((min(best_cand[0], best_cand[1]), max(best_cand[0], best_cand[1])))
                adj[best_cand[0]].add(best_cand[1])
                adj[best_cand[1]].add(best_cand[0])
                
                if len(edges) >= best_count:
                    break
            
            # Check result
            max_d, _ = _all_pairs_dist(edges)
            if max_d is not None and max_d <= 4 and len(edges) < best_count:
                if _validate_fast(edges):
                    best_edges = [list(e) for e in edges]
                    best_count = len(edges)
    
    # Phase 2: Local search - try removing edges
    if best_edges is not None:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_edges)):
                if time.time() > deadline:
                    break
                trial = best_edges[:i] + best_edges[i+1:]
                if _validate_fast(trial):
                    best_edges = trial
                    best_count = len(trial)
                    improved = True
                    break
        
        # Try 1-swap
        if time.time() < deadline:
            current_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best_edges)
            all_candidates = []
            for e in range(0, 24, 2):
                for o in range(1, 24, 2):
                    key = (min(e, o), max(e, o))
                    if key not in base_edge_set and key not in current_set:
                        all_candidates.append([e, o])
            
            for i in range(len(best_edges)):
                if time.time() > deadline:
                    break
                trial_base = best_edges[:i] + best_edges[i+1:]
                trial_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in trial_base)
                
                for cand in all_candidates:
                    if time.time() > deadline:
                        break
                    key = (min(cand[0], cand[1]), max(cand[0], cand[1]))
                    if key in trial_set:
                        continue
                    trial = trial_base + [cand]
                    if _validate_fast(trial):
                        best_edges = trial
                        best_count = len(best_edges)
                        break
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline:
        random.seed(42)
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initial configuration
            num_edges = random.randint(4, min(8, best_count))
            all_possible = []
            for e in range(0, 24, 2):
                for o in range(1, 24, 2):
                    key = (min(e, o), max(e, o))
                    if key not in base_edge_set:
                        all_possible.append([e, o])
            
            if len(all_possible) < num_edges:
                continue
            
            edges = random.sample(all_possible, num_edges)
            
            # Greedy: add edges to fix violations
            used = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
            adj = [set() for _ in range(n)]
            for u, v in P['edges']:
                adj[u].add(v)
                adj[v].add(u)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            for _ in range(20):  # Max 20 additions
                if time.time() > deadline:
                    break
                max_d, _ = _all_pairs_dist(edges)
                if max_d is not None and max_d <= 4:
                    break
                if len(edges) >= best_count:
                    break
                
                # Find worst pair and try to fix
                worst_pair = None
                worst_dist = 0
                for start in range(n):
                    dist = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                queue.append(v)
                    for v in range(start + 1, n):
                        if dist[v] > worst_dist:
                            worst_dist = dist[v]
                            worst_pair = (start, v)
                
                if worst_pair is None:
                    break
                
                # Try random candidates
                available = [c for c in all_possible if (min(c[0],c[1]), max(c[0],c[1])) not in used]
                if not available:
                    break
                
                best_cand = None
                best_score = worst_dist + 1
                for cand in random.sample(available, min(20, len(available))):
                    u, v = cand
                    adj[u].add(v)
                    adj[v].add(u)
                    new_max, _ = _all_pairs_dist(edges + [cand])
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if new_max is not None and new_max < best_score:
                        best_score = new_max
                        best_cand = cand
                
                if best_cand is None:
                    break
                
                edges.append(best_cand)
                used.add((min(best_cand[0], best_cand[1]), max(best_cand[0], best_cand[1])))
                adj[best_cand[0]].add(best_cand[1])
                adj[best_cand[1]].add(best_cand[0])
            
            # Local search: remove edges
            if _validate_fast(edges) and len(edges) < best_count:
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(len(edges)):
                        if time.time() > deadline:
                            break
                        trial = edges[:i] + edges[i+1:]
                        if _validate_fast(trial):
                            edges = trial
                            improved = True
                            break
                
                if len(edges) < best_count:
                    best_edges = [list(e) for e in edges]
                    best_count = len(edges)
    
    if best_edges is None:
        best_edges = fallback
    
    result = []
    for edge in best_edges:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
