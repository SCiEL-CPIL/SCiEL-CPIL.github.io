import time
import random
import sys
from itertools import combinations

P = {'n': 16}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # All possible undirected edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    # Precompute adjacency from edge set (fast version using bitmasks)
    def build_adjacency(edge_set):
        adj = [0] * n  # bitmask for each vertex
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def bfs_distances(adj):
        """Compute all-pairs shortest distances using BFS with bitmasks."""
        dist = [[-1] * n for _ in range(n)]
        for src in range(n):
            visited = [False] * n
            dist[src][src] = 0
            visited[src] = True
            current = [src]
            d = 1
            while current and d < n:
                next_frontier = []
                for u in current:
                    neighbors = adj[u]
                    for v in range(n):
                        if (neighbors >> v) & 1 and not visited[v]:
                            visited[v] = True
                            dist[src][v] = d
                            next_frontier.append(v)
                current = next_frontier
                d += 1
        return dist
    
    def diameter(dist):
        mx = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] == -1:
                    return -1  # disconnected
                if dist[i][j] > mx:
                    mx = dist[i][j]
        return mx
    
    def is_valid_diameter2(dist):
        """Check diameter is exactly 2 (connected and all pairs <= 2, some pair = 2)."""
        mx = 0
        has_pair_2 = False
        for i in range(n):
            for j in range(i+1, n):
                if dist[i][j] == -1:
                    return False
                if dist[i][j] > 2:
                    return False
                if dist[i][j] == 2:
                    has_pair_2 = True
                mx = max(mx, dist[i][j])
        return has_pair_2  # must have at least one pair at distance exactly 2
    
    def check_essentiality(edge_set):
        """Check that removing each edge breaks diameter-2 or disconnects."""
        m = len(edge_set)
        for (u, v) in edge_set:
            new_set = edge_set - {(u, v)}
            adj = build_adjacency(new_set)
            dist = bfs_distances(adj)
            if is_valid_diameter2(dist):
                return False  # this edge is NOT essential
        return True
    
    def score(edge_set):
        """Score: higher is better. Penalize non-essential edges, reward valid diameter-2."""
        if len(edge_set) == 0:
            return -10000
        adj = build_adjacency(edge_set)
        dist = bfs_distances(adj)
        if not is_valid_diameter2(dist):
            # Penalize: count how many pairs are bad
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i][j] == -1:
                        penalty += 10
                    elif dist[i][j] > 2:
                        penalty += 5
            return -penalty
        # Valid diameter-2: check essentiality
        non_essential = 0
        for (u, v) in edge_set:
            new_set = edge_set - {(u, v)}
            adj2 = build_adjacency(new_set)
            dist2 = bfs_distances(adj2)
            if is_valid_diameter2(dist2):
                non_essential += 1
        # We want: valid diameter-2 AND all edges essential
        # Score: edge_count * 10 - non_essential * 100
        return len(edge_set) * 10 - non_essential * 100
    
    # Initialize: start with a star (diameter 2, but not all edges essential)
    # Actually star has diameter 2 and removing any edge disconnects, so star IS valid!
    # Let's verify: star on 16 vertices, remove edge (0,i), vertex i is disconnected. Valid!
    # So star is a valid solution with 15 edges. We want to add more edges.
    
    current_edges = frozenset((0, i) for i in range(1, n))
    current_score = score(current_edges)
    best_edges = current_edges
    best_score = current_score
    
    # Simulated annealing
    temperature = 500.0
    cooling = 0.995
    min_temp = 1.0
    max_steps = 50000
    
    step = 0
    rng = random.Random(42)
    
    while step < max_steps and temperature > min_temp:
        if time.time() > deadline:
            break
        
        step += 1
        
        # Propose a move: add or remove a random edge
        move_type = rng.random()
        if move_type < 0.45:
            # Add a random edge
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates:
                temperature *= cooling
                continue
            edge = rng.choice(candidates)
            new_edges = frozenset(current_edges | {edge})
        elif move_type < 0.9:
            # Remove a random edge
            if not current_edges:
                temperature *= cooling
                continue
            edge = rng.choice(tuple(current_edges))
            new_edges = frozenset(current_edges - {edge})
        else:
            # Swap: remove one, add one
            if not current_edges:
                temperature *= cooling
                continue
            remove_edge = rng.choice(tuple(current_edges))
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates:
                temperature *= cooling
                continue
            add_edge = rng.choice(candidates)
            new_edges = frozenset((current_edges - {remove_edge}) | {add_edge})
        
        new_score = score(new_edges)
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / temperature):
            current_edges = new_edges
            if new_score > best_score:
                best_score = new_score
                best_edges = new_edges
        
        temperature *= cooling
    
    # Final polish: try to remove non-essential edges greedily
    # Then try to add edges that maintain essentiality
    polished = set(best_edges)
    changed = True
    while changed:
        changed = False
        for edge in list(polished):
            test_set = polished - {edge}
            if not test_set:
                break
            adj = build_adjacency(test_set)
            dist = bfs_distances(adj)
            if is_valid_diameter2(dist):
                # Check if all remaining edges are essential
                essential = True
                for (u, v) in test_set:
                    test2 = test_set - {(u, v)}
                    adj2 = build_adjacency(test2)
                    dist2 = bfs_distances(adj2)
                    if is_valid_diameter2(dist2):
                        essential = False
                        break
                if essential:
                    polished = test_set
                    changed = True
                    break
        if time.time() > deadline:
            break
    
    # Try adding edges
    for edge in all_edges:
        if edge in polished:
            continue
        if time.time() > deadline:
            break
        test_set = polished | {edge}
        adj = build_adjacency(test_set)
        dist = bfs_distances(adj)
        if not is_valid_diameter2(dist):
            continue
        # Check essentiality of all edges in test_set
        essential = True
        for (u, v) in test_set:
            test2 = test_set - {(u, v)}
            adj2 = build_adjacency(test2)
            dist2 = bfs_distances(adj2)
            if is_valid_diameter2(dist2):
                essential = False
                break
        if essential:
            polished = test_set
    
    # Final validation
    final_adj = build_adjacency(polished)
    final_dist = bfs_distances(final_adj)
    if not is_valid_diameter2(final_dist):
        # Fallback to star
        polished = set((0, i) for i in range(1, n))
    
    # Verify essentiality one more time
    for (u, v) in list(polished):
        test_set = polished - {(u, v)}
        adj = build_adjacency(test_set)
        dist = bfs_distances(adj)
        if is_valid_diameter2(dist):
            polished.discard((u, v))
    
    # Ensure we have a valid answer
    final_adj = build_adjacency(polished)
    final_dist = bfs_distances(final_adj)
    if not is_valid_diameter2(final_dist):
        polished = set((0, i) for i in range(1, n))
    
    # Convert to list of lists
    result = [[u, v] for (u, v) in sorted(polished)]
    return result
