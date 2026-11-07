import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        queue.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist

    def is_triangle_free(edge_set):
        adj = build_adj(edge_set)
        for u, v in edge_set:
            for w in adj[u]:
                if w in adj[v] and w != u and w != v:
                    return False
        return True

    def is_valid(edge_set):
        if not is_triangle_free(edge_set):
            return False
        adj = build_adj(edge_set)
        # Check connectivity
        visited = [False] * n
        queue = deque([0])
        visited[0] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        if get_diameter(adj) > 4:
            return False
        return True

    # Constructive approach: use hub vertices to cover the path
    # Strategy: pick k hubs, connect them in a chain, and connect each non-hub to nearest hub
    # This creates a tree-like structure with small diameter
    
    best_solution = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 85
    
    # Try different numbers of hubs and hub placements
    for num_hubs in range(3, 10):
        if time.time() > deadline:
            break
        
        # Try evenly spaced hubs
        positions = [int(i * (n-1) / (num_hubs-1)) for i in range(num_hubs)]
        
        # Build edge set
        edge_set = set()
        
        # Connect hubs in a chain
        for i in range(num_hubs - 1):
            u, v = positions[i], positions[i+1]
            if u != v:
                edge_set.add((min(u, v), max(u, v)))
        
        # Connect each non-hub vertex to its nearest hub
        hub_set = set(positions)
        for v in range(n):
            if v not in hub_set:
                # Find nearest hub
                min_dist = float('inf')
                nearest_hub = None
                for h in positions:
                    d = abs(v - h)
                    if d < min_dist:
                        min_dist = d
                        nearest_hub = h
                edge_set.add((min(v, nearest_hub), max(v, nearest_hub)))
        
        # Check if valid
        if is_valid(edge_set):
            if len(edge_set) < best_count:
                best_count = len(edge_set)
                best_solution = set(edge_set)
        
        # Also try with some offset variations
        for offset in range(3):
            if time.time() > deadline:
                break
            
            positions = [int((i + offset) * (n-1) / (num_hubs-1)) % n for i in range(num_hubs)]
            # Ensure unique positions
            positions = sorted(set(positions))
            if len(positions) < num_hubs:
                continue
            
            edge_set = set()
            
            # Connect hubs in a chain
            for i in range(len(positions) - 1):
                u, v = positions[i], positions[i+1]
                if u != v:
                    edge_set.add((min(u, v), max(u, v)))
            
            # Connect each non-hub vertex to its nearest hub
            hub_set = set(positions)
            for v in range(n):
                if v not in hub_set:
                    min_dist = float('inf')
                    nearest_hub = None
                    for h in positions:
                        d = abs(v - h)
                        if d < min_dist:
                            min_dist = d
                            nearest_hub = h
                    edge_set.add((min(v, nearest_hub), max(v, nearest_hub)))
            
            if is_valid(edge_set):
                if len(edge_set) < best_count:
                    best_count = len(edge_set)
                    best_solution = set(edge_set)
    
    # Try a specific pattern: star-like with 3 centers
    # Centers at 4, 12, 20
    # Connect centers: (4,12), (12,20)
    # Connect each vertex to nearest center
    edge_set = set()
    centers = [4, 12, 20]
    edge_set.add((4, 12))
    edge_set.add((12, 20))
    
    center_set = set(centers)
    for v in range(n):
        if v not in center_set:
            min_dist = float('inf')
            nearest = None
            for c in centers:
                d = abs(v - c)
                if d < min_dist:
                    min_dist = d
                    nearest = c
            edge_set.add((min(v, nearest), max(v, nearest)))
    
    if is_valid(edge_set):
        if len(edge_set) < best_count:
            best_count = len(edge_set)
            best_solution = set(edge_set)
    
    # Try 4 centers: 3, 9, 15, 21
    edge_set = set()
    centers = [3, 9, 15, 21]
    for i in range(len(centers) - 1):
        edge_set.add((min(centers[i], centers[i+1]), max(centers[i], centers[i+1])))
    
    center_set = set(centers)
    for v in range(n):
        if v not in center_set:
            min_dist = float('inf')
            nearest = None
            for c in centers:
                d = abs(v - c)
                if d < min_dist:
                    min_dist = d
                    nearest = c
            edge_set.add((min(v, nearest), max(v, nearest)))
    
    if is_valid(edge_set):
        if len(edge_set) < best_count:
            best_count = len(edge_set)
            best_solution = set(edge_set)
    
    if best_solution is None:
        # Fallback: use the greedy approach as a last resort
        edge_set = set()
        adj = build_adj(edge_set)
        
        while True:
            if time.time() > deadline:
                break
            
            diam = get_diameter(adj)
            if diam <= 4:
                break
            
            # Find far pairs
            all_dists = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                all_dists.append(dist)
            
            far_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if all_dists[i][j] > 4:
                        far_pairs.append((i, j))
            
            if not far_pairs:
                break
            
            # Pick the first far pair that doesn't create a triangle
            added = False
            for u, v in far_pairs:
                if (u, v) in edge_set or (u, v) in base_edges:
                    continue
                # Check triangle
                conflict = False
                for w in adj[u]:
                    if w in adj[v] and w != u and w != v:
                        conflict = True
                        break
                if not conflict:
                    edge_set.add((u, v))
                    adj = build_adj(edge_set)
                    added = True
                    break
            
            if not added:
                break
        
        if is_valid(edge_set):
            return [[int(u), int(v)] for u, v in edge_set]
        
        # Absolute fallback
        edges = []
        for i in range(20):
            edges.append([i, i+4])
        return edges
    
    return [[int(u), int(v)] for u, v in best_solution]
