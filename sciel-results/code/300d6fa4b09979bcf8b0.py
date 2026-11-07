import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute valid bipartite edges
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if u % 2 != v % 2:
                valid_edges.append((u, v))
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_diameter_and_fars(adj):
        max_dist = 0
        far_pairs = []
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
                        if dist[v] > 4:
                            a, b = min(src, v), max(src, v)
                            far_pairs.append((a, b))
                        queue.append(v)
            if -1 in dist:
                return float('inf'), []
        return max_dist, far_pairs

    def is_valid(added_edges):
        adj = build_adj(added_edges)
        diam, _ = get_diameter_and_fars(adj)
        return diam <= 4

    # Inspiration 2's structural idea: use a hub with an independent dominating set for the remaining path segments
    # This gives a strong initial solution (7 edges) which we then try to improve
    def hub_ides_solution():
        hub = 11
        targets = [1, 4, 7, 15, 18, 21, 23]
        edges = set()
        for t in targets:
            edges.add((hub, t))
        return edges

    # Try to find a better solution using local search
    def local_search_reduce(added_edges, deadline):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                new_set = added_edges - {edge}
                if is_valid(new_set):
                    added_edges = new_set
                    improved = True
                    break
        return added_edges

    # Try to add edges strategically to reduce the number
    def local_search_swap(added_edges, deadline):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try removing each edge and see if we can compensate with a different one
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                new_set = added_edges - {edge}
                if not is_valid(new_set):
                    # Try to find a replacement edge
                    for candidate in valid_edges:
                        if candidate in added_edges:
                            continue
                        test_set = new_set | {candidate}
                        if is_valid(test_set):
                            added_edges = test_set
                            improved = True
                            break
                if improved:
                    break
        return added_edges

    # Start with the hub+IDES solution
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    current = hub_ides_solution()
    
    # First, try to reduce
    current = local_search_reduce(current, deadline)
    
    # Then try swaps
    current = local_search_swap(current, deadline)
    
    # If we still have more than 5 edges, try a different hub
    if len(current) > 5 and time.time() < deadline:
        for alt_hub in [10, 12, 9, 13]:
            if time.time() > deadline:
                break
            # Find IDS for the remaining segments with this hub
            # This is more complex, so we just try a few fixed patterns
            patterns = [
                [alt_hub, 1, 4, 7, 15, 18, 21, 23],
                [alt_hub, 0, 3, 6, 8, 15, 18, 22],
                [alt_hub, 1, 5, 8, 14, 17, 20, 23],
            ]
            for pattern in patterns:
                if time.time() > deadline:
                    break
                candidate = set()
                valid = True
                for t in pattern:
                    if t == alt_hub:
                        continue
                    if (min(alt_hub, t), max(alt_hub, t)) in base_edges:
                        valid = False
                        break
                    if alt_hub % 2 == t % 2:
                        valid = False
                        break
                    candidate.add((alt_hub, t))
                if valid and is_valid(candidate):
                    candidate = local_search_reduce(candidate, deadline)
                    candidate = local_search_swap(candidate, deadline)
                    if len(candidate) < len(current):
                        current = candidate
    
    # Convert to list of lists with Python ints
    result = [[int(u), int(v)] for u, v in current]
    return result
