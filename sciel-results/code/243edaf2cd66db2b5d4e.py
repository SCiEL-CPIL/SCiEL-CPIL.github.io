import time
import random
from collections import defaultdict, deque
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def min_weight_perfect_matching(vertices, weights):
        """
        vertices: list of vertex indices (relative to the T-set)
        weights: dict (i, j) -> weight
        Returns: list of matched pairs (i, j) and total weight
        """
        n = len(vertices)
        if n == 0:
            return [], 0
        
        # DP over subsets
        # mask represents which vertices are already matched
        # dp[mask] = min weight to match vertices in mask
        # We use bitmasks. n <= 30, but |T| is usually much smaller, e.g., 10-20.
        # If |T| is large (e.g., 30), 2^15 = 32768 states. Fast.
        
        size = 1 << n
        dp = [float('inf')] * size
        dp[0] = 0
        parent = [-1] * size # store the last vertex added to the mask for reconstruction
        
        for mask in range(size):
            if dp[mask] == float('inf'):
                continue
            
            # Find the first unmatched vertex in mask
            # Actually, standard DP: pick the lowest set bit that is NOT in mask?
            # No, mask usually represents the set of MATCHED vertices.
            # Let's use mask = set of UNMATCHED vertices.
            # dp[mask] = min cost to match all vertices in mask.
            
            # Find lowest set bit in mask
            if mask == 0:
                continue
            
            # Find the first vertex v in mask
            v_bit = 1
            while not (mask & v_bit):
                v_bit <<= 1
            v_idx = v_bit.bit_length() - 1
            
            # Try to match v_idx with any other u in mask
            rest_mask = mask ^ v_bit
            u_bit = 1
            while u_bit < (1 << n):
                if rest_mask & u_bit:
                    u_idx = u_bit.bit_length() - 1
                    if v_idx < u_idx: # Avoid double counting
                        new_mask = rest_mask ^ u_bit
                        w = weights.get((v_idx, u_idx), float('inf'))
                        if dp[new_mask] + w < dp[mask]:
                            dp[mask] = dp[new_mask] + w
                            parent[mask] = (v_idx, u_idx)
                u_bit <<= 1
                
        # Reconstruct
        pairs = []
        mask = (1 << n) - 1
        while mask:
            if parent[mask] == -1:
                break
            v, u = parent[mask]
            pairs.append((v, u))
            mask ^= (1 << v)
            mask ^= (1 << u)
            
        return pairs, dp[(1 << n) - 1]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        # 1. Identify odd degree vertices
        degrees = defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
        
        # If no odd vertices, graph is already Eulerian
        # T-Join is empty
        
        edges_to_remove = set()
        
        if odd_vertices:
            # 2. Compute shortest paths between all odd vertices
            # Use Dijkstra from each odd vertex
            n_nodes = max(edges)[1] + 1 if edges else 0
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append((v, 1))
                adj[v].append((u, 1))
                
            # dist[i][j] = shortest distance between odd_vertices[i] and odd_vertices[j]
            k = len(odd_vertices)
            dist = [[0] * k for _ in range(k)]
            
            for i, src in enumerate(odd_vertices):
                # Dijkstra
                dist_src = [float('inf')] * n_nodes
                dist_src[src] = 0
                pq = [(0, src)]
                visited = [False] * n_nodes
                
                while pq:
                    d, u = heapq.heappop(pq)
                    if visited[u]:
                        continue
                    visited[u] = True
                    for v, w in adj[u]:
                        if not visited[v] and dist_src[u] + w < dist_src[v]:
                            dist_src[v] = dist_src[u] + w
                            heapq.heappush(pq, (dist_src[v], v))
                            
                for j in range(i + 1, k):
                    dst = odd_vertices[j]
                    dist[i][j] = dist_src[dst]
                    dist[j][i] = dist_src[dst]
                    
            # 3. Min Weight Perfect Matching on odd_vertices
            weights = {}
            for i in range(k):
                for j in range(i + 1, k):
                    weights[(i, j)] = dist[i][j]
                    
            pairs, _ = min_weight_perfect_matching(odd_vertices, weights)
            
            # 4. Reconstruct paths for the matching pairs
            # For each pair (u, v) in pairs, find the shortest path in G
            # We need to store parent pointers from Dijkstra to reconstruct paths
            # Let's re-run Dijkstra or store parents.
            # Since n is small, we can just re-run or use a simple BFS/DFS for unweighted graph shortest path
            # Actually, we already have dist. We just need the edges.
            
            # Let's use a simple function to find shortest path edges
            def get_shortest_path_edges(src, dst):
                # BFS
                parent = {src: None}
                queue = deque([src])
                while queue:
                    u = queue.popleft()
                    if u == dst:
                        break
                    for v, _ in adj[u]:
                        if v not in parent:
                            parent[v] = u
                            queue.append(v)
                            
                path = []
                cur = dst
                while cur != src:
                    prev = parent[cur]
                    path.append((min(prev, cur), max(prev, cur)))
                    cur = prev
                return path
            
            for i, j in pairs:
                u = odd_vertices[i]
                v = odd_vertices[j]
                path_edges = get_shortest_path_edges(u, v)
                for e in path_edges:
                    edges_to_remove.add(e)
                    
        # 5. Remaining edges form Eulerian graph
        remaining_edges = set(edges) - edges_to_remove
        
        pieces = []
        
        # 6. Decompose remaining_edges into cycles
        # Greedy: Find any cycle, remove it, repeat.
        # To bias towards longer cycles, we can start from high-degree vertices.
        
        while remaining_edges:
            if time.time() > time_limit:
                break
                
            # Find a cycle
            # Pick a vertex with highest degree in remaining graph
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                
            if not deg:
                break
                
            start = max(deg, key=deg.get)
            
            # DFS to find a cycle starting/ending at start
            # We want a simple cycle.
            # Path: [start, v1, v2, ...]
            # If we hit start again, we have a cycle.
            
            visited = {start}
            path = [start]
            cycle = None
            
            def dfs(u):
                nonlocal cycle
                if cycle:
                    return
                for v, _ in adj[u]:
                    # Check if (u,v) is in remaining_edges
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        continue
                    if v == start and len(path) > 1:
                        cycle = list(path)
                        return
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs(v)
                        if cycle:
                            return
                        path.pop()
                        visited.remove(v)
                        
            dfs(start)
            
            if cycle:
                pieces.append(cycle)
                # Remove edges
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    e = (min(u, v), max(u, v))
                    remaining_edges.discard(e)
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                remaining_edges.discard(e)
            else:
                # Should not happen if remaining_edges is Eulerian and non-empty
                # But if there's a bug or time limit, break
                break
                
        # 7. Add single edges (the T-Join edges)
        for e in edges_to_remove:
            pieces.append(list(e))
            
        return pieces

    import heapq # Needed for Dijkstra
    
    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        graphs_left = len(graphs) - i
        time_per_graph = max(1.0, remaining_time / graphs_left * 0.9)
        time_limit = time.time() + time_per_graph
        
        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
    print(f"Total pieces: {sum(len(r) for r in res)}")
