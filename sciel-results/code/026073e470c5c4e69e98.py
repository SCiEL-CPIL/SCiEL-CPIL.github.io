import time
import random
from itertools import combinations
from collections import deque
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Safety margin

    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]

    # Helper: Max Flow for vertex-disjoint paths
    # Uses simple BFS-based max flow (Edmonds-Karp) on split graph
    def max_vertex_disjoint_paths(src, dst, forbidden):
        """
        Returns max number of internally disjoint paths from src to dst
        avoiding vertices in 'forbidden' (except src and dst).
        """
        if src == dst:
            return 0
        if src in forbidden or dst in forbidden:
            return 0
            
        # Split vertices: v -> 2*v (in), 2*v+1 (out)
        # Edges: (2*u, 2*u+1) cap 1 (or inf for src/dst)
        # (2*u+1, 2*v) cap inf for u~v
        N_split = 2 * n
        # Use a simple flow network representation
        # Since N is small, we can just do BFS on the residual graph
        
        # Build residual graph structure
        # flow[u][v]
        flow = [[0] * N_split for _ in range(N_split)]
        
        # Add edges
        for u in range(n):
            if u in forbidden and u != src and u != dst:
                continue
            cap = float('inf') if u == src or u == dst else 1
            flow[2*u][2*u+1] = cap
            
        for u in range(n):
            if u in forbidden and u != src and u != dst:
                continue
            for v in adj[u]:
                if v in forbidden and v != src and v != dst:
                    continue
                # Edge from u_out to v_in
                flow[2*u+1][2*v] = float('inf')
                # Edge from v_out to u_in (undirected)
                flow[2*v+1][2*u] = float('inf')

        total_flow = 0
        while True:
            # BFS to find augmenting path
            visited = [False] * N_split
            parent = [-1] * N_split
            q = deque([2*src]) # Start at src_in? No, src_out is better for outgoing?
            # Actually, standard split: src_in -> src_out -> ... -> dst_in -> dst_out?
            # Let's say we enter src_in, go to src_out, then to neighbors.
            # And we end at dst_in? Or dst_out?
            # If we end at dst_in, we haven't used dst's capacity.
            # Usually, we want to enter dst_in and stop.
            # So source is 2*src, sink is 2*dst.
            visited[2*src] = True
            while q:
                u = q.popleft()
                if u == 2*dst:
                    break
                for v in range(N_split):
                    if not visited[v] and flow[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        q.append(v)
            
            if not visited[2*dst]:
                break
                
            # Find min capacity on path
            path_cap = float('inf')
            v = 2*dst
            while v != 2*src:
                u = parent[v]
                path_cap = min(path_cap, flow[u][v])
                v = u
                
            # Update residual graph
            v = 2*dst
            while v != 2*src:
                u = parent[v]
                flow[u][v] -= path_cap
                flow[v][u] += path_cap
                v = u
                
            total_flow += path_cap
            
        return total_flow

    # Compute Vertex Connectivity kappa(G)
    # If graph is complete, kappa = n-1
    is_complete = True
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                is_complete = False
                break
        if not is_complete:
            break
            
    if is_complete:
        kappa = n - 1
    else:
        kappa = n
        # Check non-adjacent pairs
        for u in range(n):
            if time.time() > deadline: break
            for v in range(u+1, n):
                if v not in adj[u]:
                    f = max_vertex_disjoint_paths(u, v, set())
                    if f < kappa:
                        kappa = f
                    if kappa == 0:
                        break
            if kappa == 0:
                break
                
    K_max = kappa + 1
    if K_max > n:
        K_max = n

    # Bron-Kerbosch for Max Clique
    def find_max_clique():
        best = []
        def bronk(R, P_set, X):
            nonlocal best
            if not P_set and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P_set:
                return
            pivot_candidates = list(P_set | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
                candidates = list(P_set - adj[pivot])
            else:
                candidates = list(P_set)
            for v in candidates:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        bronk(set(), set(range(n)), set())
        return best

    max_clique = find_max_clique()

    def find_shortest_path(u, v, forbidden):
        if u == v: return None
        visited = {u}
        parent = {u: None}
        q = deque([u])
        while q:
            curr = q.popleft()
            if curr == v:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[curr]:
                if nb not in visited and (nb == v or nb not in forbidden):
                    visited.add(nb)
                    parent[nb] = curr
                    q.append(nb)
        return None

    def try_subdivision(branches):
        k = len(branches)
        if k < 2: return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Pre-compute "difficulty" using max flow in original graph
        # This helps ordering
        pair_info = []
        for u, v in pairs:
            # Use a fast heuristic: shortest path length
            d = 1
            # Simple BFS for dist
            dist = [n+1]*n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v: break
                for nb in adj[curr]:
                    if dist[nb] == n+1:
                        dist[nb] = dist[curr] + 1
                        q.append(nb)
            d = dist[v]
            if d == n+1: return None
            pair_info.append((d, u, v))
        
        # Sort by longest distance first (most constrained?)
        # Or by fewest available paths?
        # Let's stick to distance for now, it's a proxy.
        pair_info.sort(key=lambda x: -x[0])

        paths = {}
        used_int = set()

        def backtrack(idx):
            if idx == len(pair_info):
                return True
            d, u, v = pair_info[idx]
            forbidden = used_int | (branch_set - {u, v})
            
            # Check feasibility with Max Flow
            # If max flow is 0, no path exists
            # This is expensive, so maybe only do it if simple BFS fails?
            # Or do it always for pruning?
            # Let's try BFS first.
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return False
            
            interiors = set(path[1:-1])
            if interiors & used_int:
                return False
                
            # Try this path
            paths[(u, v)] = path
            used_int.update(interiors)
            
            if backtrack(idx + 1):
                return True
                
            del paths[(u, v)]
            used_int.difference_update(interiors)
            
            # If shortest path failed, try others?
            # For now, we just fail. This is the weakness.
            # To improve, we would need to find the next shortest path.
            # Given time constraints, we rely on trying many branch sets.
            return False

        if backtrack(0):
            return paths
        return None

    best_result = None
    best_k = 0

    # Generate candidate sets
    # 1. Max clique
    # 2. Top K by degree
    # 3. Random samples

    # Try to find a valid set for K from K_max down to 2
    for K in range(min(K_max, n), 1, -1):
        if time.time() > deadline:
            break
            
        candidates = set()
        
        # Add max clique if large enough
        if len(max_clique) >= K:
            # Take subsets of max clique of size K
            # Just try the full max clique if it's exactly K, or a subset
            candidates.add(tuple(sorted(max_clique[:K])))
            
        # Add top K by degree
        top_k = sorted(range(n), key=lambda x: -degrees[x])[:K]
        candidates.add(tuple(sorted(top_k)))
        
        # Add random candidates
        num_random = 100
        for _ in range(num_random):
            if time.time() > deadline:
                break
            # Weighted sampling
            weights = [degrees[i] + 1 for i in range(n)]
            cand = random.sample(range(n), K, weights=weights)
            candidates.add(tuple(sorted(cand)))
            
        # Try each candidate
        for cand in candidates:
            if time.time() > deadline:
                break
            result = try_subdivision(list(cand))
            if result is not None:
                best_result = (list(cand), result)
                best_k = K
                break # Found one for this K, go to next K (smaller)
        
        if best_result is not None:
            break

    if best_result is None:
        # Fallback: try any 2 vertices
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    best_result = ([u, v], {(u, v): [u, v]})
                    break
            if best_result: break

    branches, paths = best_result
    branches_out = [int(b) for b in branches]
    paths_out = []
    for (u, v), path in sorted(paths.items()):
        paths_out.append([int(x) for x in path])

    return {'branches': branches_out, 'paths': paths_out}
