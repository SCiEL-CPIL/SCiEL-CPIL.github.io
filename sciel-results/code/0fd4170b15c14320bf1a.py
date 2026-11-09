"""New approach: Backtracking Disjoint Path Packing with Branch-and-Bound and Centrality Heuristics."""
# EVOLVE-BLOCK-START
import itertools
import random
import sys
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = int(P['n'])
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute distances for heuristics
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)

    # Compute simple betweenness centrality (approximate)
    # For n=24, we can do this efficiently
    betweenness = [0.0] * n
    for s in range(n):
        # BFS from s
        sigma = [0] * n # number of shortest paths
        dist_s = [-1] * n
        pred = [[] for _ in range(n)]
        sigma[s] = 1
        dist_s[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist_s[v] == -1:
                    dist_s[v] = dist_s[u] + 1
                    q.append(v)
                if dist_s[v] == dist_s[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        
        # Backward propagation
        delta = [0.0] * n
        order = [u for u in range(n) if dist_s[u] != -1]
        order.sort(key=lambda x: dist_s[x], reverse=True)
        
        for w in order:
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    
    # Score vertices: combine degree and betweenness
    scores = [len(adj[i]) + betweenness[i] for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: scores[x], reverse=True)

    def find_simple_paths(src, dst, blocked, max_len):
        """Generator yielding simple paths from src to dst avoiding blocked, len <= max_len."""
        if src == dst:
            yield [src]
            return
        if dst in blocked:
            return
        if dist[src][dst] == -1 or dist[src][dst] > max_len:
            return
            
        visited = {src}
        path = [src]
        
        def dfs(node):
            if len(path) > max_len:
                return
            if node == dst:
                yield list(path)
                return
            for neighbor in adj[node]:
                if neighbor in visited or neighbor in blocked:
                    continue
                if neighbor == dst:
                    path.append(neighbor)
                    yield list(path)
                    path.pop()
                else:
                    visited.add(neighbor)
                    path.append(neighbor)
                    yield from dfs(neighbor)
                    path.pop()
                    visited.remove(neighbor)
        
        yield from dfs(src)

    def pack_paths(branches):
        """
        Attempts to find disjoint paths for all pairs in branches.
        Returns dict of paths or None.
        """
        branches_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Sort pairs by "difficulty": shortest path length first (tightest constraints)
        # If distance is -1, it's impossible, so return None early
        for u, v in pairs:
            if dist[u][v] == -1:
                return None
        pairs.sort(key=lambda p: dist[p[0]][p[1]])
        
        used_interiors = set()
        paths_dict = {}
        
        def backtrack(idx):
            if idx == len(pairs):
                return True
            
            u, v = pairs[idx]
            blocked = branches_set - {u, v}
            blocked.update(used_interiors)
            
            # Find paths
            # We limit max_len to something reasonable, e.g., n
            max_len = n 
            # To avoid explosion, we might want to limit the number of paths tried?
            # But backtracking handles it.
            
            # Heuristic: Try shorter paths first? The generator yields them in DFS order,
            # which doesn't guarantee shortest first.
            # Let's collect a few shortest paths first?
            # For now, just iterate.
            
            for path in find_simple_paths(u, v, blocked, max_len):
                interiors = path[1:-1]
                # Check disjointness (should be guaranteed by blocked, but double check)
                if any(x in used_interiors for x in interiors):
                    continue
                
                new_used = used_interiors | set(interiors)
                paths_dict[(u, v)] = path
                paths_dict[(v, u)] = path
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                del paths_dict[(u, v)]
                del paths_dict[(v, u)]
                
            return False
        
        if backtrack(0):
            return paths_dict
        else:
            return None

    # Search for max k
    # We try k from a high bound down to 2
    # Upper bound for k: maybe 10?
    # Let's try up to 12.
    max_k_to_try = min(n, 12)
    
    best_result = None
    
    for k in range(max_k_to_try, 1, -1):
        found = False
        
        # Generate candidate sets
        # 1. Top k vertices
        candidates = [sorted_vertices[:k]]
        
        # 2. Random subsets from top 2k vertices
        top_pool = sorted_vertices[:min(n, 2*k)]
        for _ in range(50):
            cand = random.sample(top_pool, k)
            if cand not in candidates:
                candidates.append(cand)
        
        for branches in candidates:
            paths_dict = pack_paths(branches)
            if paths_dict is not None:
                # Construct result
                result_paths = []
                for u, v in itertools.combinations(branches, 2):
                    result_paths.append(paths_dict[(u, v)])
                
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': result_paths
                }
                found = True
                break
        
        if found:
            break

    if best_result:
        return best_result

    # Fallback
    if edges:
        u, v = edges[0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    else:
        return {
            'branches': [0, 1],
            'paths': [[0, 1]]
        }
# EVOLVE-BLOCK-END
