"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build edge set for quick lookup
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    
    def bfs_path(src, dst, blocked):
        """Find a simple path from src to dst avoiding blocked vertices (except src and dst)."""
        if src == dst:
            return [src]
        if dst in blocked and dst != src and dst != dst:
            return None
        # blocked should not contain src or dst
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in visited and w not in blocked:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
        return None
    
    def try_assign_paths(branches):
        """Try to assign internally disjoint paths for all pairs in branches.
        Returns dict of (u,v) -> path, or None if fails."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Shuffle pairs for diversity
        random.shuffle(pairs)
        used_internal = set()
        paths = {}
        for u, v in pairs:
            blocked = bset - {u, v}
            blocked |= used_internal
            path = bfs_path(u, v, blocked)
            if path is None:
                return None
            # Mark internal vertices as used
            for vertex in path[1:-1]:
                used_internal.add(vertex)
            paths[(u, v)] = path
            paths[(v, u)] = path[::-1]
        return paths
    
    # Find large cliques as starting points
    # For 24 vertices, we can try all subsets up to a certain size
    # But that's expensive. Let's use a greedy approach.
    
    def greedy_clique():
        """Find a large clique greedily."""
        remaining = list(range(n))
        clique = []
        while remaining:
            # Pick vertex with max degree to remaining
            best_v = None
            best_deg = -1
            for v in remaining:
                deg = len(adj[v] & set(remaining))
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v is None or best_deg == 0:
                break
            clique.append(best_v)
            remaining = [v for v in remaining if v in adj[best_v]]
        return clique
    
    # Try multiple random orderings
    best_result = None
    best_size = 0
    
    # First, try to find a large clique
    random.seed(42)
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Generate a random starting set
        # Mix of clique-based and random
        if attempt % 3 == 0:
            # Greedy clique
            branches = greedy_clique()
        elif attempt % 3 == 1:
            # Random set of size 5-8
            size = random.randint(5, 8)
            branches = random.sample(range(n), size)
        else:
            # Start from a clique, add random vertices
            base = greedy_clique()
            extra = [v for v in range(n) if v not in base]
            random.shuffle(extra)
            add_count = random.randint(1, 3)
            branches = base + extra[:add_count]
        
        branches = list(set(branches))
        if len(branches) < 2:
            continue
        
        # Try to assign paths
        paths = try_assign_paths(branches)
        if paths is not None:
            if len(branches) > best_size:
                best_size = len(branches)
                best_result = (branches, paths)
        
        # Try to grow the current best
        if best_result is not None:
            if time.time() > deadline:
                break
            cur_branches, cur_paths = best_result
            # Try adding one vertex
            candidates = [v for v in range(n) if v not in cur_branches]
            random.shuffle(candidates)
            for cand in candidates[:5]:
                if time.time() > deadline:
                    break
                new_branches = cur_branches + [cand]
                new_paths = try_assign_paths(new_branches)
                if new_paths is not None:
                    cur_branches = new_branches
                    cur_paths = new_paths
                    best_result = (cur_branches, cur_paths)
                    best_size = len(cur_branches)
    
    # Fallback: triangle
    if best_result is None:
        for row in combinations(range(n), 3):
            if all((a, b) in es for a, b in combinations(row, 2)):
                best_result = (list(row), {(a, b): [a, b] for a, b in combinations(row, 2)})
                break
        if best_result is None:
            u, v = P['edges'][0]
            best_result = ([u, v], {(u, v): [u, v]})
    
    branches, paths_dict = best_result
    
    # Format output
    branch_list = [int(b) for b in branches]
    path_list = []
    for u, v in combinations(branch_list, 2):
        p = paths_dict[(u, v)]
        path_list.append([int(x) for x in p])
    
    return {'branches': branch_list, 'paths': path_list}
# EVOLVE-BLOCK-END
