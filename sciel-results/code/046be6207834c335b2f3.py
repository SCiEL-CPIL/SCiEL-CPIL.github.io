"""Search-based solver: generate many candidate branch sets, keep the best valid one."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def time_up():
        return time.time() - t_start > DEADLINE
    
    def find_path(u, v, banned):
        """Find a simple path from u to v avoiding vertices in banned set."""
        from collections import deque
        if u in banned or v in banned:
            return None
        if u == v:
            return [u]
        # BFS
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            if cur == v:
                # Reconstruct path
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in parent and nb not in banned:
                    parent[nb] = cur
                    q.append(nb)
        return None
    
    def try_branch_set(branches):
        """Try to find disjoint paths for all pairs in branches. Returns paths dict or None."""
        if len(branches) < 2:
            return None
        banned = set(branches)
        paths = {}
        # Order pairs by "difficulty" - try shorter paths first to leave more room
        pairs = list(combinations(branches, 2))
        # Sort by distance (shorter first) to find easy paths early
        def pair_key(p):
            u, v = p
            # Estimate distance via BFS depth
            return 0
        # Just process in order; use BFS which finds shortest paths
        for u, v in pairs:
            path = find_path(u, v, banned)
            if path is None:
                return None
            # Mark interior vertices as banned
            for node in path[1:-1]:
                if node in banned and node not in branches:
                    return None  # Shouldn't happen
                banned.add(node)
            paths[(u, v)] = path
        return paths
    
    def greedy_build():
        """Greedy: start with a triangle or edge, try to add vertices one by one."""
        best_branches = None
        best_paths = None
        
        # Try all starting pairs
        for u, v in edges:
            if time_up():
                break
            # Start with this edge
            branches = [u, v]
            paths = try_branch_set(branches)
            if paths is None:
                continue
            if best_branches is None or len(branches) > len(best_branches):
                best_branches = list(branches)
                best_paths = dict(paths)
            
            # Try to add more vertices
            changed = True
            while changed and not time_up():
                changed = False
                for w in range(n):
                    if w in branches:
                        continue
                    new_branches = branches + [w]
                    new_paths = try_branch_set(new_branches)
                    if new_paths is not None:
                        branches = new_branches
                        paths = new_paths
                        if best_branches is None or len(branches) > len(best_branches):
                            best_branches = list(branches)
                            best_paths = dict(paths)
                        changed = True
                        break
        
        return best_branches, best_paths
    
    def random_restart():
        """Randomized search: pick random starting sets and try to grow."""
        best_branches = None
        best_paths = None
        
        for _ in range(200):
            if time_up():
                break
            
            # Random starting pair
            u, v = random.choice(edges)
            branches = [u, v]
            paths = try_branch_set(branches)
            if paths is None:
                continue
            
            # Try to grow randomly
            for _ in range(50):
                if time_up():
                    break
                # Pick a random vertex to try adding
                candidates = [w for w in range(n) if w not in branches]
                if not candidates:
                    break
                w = random.choice(candidates)
                new_branches = branches + [w]
                new_paths = try_branch_set(new_branches)
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
                    if best_branches is None or len(branches) > len(best_branches):
                        best_branches = list(branches)
                        best_paths = dict(paths)
        
        return best_branches, best_paths
    
    # First try greedy
    gb, gp = greedy_build()
    if gb is not None and gp is not None:
        return {'branches': list(gb), 'paths': [list(gp[p]) for p in combinations(gb, 2)]}
    
    # Fallback: random restarts
    rb, rp = random_restart()
    if rb is not None and rp is not None:
        return {'branches': list(rb), 'paths': [list(rp[p]) for p in combinations(rb, 2)]}
    
    # Last resort: single edge
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}
# EVOLVE-BLOCK-END
