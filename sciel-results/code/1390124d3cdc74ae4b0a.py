import time
import sys
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin for safety
    
    n = P['n']
    # Build adjacency list
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    def time_check():
        return time.time() > deadline
    
    def bfs_path(src, dst, blocked):
        """Find a simple path from src to dst avoiding blocked vertices (except src and dst)."""
        if src == dst:
            return [src]
        # BFS
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None
    
    def find_paths_for_branches(branches):
        """Try to find disjoint internal paths for all pairs of branches.
        Returns dict of pair -> path, or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Sort pairs: adjacent pairs first (they use no internal vertices),
        # then by some heuristic
        def pair_key(pair):
            u, v = pair
            if v in adj_list[u]:  # Check adjacency via set for speed
                return 0
            return 1
        
        # Actually, let's be smarter: process pairs in an order that's likely to work.
        # Pairs with fewer common neighbors should be processed first (more constrained).
        
        # For each pair, compute the set of vertices that could be internal
        # (adjacent to both, not a branch)
        pair_options = {}
        for u, v in pairs:
            if u in adj_list[v]:  # direct edge check
                pair_options[(u, v)] = 'direct'
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_list[v]:
                        common.add(w)
                pair_options[(u, v)] = common
        
        # Check feasibility: every non-direct pair needs at least one option
        for pair in pairs:
            if pair_options[pair] != 'direct' and len(pair_options[pair]) == 0:
                return None
        
        # Greedy with backtracking: assign internal vertices to pairs
        # Sort non-direct pairs by number of options (ascending)
        non_direct_pairs = [p for p in pairs if pair_options[p] != 'direct']
        non_direct_pairs.sort(key=lambda p: len(pair_options[p]))
        
        # Backtracking search
        assignment = {}  # pair -> list of internal vertices (path interior)
        used_internals = set()
        
        def backtrack(idx):
            if time_check():
                return False
            if idx == len(non_direct_pairs):
                return True
            pair = non_direct_pairs[idx]
            u, v = pair
            # Try to find a path from u to v using only unused internal vertices
            # and not using any branch vertices as internal
            # The path must go u -> ... -> v where internal vertices are all in 
            # the "available" set (not used, not branches)
            
            # Build blocked set: all branches + all used internals
            blocked = branch_set | used_internals
            # But u and v are allowed as endpoints
            blocked.discard(u)
            blocked.discard(v)
            
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            
            # Extract internal vertices
            internal = path[1:-1]
            
            # Check that all internal vertices are not in used_internals
            # (BFS already ensures they're not blocked, which includes used_internals)
            
            # Also verify internal vertices are distinct and not branches
            if len(internal) != len(set(internal)):
                return False
            for w in internal:
                if w in branch_set:
                    return False
            
            # Assign
            for w in internal:
                used_internals.add(w)
            assignment[pair] = internal
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            for w in internal:
                used_internals.discard(w)
            del assignment[pair]
            
            return False
        
        if not backtrack(0):
            return None
        
        # Build full paths
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_options[pair] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]
        
        return result_paths
    
    def solve():
        """Find maximum number of branches."""
        best_result = None
        best_k = 0
        
        # Try k from 7 down to 2
        for k in range(7, 1, -1):
            if time_check():
                break
            
            # Try combinations of k vertices
            # To speed up, we can prioritize vertices with high degree
            degree = [len(adj_list[i]) for i in range(n)]
            # Sort vertices by degree descending, take top candidates
            sorted_vertices = sorted(range(n), key=lambda x: -degree[x])
            
            # Try all combinations of k from the full set
            # But if k is large, this could be expensive. 
            # For k=7, C(24,7) = 346104, which is too many.
            # Let's try a smarter approach: random sampling + greedy
            
            # First, try a greedy approach to find a good set of k branches
            # Then try to verify and improve
            
            # Strategy: for each k, try multiple random subsets and verify
            import random
            random.seed(42)
            
            tried = set()
            found = False
            
            # Try all combinations if feasible, otherwise sample
            total_combos = 1
            for i in range(k):
                total_combos *= (n - i)
                total_combos //= (i + 1)
            
            if total_combos <= 50000:
                # Try all
                for combo in combinations(range(n), k):
                    if time_check():
                        break
                    combo_frozenset = frozenset(combo)
                    if combo_frozenset in tried:
                        continue
                    tried.add(combo_frozenset)
                    
                    paths = find_paths_for_branches(list(combo))
                    if paths is not None:
                        result = {
                            'branches': list(combo),
                            'paths': [paths[p] for p in sorted(paths.keys())]
                        }
                        if best_result is None or k > best_k:
                            best_result = result
                            best_k = k
                        found = True
                        break
            else:
                # Sample random combinations
                for attempt in range(20000):
                    if time_check():
                        break
                    combo = tuple(random.sample(range(n), k))
                    combo_frozenset = frozenset(combo)
                    if combo_frozenset in tried:
                        continue
                    tried.add(combo_frozenset)
                    
                    paths = find_paths_for_branches(list(combo))
                    if paths is not None:
                        result = {
                            'branches': list(combo),
                            'paths': [paths[p] for p in sorted(paths.keys())]
                        }
                        if best_result is None or k > best_k:
                            best_result = result
                            best_k = k
                        found = True
                        break
            
            if found:
                break
        
        if best_result is not None:
            return best_result
        
        # Fallback: find at least 2 branches
        for u, v in P['edges']:
            return {'branches': [u, v], 'paths': [[u, v]]}
        
        return {'branches': [0, 1], 'paths': [[0, 1]]}
    
    return solve()
# EVOLVE-BLOCK-END
