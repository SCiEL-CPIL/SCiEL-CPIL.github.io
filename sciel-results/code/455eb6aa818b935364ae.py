import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute all simple paths between all pairs? No, too many.
    # We will compute them on the fly with constraints.
    
    def find_paths(branches, used_interiors):
        # branches: list of branch vertices
        # used_interiors: set of vertices that cannot be used as interiors
        # Returns a dict mapping (u, v) to path, or None if impossible
        # We need to find paths for all pairs in branches
        
        # Optimization: if len(branches) <= 1, trivial
        if len(branches) <= 1:
            return {tuple(sorted([b])): [b] for b in branches} # Dummy, not used
        
        pairs = list(combinations(sorted(branches), 2))
        # We will use backtracking
        # State: index of next pair to assign, current paths, current used_interiors
        # To speed up, we pick the pair with fewest options first?
        # Let's just iterate in a fixed order for simplicity, or randomize?
        # Fixed order is deterministic.
        
        # Let's try to find paths one by one.
        # We need to be careful about the order.
        # A heuristic: pick the pair with the shortest distance first?
        # Or just the first one.
        
        # We'll use a recursive function
        paths_found = {}
        
        def backtrack(pair_idx, used):
            if pair_idx == len(pairs):
                return True
            
            u, v = pairs[pair_idx]
            # Find all simple paths from u to v avoiding 'used' and branches except u, v
            # Branches that are not u or v cannot be used in the path
            forbidden = set(used)
            forbidden.update(branches)
            forbidden.discard(u)
            forbidden.discard(v)
            
            # BFS to find paths
            # We want shortest paths first
            queue = deque()
            queue.append((u, [u]))
            visited = {u}
            found_paths = []
            
            while queue:
                node, path = queue.popleft()
                if node == v:
                    found_paths.append(path)
                    # Don't break, find more? Or just take the first few?
                    # Taking the first (shortest) is usually good.
                    # But sometimes we need to try others.
                    # Let's collect up to 10 shortest paths.
                    if len(found_paths) >= 10:
                        break
                    continue
                
                for neighbor in adj[node]:
                    if neighbor in forbidden:
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
            
            # Shuffle found_paths to randomize choice?
            random.shuffle(found_paths)
            
            for path in found_paths:
                # Mark interiors as used
                new_used = set(used)
                new_used.update(path[1:-1])
                
                if backtrack(pair_idx + 1, new_used):
                    paths_found[(u, v)] = path
                    return True
            
            # If no path worked, try next pair order?
            # This simple backtracking might get stuck if the order is bad.
            # To improve, we can try to swap the order of pairs.
            # For now, let's rely on the fact that we restart many times.
            return False
        
        if backtrack(0, used_interiors):
            return paths_found
        return None

    # Search Strategy
    best_branches = []
    best_paths = {}
    
    deadline = time.time() + 50
    
    # Start with max clique?
    # Let's find a large clique first
    def find_max_clique():
        # Simple branch and bound or just try all subsets of size 6, 7...
        # Given n=24, we can try to find a clique of size 6 quickly.
        # The seed finds one.
        # Let's just use the seed's logic to get an initial solution.
        es = {tuple(e) for e in P['edges']}
        for row in combinations(range(n), 6):
            if all(e in es for e in combinations(row, 2)):
                return list(row)
        return [0, 1]
        
    initial_clique = find_max_clique()
    # Check if initial_clique is a valid subdivision (it is, paths are edges)
    # But we want to expand it.
    
    # Randomized expansion
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        # Start with a random set of vertices
        # Maybe start with the clique and add random vertices?
        # Or just random vertices?
        
        # Let's try starting with a random subset of size 4 or 5
        # And try to expand it.
        
        # Strategy 1: Start with clique, add random vertices
        B = list(initial_clique)
        random.shuffle(B)
        
        # Try to add vertices
        added = True
        while added and time.time() < deadline:
            added = False
            candidates = [v for v in range(n) if v not in B]
            random.shuffle(candidates)
            for v in candidates:
                new_B = B + [v]
                # Check feasibility
                res = find_paths(new_B, set())
                if res is not None:
                    B = new_B
                    best_branches = B
                    best_paths = res
                    added = True
                    break # Restart the while loop to try adding another
        
        # Strategy 2: Random start
        if time.time() > deadline:
            break
        size = random.randint(4, 8)
        B = random.sample(range(n), size)
        res = find_paths(B, set())
        if res is not None:
            # Try to expand this random set
            added = True
            while added and time.time() < deadline:
                added = False
                candidates = [v for v in range(n) if v not in B]
                random.shuffle(candidates)
                for v in candidates:
                    new_B = B + [v]
                    res_new = find_paths(new_B, set())
                    if res_new is not None:
                        B = new_B
                        best_branches = B
                        best_paths = res_new
                        added = True
                        break

    # Ensure at least 2 branches
    if len(best_branches) < 2:
        # Fallback to edge
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
        
    # Format output
    branches_out = list(best_branches)
    paths_out = []
    for u, v in combinations(sorted(best_branches), 2):
        # The key in best_paths is (u, v) or (v, u)?
        # In find_paths, we used sorted(branches) and combinations, so keys are sorted tuples.
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path:
            paths_out.append(list(path))
        else:
            # Should not happen if best_paths was from a valid check
            # But if we updated best_branches without updating best_paths correctly?
            # We did update.
            pass
            
    return {'branches': branches_out, 'paths': paths_out}

# Test locally? No, just output.
