"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    n = P['n']
    edges = P['edges']
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find a maximal clique to start with
    # Simple greedy clique finding
    def find_maximal_clique():
        # Use Bron-Kerbosch or simple heuristic
        # For n=24, we can just check all subsets? No.
        # Greedy: pick highest degree, add neighbors, etc.
        # Or just return a small clique found by checking triangles.
        # Let's find the largest clique using a simple backtracking with pruning.
        best_clique = []
        vertices = list(range(n))
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
            # Pivot selection
            if P:
                pivot = max(P, key=lambda x: len(adj[x] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = []
                
            for v in candidates:
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}
                
        bron_kerbosch(set(), set(vertices), set())
        return best_clique

    initial_clique = find_maximal_clique()
    best_k = len(initial_clique)
    best_branches = initial_clique
    best_paths = {}
    if best_k >= 3:
        for u, v in combinations(best_branches, 2):
            if u in adj[v]:
                best_paths[(u, v)] = [u, v]
                best_paths[(v, u)] = [v, u]
            else:
                # Should not happen for a clique
                pass
    elif best_k >= 2:
        u, v = best_branches[0], best_branches[1]
        best_paths[(u, v)] = [u, v]
        best_paths[(v, u)] = [v, u]
    else:
        # Fallback
        if edges:
            u, v = edges[0]
            best_branches = [u, v]
            best_k = 2
            best_paths = {(u, v): [u, v], (v, u): [v, u]}
        else:
            return {'branches': [0, 1], 'paths': [[0, 1]]}

    def find_shortest_path(u, v, forbidden):
        """Find shortest path from u to v avoiding 'forbidden' vertices (except u and v)."""
        if u == v:
            return [u]
        if v in forbidden and v != u and v != v: # v is never in forbidden in this call context usually
            return None
        visited = {u}
        parent = {u: None}
        queue = deque([u])
        while queue:
            curr = queue.popleft()
            if curr == v:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1]
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    # Allow u and v to be in path, but no other forbidden
                    if neighbor == u or neighbor == v:
                        pass
                    else:
                        if neighbor in forbidden:
                            continue
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)
        return None

    def check_subdivision(branches):
        """
        Check if 'branches' can form a valid clique subdivision.
        Returns dict of paths if valid, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Sort pairs by shortest path length in the original graph (heuristic for difficulty)
        # We calculate distance in G without considering other paths yet.
        def get_dist(u, v):
            # BFS
            if u == v: return 0
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if curr == v:
                    dist = 0
                    node = v
                    while node != u:
                        dist += 1
                        node = parent[node]
                    return dist
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = curr
                        queue.append(neighbor)
            return float('inf')
            
        pair_dists = [(get_dist(u, v), u, v) for u, v in pairs]
        pair_dists.sort()
        ordered_pairs = [(u, v) for d, u, v in pair_dists]
        
        paths_found = {}
        
        def backtrack(idx, used_interiors):
            if time.time() > deadline:
                return False
            if idx == len(ordered_pairs):
                return True
            
            u, v = ordered_pairs[idx]
            # Forbidden: used_interiors U (branches \ {u, v})
            forbidden = used_interiors | (branch_set - {u, v})
            
            # Find a path from u to v avoiding forbidden
            # We try the shortest path first. If it fails, we might need a longer one.
            # To keep it simple and fast, we'll just try the shortest path.
            # If this fails, we return False.
            # This is a greedy heuristic. It might miss solutions.
            # To improve, we could try all paths up to a certain length, or use a more complex search.
            # Given n=24, let's try a few alternative paths if the shortest one doesn't work?
            # Actually, just finding *a* path is hard if we need disjointness.
            # Let's stick to shortest path first.
            
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return False
            
            # Check if path interior is disjoint from used_interiors
            # find_shortest_path already ensures this, but let's be sure.
            interior = set(path[1:-1])
            if interior & used_interiors:
                return False
            if interior & (branch_set - {u, v}):
                return False
                
            # Commit
            paths_found[(u, v)] = path
            paths_found[(v, u)] = list(reversed(path))
            
            new_used = used_interiors | interior
            if backtrack(idx + 1, new_used):
                return True
            
            # Backtrack
            del paths_found[(u, v)]
            del paths_found[(v, u)]
            return False

        if backtrack(0, set()):
            return paths_found
        return None

    # Try to find larger k
    # We know best_k is achievable.
    # Try k from 10 down to best_k + 1
    # Generate candidates
    
    # Strategy:
    # 1. Try to expand the initial clique.
    # 2. Try random sets.
    
    # First, ensure we have a valid result for best_k
    # If initial_clique was a clique, the paths are edges.
    # But what if we can find a subdivision for the same set with longer paths? Doesn't matter, we just need *a* valid one.
    # The edge-based one is valid.
    
    # Now try to find k > best_k
    # We will try k from 10 down to best_k + 1
    # For each k, we try a number of candidate sets.
    
    # Candidate generation:
    # - Subsets of high-degree vertices.
    # - Random subsets.
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Try to find a valid subdivision for k
    for k in range(10, best_k, -1):
        if time.time() > deadline - 5:
            break
            
        candidates = set()
        
        # Add subsets of top degree vertices
        # We can't check all C(24, k).
        # Let's try a few specific structures.
        
        # 1. Top k degree vertices
        candidates.add(tuple(sorted(sorted_vertices[:k])))
        
        # 2. Mix of top degree and random
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Pick k vertices
            # Bias towards high degree
            cands = []
            for _ in range(k):
                # Weighted random choice
                # Simple: pick from top 10 usually
                if random.random() < 0.7:
                    cands.append(random.choice(sorted_vertices[:10]))
                else:
                    cands.append(random.randint(0, n-1))
            # Remove duplicates
            c = list(set(cands))
            if len(c) < k:
                # Fill with random
                while len(c) < k:
                    v = random.randint(0, n-1)
                    if v not in c:
                        c.append(v)
            c = c[:k]
            candidates.add(tuple(sorted(c)))
            
        # 3. Try to expand the best clique found so far
        if len(initial_clique) < k:
             # Add random vertices to the clique
             for _ in range(20):
                 if time.time() > deadline - 5:
                     break
                 c = list(initial_clique)
                 while len(c) < k:
                     v = random.randint(0, n-1)
                     if v not in c:
                         c.append(v)
                 candidates.add(tuple(sorted(c)))
                 
        # Check candidates
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            paths = check_subdivision(list(cand))
            if paths is not None:
                best_k = k
                best_branches = list(cand)
                best_paths = paths
                break # Found one for this k, move to next k (smaller)
                
    # Construct final result
    # Ensure types are correct
    final_branches = [int(b) for b in best_branches]
    final_paths_dict = {}
    for u, v in combinations(final_branches, 2):
        # Get path from best_paths
        # best_paths keys are tuples from the check_subdivision call
        # We need to map them back
        # In check_subdivision, keys are (u, v) and (v, u)
        # But we iterate over combinations of final_branches
        # The path for (u, v) should be in best_paths
        # Note: best_paths might have keys in different order?
        # check_subdivision uses combinations(branches, 2) which is sorted.
        # So keys are (min(u,v), max(u,v)) ? No, combinations returns tuples in input order.
        # If branches are sorted, combinations are sorted.
        # In check_subdivision, we did `pairs = list(combinations(branches, 2))`.
        # If `branches` is sorted, pairs are sorted.
        # So best_paths should have keys (u, v) where u < v.
        
        if (u, v) in best_paths:
             p = best_paths[(u, v)]
        elif (v, u) in best_paths:
             p = best_paths[(v, u)]
             p = list(reversed(p))
        else:
             # Fallback: direct edge?
             if v in adj[u]:
                 p = [u, v]
             else:
                 # Should not happen if valid
                 p = [u, v]
        
        # Ensure path is a list of ints
        final_paths_dict[(u, v)] = [int(x) for x in p]
        
    # The output format requires 'paths' to be a dict?
    # "paths': one simple path for each unordered pair of branches"
    # The seed program returns: 'paths':[list(e) for e in combinations(row,2)]
    # Wait, the seed program returns a LIST of paths?
    # "Return a dict with exactly the keys 'branches' and 'paths'. Each path is a list..."
    # The seed code: `return {'branches':list(row),'paths':[list(e) for e in combinations(row,2)]}`
    # This returns a list of paths, not a dict.
    # The contract says: "paths': one simple path for each unordered pair of branches"
    # Usually this implies a mapping or a list corresponding to the pairs.
    # Given the seed returns a list, I should return a list.
    # The order of the list should probably correspond to the order of combinations(branches, 2).
    
    final_paths_list = []
    for u, v in combinations(final_branches, 2):
        # We stored in best_paths with keys (u,v)
        # If best_paths is a dict, we can retrieve it.
        # But wait, in my check_subdivision, I stored in `paths_found` which is returned.
        # So best_paths is a dict.
        # I need to convert it to a list in the correct order.
        
        # My best_paths dict has keys (u,v) and (v,u).
        # I will just iterate combinations and pick the path.
        if (u, v) in best_paths:
            p = best_paths[(u, v)]
        elif (v, u) in best_paths:
            p = best_paths[(v, u)]
            p = list(reversed(p))
        else:
            p = [u, v] # Fallback
            
        final_paths_list.append([int(x) for x in p])
        
    return {
        'branches': final_branches,
        'paths': final_paths_list
    }

# EVOLVE-BLOCK-END
