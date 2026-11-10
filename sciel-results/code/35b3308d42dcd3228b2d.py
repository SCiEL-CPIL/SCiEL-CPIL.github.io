"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Candidate vertices: top by degree
    # We want to find a large set. Let's try sizes 7 down to 3.
    # But we need to be careful with time.
    
    # Heuristic: Sort vertices by degree descending
    verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_branches = None
    best_paths = None
    
    # Try to find a valid configuration for k = 7, 6, 5...
    # We will sample subsets of high-degree vertices.
    
    def find_paths(branches):
        """
        Given a list of branch vertices, try to find internally disjoint paths
        for all pairs.
        Returns a dict mapping (u, v) to path list, or None if failed.
        """
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        
        # State for backtracking
        # used_interiors: set of vertices used in path interiors
        # assigned_paths: list of paths found so far
        # unassigned_pairs: list of pairs yet to be assigned
        
        # We will use a recursive backtracking search.
        # To make it faster, we can pre-compute shortest paths or just do BFS on the fly.
        
        # Optimization: If the graph is dense, direct edges are preferred.
        
        def backtrack(used, unassigned, paths_found):
            if not unassigned:
                return paths_found
            
            # Choose a pair to process next.
            # Heuristic: Choose the pair with the fewest available shortest paths?
            # For simplicity, pick the first one, or the one with smallest index.
            # Better: Pick a pair that is "harder". 
            # Let's just pick the first one for now, but we can randomize order if needed.
            
            idx = 0 # Pick first
            i, j = unassigned[idx]
            u, v = branches[i], branches[j]
            
            # Remove from unassigned
            unassigned = unassigned[:idx] + unassigned[idx+1:]
            
            # Find a path from u to v in the graph with 'used' vertices removed.
            # The path must be simple and not go through any other branch.
            # Actually, the path endpoints are u and v. Interiors must not be branches.
            # And interiors must not be in 'used'.
            
            # BFS to find shortest path? Or DFS to find any path?
            # Shortest path is usually good for leaving room for others.
            
            # Available nodes: all nodes except 'used' and all branches except u and v.
            forbidden = used | set(branches)
            forbidden.discard(u)
            forbidden.discard(v)
            
            # BFS from u to v
            from collections import deque
            parent = {u: None}
            q = deque([u])
            found = False
            while q:
                curr = q.popleft()
                if curr == v:
                    found = True
                    break
                for nei in adj[curr]:
                    if nei not in parent and nei not in forbidden:
                        parent[nei] = curr
                        q.append(nei)
            
            if not found:
                return None
            
            # Reconstruct path
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            # Check if path is valid (it is by construction)
            # Interiors are path[1:-1]
            interiors = set(path[1:-1])
            
            # If interiors intersect with used, this path is invalid?
            # No, 'forbidden' already excluded 'used'.
            
            # New used set
            new_used = used | interiors
            
            # Recurse
            res = backtrack(new_used, unassigned, paths_found + [path])
            if res is not None:
                return res
            
            # If failed, try next path?
            # For a simple implementation, we only try the shortest path.
            # To be more robust, we could try multiple paths.
            # Given time constraints, let's stick to shortest path first.
            # If it fails, we return None.
            
            return None
        
        return backtrack(set(), list(pairs), [])

    # Try k = 7 down to 3
    # We will test subsets of the top vertices.
    # To save time, we limit the number of subsets tested.
    
    # Get top 15 vertices
    top_verts = verts[:15]
    
    for k in range(7, 2, -1):
        if time.time() > deadline:
            break
            
        # Generate subsets of size k from top_verts
        # If k is large, combinations are many.
        # For k=7, C(15,7) = 6435.
        # For each, we run backtrack.
        
        # We can add a quick check: if the subgraph induced by branches + neighbors is not connected enough, skip?
        # Or just try.
        
        # To avoid trying all, we can randomize or just try the first few.
        # Let's try a few random subsets and the "most central" ones.
        
        # Strategy: Try the first C(15, k) subsets? Might be too slow.
        # Let's limit to 100 random subsets per k.
        
        tested = set()
        attempts = 0
        max_attempts = 200 # per k
        
        while attempts < max_attempts and time.time() < deadline:
            subset = tuple(sorted(random.sample(top_verts, k)))
            if subset in tested:
                continue
            tested.add(subset)
            attempts += 1
            
            branches = list(subset)
            paths = find_paths(branches)
            
            if paths:
                best_branches = branches
                best_paths = paths
                break # Found a valid one for this k, go to next k (smaller) or done?
                # We want max k. So if we find one for k, we can stop searching for this k
                # and move to k-1? No, we want the largest k.
                # So if we find one for k=7, we are done?
                # Yes, we iterate k from 7 down. So first success is the best.
                # But wait, maybe a different subset of size 7 works? We just need one.
                # So we can return immediately?
                # Yes, the task is to maximize number of branches.
                # So finding ANY valid set of size 7 is better than ANY valid set of size 6.
                # So we can return as soon as we find a valid set for the current k.
                
        # If no valid set found for k, continue to k-1

    if best_branches is None:
        # Fallback to seed behavior (triangle)
        # Find a triangle
        es = {tuple(e) for e in edges}
        found = False
        for row in combinations(range(n), 3):
            if all(e in es for e in combinations(row, 2)):
                best_branches = list(row)
                best_paths = [list(e) for e in combinations(row, 2)]
                found = True
                break
        if not found:
            u, v = edges[0]
            best_branches = [u, v]
            best_paths = [[u, v]]

    # Format output
    # Ensure types are correct
    out_branches = [int(b) for b in best_branches]
    out_paths = [ [int(p) for p in path] for path in best_paths ]
    
    return {'branches': out_branches, 'paths': out_paths}

# EVOLVE-BLOCK-END
