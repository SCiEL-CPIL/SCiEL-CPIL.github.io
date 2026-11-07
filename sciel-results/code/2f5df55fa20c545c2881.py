"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency list as sets for O(1) lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def check_time():
        return time.time() < deadline

    def find_shortest_path(u, v, forbidden):
        """
        BFS to find a simple path from u to v avoiding vertices in 'forbidden'.
        Returns list of vertices or None.
        """
        if u in forbidden or v in forbidden:
            return None
        if u == v:
            return [u, v]
        
        visited = {u: None}
        queue = deque([u])
        
        while queue:
            node = queue.popleft()
            if node == v:
                path = [v]
                while path[-1] != u:
                    path.append(visited[path[-1]])
                path.reverse()
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited[neighbor] = node
                    queue.append(neighbor)
        return None

    def verify_solution(branches, paths):
        """
        Strictly verify if the given branches and paths form a valid clique subdivision.
        Returns True if valid, False otherwise.
        """
        if len(branches) < 2:
            return False
        if len(set(branches)) != len(branches):
            return False
            
        used_interiors = set()
        
        # Check all pairs
        for u, v in combinations(sorted(branches), 2):
            # Find path for this pair
            path = None
            for key, p in paths.items():
                if (key[0] == u and key[1] == v) or (key[0] == v and key[1] == u):
                    path = p
                    break
            
            if path is None:
                return False
            
            # Path length check
            if len(path) < 2:
                return False
            
            # Endpoints check
            if path[0] not in (u, v) or path[-1] not in (u, v):
                return False
            if path[0] == path[-1]:
                return False
                
            # Simple path check (no repeated vertices)
            if len(set(path)) != len(path):
                return False
                
            # Adjacency check
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            
            # Interiors check
            interiors = path[1:-1]
            for node in interiors:
                if node in branches:
                    return False
                if node in used_interiors:
                    return False
            used_interiors.update(interiors)
            
        return True

    # Global best solution found
    best_branches = []
    best_paths = {}

    # Strategy: Systematic Backtracking with Branch and Bound
    # We try to build a set of branches. At each step, we pick a new branch candidate
    # and try to find disjoint paths for all pairs.
    
    # To make this efficient, we pre-sort candidates by degree (high degree = more likely to have paths)
    # But we also need to be careful about the order.
    
    # Let's use a recursive search that tries to add vertices one by one.
    # State: current_branches, current_paths, used_interiors
    
    def backtrack_search(current_branches, current_paths, used_interiors, candidates):
        nonlocal best_branches, best_paths
        
        if not check_time():
            return
            
        # Update best if current is better
        if len(current_branches) > len(best_branches):
            # Deep copy to avoid mutation issues
            best_branches = list(current_branches)
            best_paths = {k: list(v) for k, v in current_paths.items()}
            
        # Pruning: If we can't possibly beat the best, return?
        # Max possible = len(current_branches) + len(candidates)
        if len(current_branches) + len(candidates) <= len(best_branches):
            return
            
        if not candidates:
            return

        # Heuristic: Try candidates that are "close" to existing branches or have high degree
        # We'll just iterate through candidates in a specific order.
        # To avoid symmetry, we can enforce an ordering on branches (e.g., sorted indices)
        # But for now, let's just pick the next candidate from the list.
        
        # To improve efficiency, we can try to add a branch that is adjacent to many existing branches?
        # Or just iterate through candidates.
        
        # We remove the candidate from the list for the next recursion level
        for i, cand in enumerate(candidates):
            if not check_time():
                return
            
            new_branches = current_branches + [cand]
            new_paths = dict(current_paths)
            new_used_interiors = set(used_interiors)
            
            success = True
            # We need to find paths from 'cand' to all existing branches in 'current_branches'
            # that are disjoint from each other and from existing paths.
            
            # This is a sub-problem: find disjoint paths from cand to each b in current_branches
            # avoiding new_used_interiors and other branches.
            
            # Let's try to find these paths one by one.
            # Order matters. Let's try branches in current_branches.
            # To increase success rate, maybe try branches that are "far" first? Or "close"?
            # Close might use fewer vertices, leaving more space. Let's try close (shortest path) first.
            
            # We need to find paths P_cand_b for each b in current_branches.
            # These paths must be disjoint from each other and from new_used_interiors.
            
            # Since we are adding one branch at a time, the new paths only interact with each other 
            # (disjointness among themselves) and the old used_interiors.
            
            # Let's attempt to find these paths.
            # We can use a simple greedy or backtracking for this small sub-problem.
            # Given the small size, a randomized greedy might work well enough if we restart.
            
            # Let's try a few random orderings of the existing branches to connect to.
            # Actually, let's just try to find paths to all existing branches.
            
            # To be robust, we'll try a few times with different random seeds/orders if it fails.
            # But for speed, let's just try one deterministic pass first: shortest path avoiding used_interiors.
            
            # If we just use BFS avoiding used_interiors, the new paths might intersect each other.
            # So we need to find them sequentially, updating a local forbidden set.
            
            local_forbidden = set(new_used_interiors)
            # Also forbid other branches
            for b in new_branches:
                local_forbidden.add(b)
            
            # We need to remove cand from forbidden when finding path TO cand? No, endpoints are allowed.
            # The function find_shortest_path allows endpoints.
            
            # Let's try to find paths to all b in current_branches.
            # We'll store the found paths in a temp dict.
            temp_new_paths = {}
            
            # To handle disjointness among the new paths, we must find them one by one,
            # adding their interiors to local_forbidden.
            
            # Order of connection:
            # Let's try to connect to branches that are "easy" first?
            # Or just iterate.
            # We'll iterate through current_branches.
            
            # To improve success, we can shuffle current_branches if the first attempt fails?
            # For now, let's just try in the order they were added.
            
            for b in current_branches:
                # Find path from cand to b
                # Forbidden: local_forbidden (which includes other branches and used interiors)
                # Note: cand and b are allowed as endpoints.
                
                # We need to be careful: local_forbidden contains cand and b.
                # find_shortest_path handles endpoints correctly (checks if u/v in forbidden, but we should allow them).
                # My current find_shortest_path rejects if u or v in forbidden.
                # I need to modify it or adjust forbidden set.
                
                # Let's create a specific forbidden set for this path
                path_forbidden = set(local_forbidden)
                path_forbidden.discard(cand)
                path_forbidden.discard(b)
                
                p = find_shortest_path(cand, b, path_forbidden)
                
                if p is None:
                    success = False
                    break
                
                # Check if this path intersects with previously found new paths?
                # We added their interiors to local_forbidden? No, we haven't updated local_forbidden yet.
                # We need to check disjointness.
                
                # Let's check if p[1:-1] intersects with any existing temp_new_paths interiors.
                # And with new_used_interiors (already in local_forbidden).
                
                # Since we update local_forbidden after finding each path, subsequent BFS calls will avoid them.
                # So we just need to add p's interiors to local_forbidden.
                
                temp_new_paths[(cand, b)] = p
                local_forbidden.update(p[1:-1])
            
            if success:
                # All paths found and are disjoint (by construction of sequential BFS with updating forbidden set)
                # We can commit this branch
                new_paths.update(temp_new_paths)
                new_used_interiors.update(p[1:-1] for p in temp_new_paths.values())
                
                # Recurse
                remaining_candidates = candidates[i+1:]
                backtrack_search(new_branches, new_paths, new_used_interiors, remaining_candidates)
            
            # Backtrack: no need to explicitly undo local_forbidden as it's local to this scope
            # But we should try other candidates.

    # Initial setup for backtracking
    # We need a good starting set of candidates.
    # Let's sort all vertices by degree descending.
    sorted_vertices = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # We can start with an empty set and let it grow.
    # But finding paths for the first 2 branches is trivial (just an edge).
    # The search space is huge if we start from 0.
    
    # Better: Start with a few high-degree vertices that are connected.
    # Or just start with the top 2 vertices.
    
    # Let's try starting with a pair of connected high-degree vertices.
    # Then backtrack to add more.
    
    # Actually, the backtracking function above assumes we have a current set.
    # Let's initialize with a small valid set.
    
    # Find a valid edge to start
    start_u, start_v = -1, -1
    for u, v in edges:
        if len(adj[u]) + len(adj[v]) > (len(adj[start_u]) + len(adj[start_v]) if start_u != -1 else -1):
            start_u, start_v = u, v
            
    if start_u != -1:
        initial_branches = [start_u, start_v]
        initial_paths = {(start_u, start_v): [start_u, start_v]}
        initial_used = set()
        
        # Candidates: all other vertices
        candidates = [v for v in sorted_vertices if v not in initial_branches]
        
        backtrack_search(initial_branches, initial_paths, initial_used, candidates)
        
        # If backtracking didn't find anything better (e.g. if it timed out or failed), 
        # we might have a 2-branch solution.
        
        # If we didn't find a 2-branch solution (unlikely), fallback
        if len(best_branches) < 2:
             best_branches = [start_u, start_v]
             best_paths = {(start_u, start_v): [start_u, start_v]}
    else:
        # No edges?
        best_branches = [0, 1] if n >= 2 else [0]
        best_paths = {(0, 1): [0, 1]} if n >= 2 else {}

    # If the backtracking search was cut short or didn't explore enough, 
    # we can try a few random restarts with different initial pairs or candidate orders.
    
    # Let's add a few random restarts to improve the chance of finding a larger clique.
    for restart in range(10):
        if not check_time():
            break
            
        # Pick a random edge to start
        u, v = random.choice(edges)
        initial_branches = [u, v]
        initial_paths = {(u, v): [u, v]}
        initial_used = set()
        
        # Shuffle candidates
        candidates = [x for x in range(n) if x not in initial_branches]
        random.shuffle(candidates)
        
        # Use a slightly modified backtrack that might be faster? 
        # Or just reuse the same one.
        backtrack_search(initial_branches, initial_paths, initial_used, candidates)

    # Final verification and formatting
    if not verify_solution(best_branches, best_paths):
        # Fallback to a simple valid solution
        if len(best_branches) >= 2:
             # Try to repair? No, just pick the first valid edge found.
             u, v = edges[0]
             best_branches = [u, v]
             best_paths = {(u, v): [u, v]}
        else:
             u, v = edges[0]
             best_branches = [u, v]
             best_paths = {(u, v): [u, v]}
             
    # Convert to required format
    final_branches = [int(b) for b in best_branches]
    final_paths = []
    for u, v in combinations(sorted(best_branches), 2):
        # Find the path in best_paths
        p = None
        for key, path in best_paths.items():
            if (key[0] == u and key[1] == v) or (key[0] == v and key[1] == u):
                p = path
                break
        if p:
            final_paths.append([int(x) for x in p])
            
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
