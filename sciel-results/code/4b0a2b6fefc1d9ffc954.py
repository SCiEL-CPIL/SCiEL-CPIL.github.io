import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    # Initialize graph
    # Adjacency list representation for fast neighbor lookups
    adj = [set() for _ in range(n)]
    initial_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        initial_edges.add((u, v) if u < v else (v, u))
    
    added_edges = []
    
    def get_diameter():
        # BFS from each node to find max distance
        max_diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            # Graph is connected, so all dist should be >= 0
            max_diam = max(max_diam, max(dist))
        return max_diam

    def is_valid_edge(u, v):
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in initial_edges or (u, v) in set(map(tuple, added_edges)):
            return False
        # Check triangle-free: no common neighbor
        # Common neighbor w exists if w in adj[u] and w in adj[v]
        # Since n is small, simple intersection is fine
        if adj[u] & adj[v]:
            return False
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        added_edges.append([u, v])

    # Greedy loop
    deadline = time.time() + 45.0 # Safety margin
    
    while time.time() < deadline:
        current_diam = get_diameter()
        if current_diam <= 4:
            break
        
        # Find the pair (u, v) with maximum distance
        # We can reuse the BFS logic or just compute it again
        # To save time, we can find the center and leaves, but n=24 is small.
        # Let's find the actual pair to prioritize edges that shorten this specific long path.
        max_pair = (0, 0)
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for end in range(n):
                if dist[end] > max_dist:
                    max_dist = dist[end]
                    max_pair = (start, end)
        
        u_far, v_far = max_pair
        
        best_edge = None
        best_new_diam = current_diam + 1
        best_dist_reduction = 0
        
        # Iterate all possible edges
        for u in range(n):
            for v in range(u + 1, n):
                if not is_valid_edge(u, v):
                    continue
                
                # Simulate adding (u, v)
                # We don't want to modify the real graph yet
                # We can compute the new distance between u_far and v_far
                # And check if it improves the diameter.
                
                # Optimization: Only consider edges that might help.
                # An edge (u,v) helps if it creates a shorter path for some pair.
                # Specifically, if it connects the two halves of the longest path.
                
                # Let's compute the new distance between u_far and v_far with the new edge.
                # New distance is min(old_dist, 1 + dist(u_far, u) + dist(v, v_far) ... wait)
                # Actually, adding edge (u,v) allows path: ...->u->v->...
                # So new_dist(u_far, v_far) = min(old_dist, dist(u_far, u) + 1 + dist(v, v_far), dist(u_far, v) + 1 + dist(u, v_far))
                
                # To be safe and simple, since n is small, we can just:
                # 1. Temporarily add edge
                # 2. Compute new diameter
                # 3. Remove edge
                # 4. Compare
                
                # But that's 24*24/2 * (24 BFS) = 276 * 24 * ~24 ops = ~160k ops per iteration.
                # Number of iterations likely < 10. Total < 1.6M ops. Very fast.
                
                # Add temp edge
                adj[u].add(v)
                adj[v].add(u)
                
                new_diam = get_diameter()
                
                # Remove temp edge
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
                    # Also track reduction in the specific far pair distance?
                    # The diameter reduction is the primary goal.
                
                # Early break if we found an edge that makes diameter 4?
                # If new_diam <= 4, it's the best possible.
                if new_diam <= 4:
                    best_edge = (u, v)
                    best_new_diam = new_diam
                    # We can't get better than 4 (target), but we want min edges.
                    # Any edge achieving 4 is equally good for the next step?
                    # Actually, we just want to reach 4.
                    # We can break out of the inner loop if we find a valid edge that achieves the target diameter?
                    # Not necessarily the "best" in terms of future steps, but for a greedy approach,
                    # taking any edge that achieves the target is usually fine.
                    # However, to be safe against getting stuck later (though connectivity is preserved),
                    # we might want the one that reduces diameter the most, or if multiple achieve 4,
                    # maybe the one that reduces the "center" distance?
                    # Given the small size, let's just pick the first one that achieves <= 4 if we find it?
                    # Or continue to find the one with min new_diam (which is 4) and maybe tie-break by distance to center?
                    # For now, let's just keep the first one that achieves the min new_diam found so far.
                    # Since we are iterating, if we find one with 4, best_new_diam becomes 4.
                    # Other edges with 4 won't replace it unless we add a tie-breaker.
                    # Let's add a tie-breaker: prefer edges that reduce the distance between u_far and v_far the most.
                    
                # Check if this edge reduces the specific far pair distance more than the current best?
                # We didn't store the far pair distance for the best edge.
                # Let's refine: store (new_diam, dist(u_far, v_far) with new edge)
                
        if best_edge is None:
            # No valid edge found? This shouldn't happen if graph is connected and not complete.
            # But if it does, we break to avoid infinite loop.
            break
            
        add_edge(best_edge[0], best_edge[1])
        
    # Ensure we return a list of lists of ints
    result = []
    for e in added_edges:
        result.append([int(e[0]), int(e[1])])
        
    return result
# EVOLVE-BLOCK-END
