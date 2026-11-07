"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Candidate edges: must connect opposite parity to maintain bipartiteness (triangle-free)
    # and must not be in base.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges_set:
                candidates.append((u, v))
                
    # Sort candidates by "impact" - larger distance in path is likely better for reducing diameter
    candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    # Helper: Check if adding edge (u, v) keeps graph triangle-free
    # Since base is bipartite and we only add edges between opposite partitions,
    # the graph remains bipartite if and only if no odd cycle is created.
    # However, a simpler check for triangle-freeness in a bipartite context:
    # If we maintain bipartiteness, it is automatically triangle-free.
    # Does adding an edge between opposite partitions break bipartiteness?
    # No, if the graph was bipartite, adding an edge between opposite partitions
    # keeps it bipartite. Wait, if there is already a path of odd length between u and v,
    # adding the edge creates an even cycle? No.
    # Let's verify: Path P is bipartite. Partitions A (even indices) and B (odd indices).
    # Any edge (u, v) with u in A, v in B is a valid bipartite edge.
    # Adding such an edge to a bipartite graph results in a bipartite graph.
    # Therefore, ANY combination of these candidate edges results in a triangle-free graph.
    # So we don't need to check for triangles explicitly if we stick to opposite parity.
    
    # Helper: Calculate diameter of the current graph
    def get_diameter(adj):
        max_dist = 0
        for i in range(n):
            # BFS from i
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        q.append(v)
            if -1 in dist:
                return float('inf') # Disconnected
        return max_dist

    # Initial graph: just the path
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    current_edges = set()
    current_diam = get_diameter(adj)
    
    # Greedy Strategy:
    # 1. Start with path.
    # 2. While diameter > 4:
    #    - Find the pair (u, v) with maximum distance (eccentricity pair).
    #    - Try to add an edge that reduces this distance.
    #    - Actually, a better greedy: Iterate through all candidate edges.
    #    - For each candidate edge, simulate adding it.
    #    - If diameter decreases, pick the one that gives the largest decrease (or just the first one if tie).
    #    - If no single edge reduces diameter, we might be stuck?
    #    - Better: Pick the edge that minimizes the new diameter. If tie, pick one that adds "most" shortcuts.
    
    # Since n=24 is small, we can afford to check diameter for each candidate at each step.
    # Number of steps likely < 10. Candidates ~ 12*12 = 144. 144 * 10 * BFS(24) is very fast.
    
    while current_diam > 4:
        if time.time() > deadline:
            break
            
        best_edge = None
        best_diam = current_diam + 1 # We want to minimize, so start high
        
        # Optimization: We only need to consider edges that connect nodes far apart?
        # Not necessarily. A short edge might help a different pair.
        # But let's just check all remaining candidates.
        
        remaining_candidates = [e for e in candidates if e not in current_edges]
        
        # Sort candidates by potential impact?
        # Distance in original path between u and v is |u-v|.
        # Larger |u-v| edges are likely more impactful for diameter.
        # Let's sort descending by |u-v| to find good reductions faster.
        # (Already sorted globally, but let's ensure order is maintained or re-sort if needed)
        # remaining_candidates is already in the order of 'candidates' which is sorted.
        
        for u, v in remaining_candidates:
            # Simulate add
            adj[u].add(v)
            adj[v].add(u)
            
            new_diam = get_diameter(adj)
            
            # Undo
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
            # If we achieved diameter 4, we can stop looking for better edges in this iteration
            # because we can't get better than 4 (path length is at least 4? No, min diam is 1. 
            # But we are targeting <= 4. If we hit 4, we are done with the while loop condition?
            # Wait, the while condition is current_diam > 4. 
            # If we find an edge that makes it 4, we add it and the loop terminates.
            # So we can break early if new_diam <= 4.
            if new_diam <= 4:
                break
                
        if best_edge is None:
            # No single edge reduces diameter? This shouldn't happen for a path if we can add any bipartite edge.
            # Actually, if the graph is already diameter 4, loop exits.
            # If it's > 4, there must be an edge that helps.
            break
            
        # Commit the best edge
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        current_edges.add((u, v))
        current_edges.add((v, u))
        current_diam = best_diam
        
        # If we reached diameter 4, we can stop adding edges
        if current_diam <= 4:
            break
            
    # Extract added edges
    added = []
    for u, v in current_edges:
        if (u, v) not in base_edges_set and (v, u) not in base_edges_set:
             # Ensure we return each undirected edge only once, normalized
             if u < v:
                 added.append([u, v])
             elif v < u:
                 added.append([v, u])
                 
    # Remove duplicates just in case (though set logic should prevent it)
    # And ensure types are int
    final_added = []
    seen = set()
    for e in added:
        key = (e[0], e[1])
        if key not in seen:
            final_added.append([int(e[0]), int(e[1])])
            seen.add(key)
            
    return final_added

# EVOLVE-BLOCK-END
