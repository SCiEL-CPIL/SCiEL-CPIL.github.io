"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    n = P['n']
    path_edges = set()
    for u, v in P['edges']:
        path_edges.add((u, v))
        path_edges.add((v, u))
        
    # 1. Identify "problem pairs" (distance > 4 in base path)
    # In a path 0-1-...-23, dist(i,j) = |i-j|.
    # Pairs with |i-j| > 4 are problematic for diameter <= 4.
    # However, adding edges reduces distances.
    # We need to select a set of edges E_add such that in G = P + E_add,
    # max dist(u,v) <= 4, and G is triangle-free.
    
    # Candidate edges:
    # 1. Must not be in path_edges
    # 2. Must not create a triangle with existing path edges.
    #    A triangle (u,v,w) exists if (u,v), (v,w), (u,w) are all edges.
    #    Since path edges only connect i, i+1, a triangle can only be formed
    #    if we add an edge (u,v) such that there is a path of length 2 between u and v.
    #    i.e., |u-v| == 2.
    #    So, we cannot add any edge (u, u+2).
    #    We can add edges with |u-v| >= 3.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in path_edges:
                continue
            if abs(u - v) == 2:
                continue
            candidates.append((u, v))
            
    # 2. Conflict Graph Construction
    # Two candidate edges e1=(u1,v1) and e2=(u2,v2) conflict if adding both creates a triangle.
    # A triangle is formed if they share a vertex and the distance between the other two vertices is 1 (path edge).
    # Case 1: Share vertex u1=u2. Then dist(v1, v2) must be 1.
    # Case 2: Share vertex v1=v2. Then dist(u1, u2) must be 1.
    # Case 3: No shared vertex. Can they form a triangle with a path edge?
    #   Triangle needs 3 edges. If e1 and e2 are disjoint, they contribute 2 edges.
    #   The 3rd edge must be a path edge connecting an endpoint of e1 to an endpoint of e2.
    #   Let e1=(a,b), e2=(c,d).
    #   Possible path edges: (a,c), (a,d), (b,c), (b,d).
    #   If any of these is a path edge, then {e1, e2, path_edge} forms a triangle.
    #   So e1 and e2 conflict if any endpoint of e1 is adjacent to any endpoint of e2 in the path.
    
    conflict = [set() for _ in range(len(candidates))]
    for i in range(len(candidates)):
        u1, v1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            # Check if any endpoint of i is path-adjacent to any endpoint of j
            if (u1, u2) in path_edges or (u1, v2) in path_edges or (v1, u2) in path_edges or (v1, v2) in path_edges:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # 3. Solve for Minimum Set Covering Problem using Independent Set Heuristic
    # We want to select a subset S of candidates such that:
    # 1. S is an independent set in the conflict graph (triangle-free constraint).
    # 2. The resulting graph has diameter <= 4.
    # 3. |S| is minimized.
    
    # Heuristic: Greedy selection.
    # Start with empty set.
    # Repeatedly pick the edge that covers the most "uncovered" problematic pairs, 
    # among those that don't conflict with already selected edges.
    
    # Precompute which problematic pairs are covered by each candidate edge.
    # A pair (a,b) is covered if dist_G(a,b) <= 4.
    # This is dynamic. Instead, let's use a simpler heuristic:
    # We know the "center" of the path is around 11-12.
    # Edges connecting the center to the periphery are most effective.
    
    # Let's use a Max Independent Set heuristic on the conflict graph, 
    # prioritizing edges that reduce the diameter most.
    
    # Sort candidates by "potential": edges that are long and central are better.
    # Potential score: min(dist_to_center(u), dist_to_center(v)) is small? No, we want to connect far nodes.
    # Actually, connecting a central node to a peripheral node reduces the diameter significantly.
    # Let's score by: 1/ (dist_path(u, v) + 1) ? No.
    
    # Let's try a local search approach.
    # Initialize with a good heuristic set.
    # Heuristic: Connect 11 or 12 to all nodes at distance > 4.
    
    def get_diameter(edges_set):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
            
        max_d = 0
        for i in range(n):
            # BFS from i
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        q.append(nei)
            if -1 in dist:
                return float('inf') # Disconnected
            max_d = max(max_d, max(dist))
        return max_d
    
    def is_triangle_free(edges_set):
        # Check if adding edges_set to path creates a triangle
        # We already filtered candidates, but need to check interactions between selected edges
        # Since we enforce independent set in conflict graph, it should be triangle-free.
        # But let's double check for safety.
        adj = set()
        for u, v in P['edges']:
            adj.add((u, v))
            adj.add((v, u))
        for u, v in edges_set:
            adj.add((u, v))
            adj.add((v, u))
            
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                for w in adj[v]:
                    if w <= v: continue
                    if (u, w) in adj:
                        return False
        return True

    # Initial solution: Hub at 11 and 12
    # Connect 11 to 0, 1, 2, 3, 4 (dist > 4 from 11? 11-0=11, 11-1=10, 11-2=9, 11-3=8, 11-4=7)
    # Wait, dist(11, 7) = 4. So 11 needs to connect to 0..6?
    # dist(11, 6) = 5. So 11 should connect to 0..6.
    # dist(12, 17) = 5. So 12 should connect to 17..23.
    
    init_edges = []
    for i in range(7): # 0 to 6
        init_edges.append((11, i))
    for i in range(17, 24): # 17 to 23
        init_edges.append((12, i))
        
    # Check if init_edges is valid
    if not is_triangle_free(init_edges):
        # Fallback to simpler hub
        init_edges = [(12, i) for i in range(24) if i != 12 and abs(i-12) > 2]
        # Filter out path edges and dist 2
        init_edges = [e for e in init_edges if e not in path_edges and abs(e[0]-e[1]) != 2]
        
    current_edges = set(init_edges)
    current_diam = get_diameter(current_edges)
    
    # Local Search: Try to remove edges, then add if needed?
    # Or try to swap edges.
    
    # Since we want to minimize edges, let's try to remove one edge at a time if diameter stays <= 4.
    improved = True
    while improved:
        improved = False
        for e in list(current_edges):
            new_edges = current_edges - {e}
            if get_diameter(new_edges) <= 4 and is_triangle_free(new_edges):
                current_edges = new_edges
                improved = True
                break # Restart loop with new set
                
    # Try to add edges to allow further removal? (Hill climbing might get stuck)
    # Given the small size, the hub solution is likely optimal or near optimal.
    # The current program has 10 edges.
    # Can we do better?
    # A star graph with 23 edges has diameter 2.
    # A path has 0 added edges, diameter 23.
    # To get diameter 4, we need to "shorten" the path significantly.
    
    # Let's return the result of the local search.
    # Ensure types are correct
    result = []
    for u, v in current_edges:
        result.append([int(u), int(v)])
        
    # Final safety check
    if get_diameter(current_edges) > 4 or not is_triangle_free(current_edges):
         # Return the known valid 10-edge solution
         return [
            [12, 0], [12, 2], [12, 4], [12, 6], [12, 8],
            [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
        ]
        
    return result

# EVOLVE-BLOCK-END
