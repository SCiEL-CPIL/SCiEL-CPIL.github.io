"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    n = P['n']
    base_edges_set = set()
    for u, v in P['edges']:
        base_edges_set.add((u, v))
        base_edges_set.add((v, u))
        
    # Initial adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Identify candidate edges
    # We restrict to bipartite edges (opposite parity) to guarantee triangle-free.
    # This is a safe subset. If this fails, we might need to expand, but for a path,
    # bipartite augmentation is usually sufficient for small diameter.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            # Only consider edges that are not in base and are bipartite
            if (u - v) % 2 == 1:
                # Also, edges with span <= 2 are useless or triangles.
                # Span 1 is base. Span 2 is triangle (if same parity, but we are opposite parity, so span 2 is impossible for bipartite? 
                # Wait. 0 and 2 have same parity. 0 and 3 have opposite parity.
                # So bipartite edges have odd span.
                # Span 1 is base. Span 3 is the smallest non-base bipartite edge.
                if abs(u - v) >= 3:
                    candidates.append((u, v))
                
    # Sort candidates by span descending to prioritize long edges
    candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)

    # Identify "critical pairs": pairs of vertices with distance > 4 in the base graph.
    # In the base path, dist(u, v) = abs(u - v).
    critical_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if abs(u - v) > 4:
                critical_pairs.append((u, v))
                
    # For each candidate edge, determine which critical pairs it "covers" (reduces distance to <= 4).
    # To do this efficiently, we can simulate adding the edge and checking distances.
    # However, doing this for all candidates and all pairs might be slow if done naively.
    # Let's pre-compute the effect of each candidate edge on the critical pairs.
    
    # Optimization: 
    # Adding an edge (u, v) creates a shortcut. The new distance between any two nodes a, b is:
    # min(dist_base(a, b), dist_base(a, u) + 1 + dist_base(v, b), dist_base(a, v) + 1 + dist_base(u, b))
    # Since we are adding multiple edges, this simple formula doesn't hold for the combined graph.
    # However, for the greedy set cover step, we want to know if adding edge e to the *current* set of edges 
    # covers pair p. This is complex.
    
    # Alternative Strategy: 
    # Since n=24 is small, we can use a simpler heuristic.
    # We will use a greedy approach that picks the edge that reduces the *maximum* distance the most, 
    # but with a twist: we explicitly check if it covers any "uncovered" critical pairs.
    
    # Actually, let's stick to the Set Cover framing but approximate the "coverage" 
    # by checking if the edge is "useful" for a pair in the context of the base graph.
    # A pair (a, b) is covered by edge (u, v) if:
    # min(abs(a-b), abs(a-u)+1+abs(v-b), abs(a-v)+1+abs(u-b)) <= 4
    # This is a necessary condition, but not sufficient if other edges are present. 
    # However, for the *first* few edges, this is a good proxy.
    
    # Let's refine: We will iteratively add edges. 
    # At each step, we compute the current distances. 
    # We identify pairs with dist > 4. 
    # We pick the edge that minimizes the number of remaining pairs with dist > 4.
    # This is a standard greedy set cover / greedy diameter reduction.
    
    # To speed up, we will limit the candidate pool to the top 30 longest edges.
    pool = candidates[:30]
    
    current_adj = [set(s) for s in adj]
    added_edges = []
    
    def get_diameter_and_critical_pairs(adj_list):
        max_d = 0
        critical = []
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_d:
                            max_d = dist[v]
                        q.append(v)
            for j in range(i + 1, n):
                if dist[j] > 4:
                    critical.append((i, j))
        return max_d, critical

    current_diam, critical = get_diameter_and_critical_pairs(current_adj)
    
    while current_diam > 4 and time.time() < deadline:
        best_edge = None
        best_crit_count = len(critical) + 1
        
        # For each candidate edge in the pool (that hasn't been added yet),
        # check how many critical pairs it would fix.
        for u, v in pool:
            if v in current_adj[u]:
                continue
            
            # Tentatively add edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            
            # Check triangle-free (should be guaranteed by bipartite restriction, but let's be safe)
            # common = current_adj[u].intersection(current_adj[v])
            # if common:
            #     current_adj[u].remove(v)
            #     current_adj[v].remove(u)
            #     continue
            
            # Calculate new critical pairs
            _, new_critical = get_diameter_and_critical_pairs(current_adj)
            
            # Undo
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
            if len(new_critical) < best_crit_count:
                best_crit_count = len(new_critical)
                best_edge = (u, v)
                
        if best_edge is None:
            # No edge improves the situation? Break to avoid infinite loop.
            break
            
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        added_edges.append((u, v))
        
        current_diam, critical = get_diameter_and_critical_pairs(current_adj)
        
        if time.time() > deadline:
            break

    if current_diam <= 4:
        final_added = [[int(u), int(v)] for u, v in added_edges]
        return final_added
    else:
        # Fallback if greedy failed to reach diameter 4
        # This might happen if the bipartite restriction is too tight or the pool is too small.
        # For now, return the best we have. If invalid, the evaluator will reject.
        # But we should try to ensure validity.
        # If we didn't reach diameter 4, the result is invalid.
        # Let's try a more exhaustive search for k=5,6 if we have time.
        # But for now, return the greedy result.
        final_added = [[int(u), int(v)] for u, v in added_edges]
        return final_added

# EVOLVE-BLOCK-END
