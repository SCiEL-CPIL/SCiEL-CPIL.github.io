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
        
    # Identify candidate edges:
    # 1. Not in base
    # 2. Opposite parity (to help triangle-free? No, triangle-free just means no 3-cycle.
    #    In a bipartite graph, all cycles are even. A path is bipartite.
    #    If we only add edges between opposite parity nodes, the graph remains bipartite, hence triangle-free.
    #    This is a HUGE simplification. We can restrict search to bipartite augmentations.
    #    Vertices 0..23. Parity: even/odd.
    #    Base edges connect even-odd.
    #    If we add edge (u,v) where u%2 != v%2, the graph stays bipartite.
    #    If we add edge (u,v) where u%2 == v%2, we create an odd cycle (triangle or longer).
    #    Specifically, if u,v same parity, and there is a path of even length between them, we get an odd cycle.
    #    In a path graph, dist(u,v) = |u-v|. If |u-v| is even, they have same parity.
    #    Adding (u,v) with same parity creates a cycle of length |u-v|+1.
    #    If |u-v|+1 == 3, i.e., |u-v|=2, we create a triangle.
    #    So adding any edge with |u-v| >= 2 and same parity creates an odd cycle.
    #    Does it have to be a triangle? No. But the constraint is "triangle-free".
    #    Wait, the constraint is "triangle-free", not "bipartite".
    #    However, if I restrict to bipartite edges (opposite parity), I am guaranteed triangle-free.
    #    Is it possible that the optimal solution requires a same-parity edge (longer than 2) to minimize edge count?
    #    Let's check. If I add (0, 4), I create a cycle 0-1-2-3-4-0 of length 5. No triangle.
    #    So same-parity edges ARE allowed as long as they don't form a triangle.
    #    But checking triangle-freeness for arbitrary sets is harder.
    #    Strategy: First try to solve using ONLY opposite-parity edges (bipartite augmentation).
    #    If that fails or is too expensive, we can expand. But bipartite is a safe subset.
    #    Given the path structure, bipartite augmentation is very powerful.
    
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges_set:
                continue
            # Check triangle-free with base graph
            # If we add (u,v), it forms a triangle if there exists w such that (u,w) and (v,w) are in base graph.
            # In a path, w must be u-1, u+1, v-1, v+1.
            # Common neighbor in path: only if dist(u,v) == 2.
            # So if abs(u-v) == 2, it's a triangle.
            # If abs(u-v) > 2, it doesn't form a triangle with the BASE graph.
            # However, it might form a triangle with OTHER added edges.
            # To keep it simple and fast, let's first assume we can find a solution in the bipartite subgraph.
            if (u - v) % 2 == 1:
                candidates.append((u, v))
                
    # Sort candidates by "usefulness". 
    # Long edges reduce diameter more.
    # Let's prioritize edges that bridge large gaps.
    # A good heuristic for diameter reduction is to add edges that connect the "ends" of the current longest paths.
    # But for a static search, let's sort by span descending.
    candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    # We will use a backtracking search.
    # State: current set of added edges.
    # Goal: diameter <= 4.
    # Constraint: triangle-free.
    
    # To make backtracking efficient, we need a fast way to check diameter and triangle-freeness.
    # Triangle-freeness:
    # If we stick to bipartite candidates, we are automatically triangle-free.
    # Let's verify this hypothesis. If the optimal solution requires a non-bipartite edge, we might miss it.
    # But for a path of 24 nodes, bipartite augmentation is likely sufficient.
    # Let's try to find the minimum k using bipartite edges first.
    
    def get_diameter(adj_list):
        max_d = 0
        # BFS from each node
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
        return max_d

    # Pre-compute base distances to speed up checks? 
    # No, the graph changes.
    
    # Backtracking
    best_solution = None
    best_k = 100
    
    # Heuristic ordering: 
    # We want to find a small set.
    # Let's try k=3, 4, 5...
    
    # To prune, we can check if the current partial solution can possibly reach diameter 4.
    # Lower bound on diameter: 
    # If we have a set of edges E_add, the diameter is at least the maximum distance between any two nodes in the graph formed by E_add + Base.
    # This is just the diameter.
    
    # Let's use a smarter search: 
    # Instead of blind combinations, use a greedy construction with backtracking (Limited Discrepancy Search or similar).
    # Or simply, since n=24 is small, and we suspect k is small, 
    # let's try to construct a solution greedily, then verify, then backtrack if stuck.
    
    # Actually, a simpler structural insight:
    # To get diameter 4 on a path of 24 nodes, we need to "fold" the path.
    # A single edge (0, 23) makes the path a cycle. Diameter of C24 is 12.
    # We need more edges.
    # Consider adding edges (0, 6), (0, 12), (0, 18)? 
    # Or (0, 12), (6, 18), etc.
    
    # Let's stick to the bipartite restriction. It guarantees triangle-free.
    # We search for the minimum number of bipartite edges to achieve diameter <= 4.
    
    # Optimization: 
    # We can pre-filter candidates. 
    # Any edge (u,v) with |u-v| <= 2 is useless for diameter reduction (it's already in base or creates triangle).
    # Actually, |u-v|=1 is base. |u-v|=2 is triangle.
    # So candidates must have |u-v| >= 3.
    # And since they are bipartite, |u-v| is odd. So |u-v| >= 3.
    
    filtered_candidates = [c for c in candidates if abs(c[0]-c[1]) >= 3]
    
    # Sort by span descending.
    filtered_candidates.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
    
    # Search
    # We will try k from 3 upwards.
    # For each k, we try combinations.
    # To speed up, we can use a recursive function that adds edges and checks diameter.
    # If diameter <= 4, we found a solution.
    # If we run out of candidates or time, we stop.
    
    # To make it faster, we can use a "greedy + backtrack" approach.
    # Start with empty set.
    # Pick the best edge (one that reduces diameter most).
    # Add it.
    # Repeat.
    # If we hit a dead end (no edge reduces diameter enough, or we exceed k), backtrack.
    
    # But simple combinations are fine if k is small.
    # C(50, 3) = 19600. C(50, 4) = 230300.
    # With n=24, number of bipartite edges with span >= 3:
    # u,v opposite parity.
    # Total bipartite edges in K12,12 is 144.
    # Base has 23.
    # Span >= 3 removes small ones.
    # Let's count roughly.
    # It's manageable.
    
    def solve_k(k):
        if time.time() > deadline:
            return None
        
        # Use itertools.combinations
        # We need to be careful with the number of candidates.
        # If len(filtered_candidates) > 30, combinations might be too slow for k=4,5.
        # We can prune the candidate list further.
        # Edges that are "dominated" by others?
        # An edge (u,v) is dominated if there is another edge (u',v') such that the path u-v is contained in u'-v' and u',v' is bipartite?
        # Not exactly.
        
        # Let's just limit the pool to the top 20 longest edges.
        # If a solution exists with k edges, it's likely using long edges.
        pool = filtered_candidates[:20]
        
        for combo in itertools.combinations(pool, k):
            if time.time() > deadline:
                return None
            
            # Build graph
            test_adj = [set(s) for s in adj]
            for u, v in combo:
                test_adj[u].add(v)
                test_adj[v].add(u)
                
            # Check diameter
            if get_diameter(test_adj) <= 4:
                return combo
                
        return None

    # Try k = 3, 4, 5
    for k in range(3, 7):
        if time.time() > deadline:
            break
        sol = solve_k(k)
        if sol:
            best_solution = sol
            best_k = k
            break
            
    if best_solution is None:
        # Fallback: Greedy without bipartite restriction? 
        # Or just return the best found so far? 
        # If no solution found in bipartite, we might need to relax.
        # But for now, let's assume bipartite works.
        # If not, we can try a greedy approach on all valid edges.
        
        # Greedy approach:
        current_adj = [set(s) for s in adj]
        current_edges = []
        current_diam = 23
        
        # All possible edges that don't form a triangle with current graph
        # This is dynamic.
        
        while current_diam > 4 and time.time() < deadline:
            best_edge = None
            best_diam = current_diam + 1
            
            # Generate candidates dynamically
            # To avoid triangles: (u,v) is valid if no w is neighbor of both u and v in current_adj.
            # And (u,v) not in current_adj.
            
            for u in range(n):
                for v in range(u+1, n):
                    if (u,v) in base_edges_set or (v,u) in base_edges_set:
                        continue
                    if v in current_adj[u]:
                        continue
                    
                    # Check triangle
                    common = current_adj[u].intersection(current_adj[v])
                    if common:
                        continue
                        
                    # Check if this edge reduces diameter
                    # Tentatively add
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    
                    new_diam = get_diameter(current_adj)
                    
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    
                    if new_diam < best_diam:
                        best_diam = new_diam
                        best_edge = (u, v)
                        
            if best_edge is None:
                break
                
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            current_edges.append((u, v))
            current_diam = best_diam
            
        if current_diam <= 4:
            best_solution = current_edges
            best_k = len(current_edges)
        else:
            # Failed to find a solution? Return empty or best effort?
            # The task says "Minimize the number of added edges".
            # If we can't satisfy diameter 4, the result is invalid?
            # "The union ... must remain ... diameter at most four."
            # So we MUST find a valid solution.
            # If we fail, we return an invalid result? 
            # Better to return a valid solution if possible.
            # Let's hope the greedy or combinatorial search finds one.
            pass

    if best_solution:
        final_added = [[int(u), int(v)] for u, v in best_solution]
        return final_added
    else:
        # Fallback: Return a known valid solution if hardcoded?
        # Or just return empty list? No, that's invalid.
        # Let's try a very aggressive greedy that ignores triangle check initially and fixes?
        # No, just return the best we have. If none, return empty (invalid).
        # But we should find one.
        return []

# EVOLVE-BLOCK-END
