import time
import random
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 20
    num_colours = 20
    
    # Adjacency matrix: -1 for no edge, else colour
    # We will work with a dense matrix for speed
    
    def get_violations(adj):
        """
        Returns a list of violating 4-cycles.
        adj: n x n matrix of colours or -1.
        A 4-cycle u-v-w-x is violating if edges exist and colours are not distinct.
        """
        violations = []
        # To speed up, we can iterate over pairs of opposite edges?
        # Or just iterate over 4-tuples.
        # n=20, 4-tuples is 20^4 = 160k. This is fast in C, slow in Python.
        # Let's try to optimize the Python loop or use a different check.
        
        # For each pair of disjoint edges (u,v) and (w,x) with same colour c,
        # check if they form a 4-cycle with two other edges of different colours.
        # Actually, the condition is: for every 4-cycle, colours are distinct.
        # This is equivalent to:
        # 1. No two adjacent edges in a 4-cycle have same colour.
        # 2. No two opposite edges in a 4-cycle have same colour.
        
        # Let's just find all 4-cycles and check.
        # To make it faster, we can break early if we just need to know IF valid.
        # But we need to find WHICH edge to remove.
        
        # Pre-compute neighbours
        # adj[u][v] != -1
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == -1:
                    continue
                c1 = adj[u, v]
                for w in range(n):
                    if w == u or w == v or adj[v, w] == -1:
                        continue
                    c2 = adj[v, w]
                    if c2 == c1:
                        # Adjacent edges same colour -> violation if part of 4-cycle
                        # But we need the 4th edge.
                        pass 
                    for x in range(n):
                        if x == u or x == v or x == w or adj[w, x] == -1 or adj[x, u] == -1:
                            continue
                        c3 = adj[w, x]
                        c4 = adj[x, u]
                        if c3 == c1 or c3 == c2 or c3 == c4 or c4 == c1 or c4 == c2:
                            # Violation
                            # Record the cycle and the conflicting edges
                            violations.append((u, v, w, x, c1, c2, c3, c4))
        return violations

    def is_valid(adj):
        return len(get_violations(adj)) == 0

    def count_edges(adj):
        return int(np.sum(adj != -1)) // 2

    def make_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] != -1:
                    edges.append([u, v, int(adj[u, v])])
        return edges

    # Strategy:
    # 1. Start with K20.
    # 2. Assign colours using a greedy proper edge-colouring (Vizing-like).
    # 3. While violations exist:
    #    a. Find the edge involved in the most violations.
    #    b. Remove it.
    #    c. Re-optimize colours of remaining graph? (Might be too slow)
    #       Instead, just remove and check again.
    #    d. If time runs out, return best.
    
    # To improve density, after removal, we can try to RE-ADD edges if possible.
    # But let's start with the removal strategy.
    
    # Initial Colouring: Random Proper Edge Colouring
    # We need a proper edge colouring to satisfy adjacent edge constraint.
    # If we just remove edges, we might end up with a graph that is not properly colourable?
    # No, any graph is properly colourable with max_degree colours.
    # But we have 20 colours.
    
    # Let's use a specific heuristic for initial colouring:
    # Order edges by degree sum (descending).
    # Assign smallest available colour.
    
    def greedy_colour(adj_mask):
        """
        adj_mask: boolean matrix where True means edge exists.
        Returns colour matrix.
        """
        col_mat = np.full((n, n), -1, dtype=int)
        # Get edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u, v]:
                    # Heuristic: order by sum of degrees
                    deg_u = np.sum(adj_mask[u])
                    deg_v = np.sum(adj_mask[v])
                    edges.append((deg_u + deg_v, u, v))
        
        edges.sort(reverse=True)
        
        for _, u, v in edges:
            # Find forbidden colours
            forbidden = set()
            for w in range(n):
                if w != v and col_mat[u, w] != -1:
                    forbidden.add(col_mat[u, w])
                if w != u and col_mat[w, v] != -1:
                    forbidden.add(col_mat[w, v])
            
            # Pick a colour
            # Try to pick one that minimizes conflicts?
            # For now, random from available
            available = [c for c in range(num_colours) if c not in forbidden]
            if not available:
                # Should not happen if we have enough colours and graph is sparse enough
                # Or if we are in a dense graph, this means proper colouring failed?
                # Vizing: Delta+1 colours suffice. Delta <= 19. So 20 colours suffice.
                # So available should never be empty.
                return None
            
            c = random.choice(available)
            col_mat[u, v] = c
            col_mat[v, u] = c
        
        return col_mat

    # Main Loop
    best_edges = 0
    best_adj = None
    
    # Start with K20
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # We will iterate:
    # 1. Colour
    # 2. Check violations
    # 3. Remove edge with max violations
    # 4. Repeat
    
    # To speed up, we can do multiple removals per check?
    # Or use a local search on the colouring before removing edges.
    
    # Let's try:
    # For each attempt:
    #   1. Start with K20
    #   2. Greedy colour
    #   3. While violations:
    #      - Count violations per edge
    #      - Remove edge with max count
    #      - Re-colour? (Expensive)
    #      - Instead, just remove and continue with existing colours (which remain proper)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Reset to K20
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Greedy Colouring
        adj = greedy_colour(mask)
        if adj is None:
            continue
            
        # Iterative Removal
        # We need to track which edges are present.
        # adj contains -1 for removed edges.
        
        # Optimization: Instead of re-running get_violations (O(n^4)) every time,
        # we can just do it. n=20 is small. 160k checks is ~0.1s in Python?
        # Let's test speed.
        
        for step in range(200): # Max removals
            if time.time() > deadline:
                break
                
            violations = get_violations(adj)
            if not violations:
                # Valid!
                edges_count = count_edges(adj)
                if edges_count > best_edges:
                    best_edges = edges_count
                    best_adj = adj.copy()
                break
            
            # Count violations per edge
            # Edge (u,v) is identified by sorted tuple
            viol_count = {}
            for (u, v, w, x, c1, c2, c3, c4) in violations:
                # Edges involved: (u,v), (v,w), (w,x), (x,u)
                # We want to remove one.
                # Which one?
                # The one that appears in the most violations.
                # Note: A violation is a specific 4-cycle.
                # An edge can be in multiple violations.
                
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, w), max(v, w))
                e3 = (min(w, x), max(w, x))
                e4 = (min(x, u), max(x, u))
                
                for e in [e1, e2, e3, e4]:
                    if e in viol_count:
                        viol_count[e] += 1
                    else:
                        viol_count[e] = 1
            
            # Find edge with max violations
            if not viol_count:
                break
                
            max_edge = max(viol_count, key=viol_count.get)
            
            # Remove this edge
            u, v = max_edge
            adj[u, v] = -1
            adj[v, u] = -1
            mask[u, v] = False
            mask[v, u] = False
            
            # Optional: Try to re-colour to fix other potential issues?
            # Re-colouring is expensive. Let's just remove.
            # However, removing an edge might allow re-adding other edges later?
            # For now, we just remove.
            
            # Check if we should break early if edge count drops too low
            if count_edges(adj) < best_edges:
                # We might still recover if we re-colour? No, we are removing.
                # But we can't add back.
                # So if we drop below best, we can stop this attempt.
                # Wait, we haven't saved the current best from THIS attempt yet.
                # We only save if valid.
                # So we must continue until valid or timeout.
                pass

    # If best_adj is None, we might have a sparse graph from a previous run?
    # No, we start from scratch.
    
    # Fallback: If no valid graph found (should not happen, star is valid),
    # return a star.
    if best_adj is None:
        # Star on vertex 0
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19
        
    return make_edge_list(best_adj)

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
