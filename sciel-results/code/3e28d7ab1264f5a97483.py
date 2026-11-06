"""Rewritten program for the suite task; MILP-based construction."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    # Number of possible edges
    num_edges = n * (n - 1) // 2
    
    # Map edge index to (u, v)
    edge_map = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map.append((u, v))
            idx += 1
    
    # For each pair of vertices (u, v), we need to ensure that
    # the number of common neighbors is at most 2.
    # A vertex w is a common neighbor of u and v if both (u,w) and (v,w) are edges.
    # So for each pair (u, v) with u != v, we create a constraint:
    #   sum_{w != u, v} x_{u,w} * x_{v,w} <= 2
    # But this is a quadratic constraint. MILP requires linear constraints.
    
    # Alternative approach: use the fact that if we have variables for edges,
    # the constraint "at most 2 common neighbors" can be linearized.
    # For each pair (u, v), let C_{u,v} = {w : w != u, w != v}
    # We need: for all w1, w2, w3 in C_{u,v}, not all three of (u,w1),(v,w1), (u,w2),(v,w2), (u,w3),(v,w3) can be present.
    # This is equivalent to: for any 3 distinct vertices w1, w2, w3 in C_{u,v},
    #   x_{u,w1} + x_{v,w1} + x_{u,w2} + x_{v,w2} + x_{u,w3} + x_{v,w3} <= 4
    # Wait, that's not right. Let me think again.
    
    # Actually, the condition is: |{w : w~u and w~v}| <= 2
    # This means: it's not the case that there exist 3 distinct w1,w2,w3 all adjacent to both u and v.
    # So for each triple (w1,w2,w3) of distinct vertices not in {u,v}, we have:
    #   NOT (x_{u,w1} AND x_{v,w1} AND x_{u,w2} AND x_{v,w2} AND x_{u,w3} AND x_{v,w3})
    # This is a 6-variable clause. Linearizing this directly is complex.
    
    # Simpler linearization: for each pair (u,v) and each triple (w1,w2,w3) of common candidate neighbors:
    #   x_{u,w1} + x_{v,w1} + x_{u,w2} + x_{v,w2} + x_{u,w3} + x_{v,w3} <= 5
    # Wait, that's still not quite right. If all 6 are 1, sum is 6 > 5, so forbidden.
    # But if 5 are 1, sum is 5 <= 5, allowed. But 5 edges among 3 pairs means at least one pair has both edges, 
    # actually with 5 edges in 3 pairs (6 possible), by pigeonhole, at least 2 pairs have both edges, 
    # meaning at least 2 common neighbors... wait no.
    
    # Let me reconsider. We have 3 vertices w1,w2,w3. For each wi, we have two edges: (u,wi) and (v,wi).
    # The number of common neighbors is the number of wi such that BOTH (u,wi) and (v,wi) are present.
    # We want this count <= 2.
    
    # Let y_{wi} = x_{u,wi} * x_{v,wi} (1 if wi is common neighbor). We want sum y_{wi} <= 2.
    # Since y is product of two binary vars, we can linearize with:
    #   y_{wi} <= x_{u,wi}
    #   y_{wi} <= x_{v,wi}
    #   y_{wi} >= x_{u,wi} + x_{v,wi} - 1
    #   y_{wi} binary
    
    # But this introduces many auxiliary variables. For n=20, pairs (u,v) = 190, 
    # and for each pair, up to 18 candidates, so 190*18 = 3420 y variables, plus 190*18*3 constraints.
    # That's manageable but might be slow.
    
    # Given the 120s timeout, let's try a simpler heuristic first: start with a known dense graph
    # and try to add edges greedily, checking the K2,3 constraint.
    
    # Actually, let's use a greedy randomized approach with local search.
    
    import random
    random.seed(42)
    
    def count_common_neighbors(edges_set, u, v):
        """Count common neighbors of u and v."""
        adj_u = set()
        adj_v = set()
        for a, b in edges_set:
            if a == u:
                adj_u.add(b)
            elif b == u:
                adj_u.add(a)
            if a == v:
                adj_v.add(b)
            elif b == v:
                adj_v.add(a)
        return len(adj_u & adj_v)
    
    def is_valid(edges_set, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if (u, v) in edges_set or (v, u) in edges_set:
            return True
        # Check: for every w, the number of common neighbors of u and v after adding (u,v)
        # Actually, adding (u,v) doesn't change common neighbors of other pairs.
        # It only affects common neighbors of pairs involving u or v.
        # Wait, no. Adding edge (u,v) means u and v become adjacent. 
        # The common neighbors of u and v are vertices w such that w~u and w~v.
        # Before adding, count = c. After adding, count is still c (adding (u,v) doesn't change adjacency to other vertices).
        # But we need to check: does adding (u,v) create a K2,3?
        # A K2,3 involves two vertices with 3 common neighbors.
        # Adding (u,v) could create a K2,3 if there are 3 vertices w1,w2,w3 all adjacent to both u and v.
        # So we just need: count_common_neighbors(edges_set, u, v) <= 2.
        return count_common_neighbors(edges_set, u, v) <= 2
    
    # Start with empty graph
    edges_set = set()
    
    # Generate all possible edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # Greedily add edges
    for u, v in all_edges:
        if is_valid(edges_set, u, v):
            edges_set.add((u, v))
    
    # Local search: try to add more edges
    improved = True
    while improved:
        improved = False
        remaining = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edges_set]
        random.shuffle(remaining)
        for u, v in remaining:
            if is_valid(edges_set, u, v):
                edges_set.add((u, v))
                improved = True
    
    # Convert to list of [u, v]
    result = [[u, v] for u, v in edges_set]
    return result
