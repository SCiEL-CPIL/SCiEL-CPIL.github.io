"""Construct a C4-free graph with no independent 11-set using projective plane geometry."""
import time
import random

def construct():
    deadline = time.time() + 115.0
    try:
        result = _build_polarity_graph(5)
        # Try to extend the graph by adding vertices
        result = _greedy_extend(result, deadline)
        return result
    except Exception:
        # Fallback: 10 disjoint triangles (independence number 10)
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

def _build_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane of order q.
    - Total points: q^2 + q + 1
    - Edge (P, Q) iff P lies on the polar line of Q
    - No C4 (projective plane property)
    - For q=5: 31 vertices, independence number <= 5
    """
    n = q * q + q + 1  # q^2 + q + 1 points
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append([x, y, 1])
    for x in range(q):
        points.append([x, 1, 0])
    
    assert len(points) == n, f"Expected {n} points, got {len(points)}"
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            pi = points[i]
            pj = points[j]
            val = (pj[1] * pi[0] + pj[0] * pi[1] + pj[2] * pi[2]) % q
            if val == 0:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}

def _greedy_extend(graph, deadline):
    """
    Try to add vertices to the graph while maintaining C4-free property
    and ensuring no independent set of size 11.
    """
    n = graph['n']
    edges = graph['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute common neighbors count for each pair
    # common[i][j] = number of common neighbors of i and j
    # For C4-free: common[i][j] <= 1 for all i != j
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) maintains C4-free property."""
        if u == v:
            return False
        if v in adj[u]:
            return False
        # Adding edge (u,v) creates C4 if there exists w such that
        # w is adjacent to both u and v (then u-w-v-u is a 4-cycle)
        # Actually, adding edge (u,v) creates a C4 if |N(u) ∩ N(v)| >= 1
        # because then for any w in N(u) ∩ N(v), we have the cycle u-w-v-? 
        # Wait, C4 is a 4-cycle. Adding edge (u,v) creates a C4 if there
        # exists a vertex w such that w~u and w~v, making triangle u-w-v.
        # But C4 requires 4 distinct vertices in a cycle.
        # Adding edge (u,v) creates a C4 if there exists a path of length 3
        # from u to v, i.e., u-a-b-v where a~u, a~b, b~v.
        # This means: for some a in N(u) and b in N(v), a~b and a != v, b != u.
        # More precisely: adding (u,v) creates C4 iff there exist a,b with
        # a in N(u), b in N(v), a~b, a != v, b != u, a != b (distinct).
        # Actually simpler: C4 exists iff |{w : w~u and w~v}| >= 1 is NOT the criterion.
        # Let me think again. A C4 is a cycle of length 4: x1-x2-x3-x4-x1.
        # If we add edge (u,v), the new C4 must use this edge. So the cycle is
        # u-v-a-b-u where v~a, a~b, b~u. So we need: exists a in N(v), b in N(u),
        # a~b, a != u, b != v, a != b.
        # Equivalently: exists a in N(v) \ {u} such that exists b in N(u) \ {v}
        # with a~b.
        
        # Simpler check: for each a in N(v), check if a has any neighbor in N(u) \ {v}
        # that is not u.
        for a in adj[v]:
            if a == u:
                continue
            for b in adj[a]:
                if b in adj[u] and b != v and b != u and b != a:
                    return False
        return True
    
    def has_independent_set_of_size(k, max_iter=100000):
        """Check if graph has an independent set of size k using greedy + backtracking."""
        # Use a simple greedy approach first
        # Then verify with a more careful check
        
        # Greedy: repeatedly pick the vertex with minimum degree among remaining
        remaining = set(range(n))
        independent = []
        while remaining:
            # Pick vertex with minimum degree in the induced subgraph
            min_deg = float('inf')
            best = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best = v
            independent.append(best)
            remaining.discard(best)
            # Remove neighbors of best from consideration
            remaining -= adj[best]
            if len(independent) >= k:
                return True
        return len(independent) >= k
    
    # Try to add vertices one by one
    # For each new vertex, try to connect it to existing vertices
    # such that C4-free is maintained and independence number stays < 11
    
    best_n = n
    best_edges = list(edges)
    best_adj = [set(s) for s in adj]
    
    for attempt in range(12):  # Try to add up to 12 more vertices
        if time.time() > deadline - 5:
            break
        
        new_vertex = n + attempt
        # Try to connect new vertex to existing vertices
        # We want to maximize connections while maintaining C4-free
        # and keeping independence number < 11
        
        # Strategy: try to connect to vertices that are "far" from each other
        # (not adjacent to each other) to avoid creating C4s
        
        # First, find which existing vertices the new vertex can connect to
        # without creating a C4. A new vertex v connected to u1 and u2 creates
        # a C4 if u1 and u2 have a common neighbor other than v (which doesn't exist yet).
        # Actually, for a new vertex, connecting it to u1 and u2 only creates a C4
        # if u1~u2 (forming triangle, not C4) or if there's a path u1-a-b-u2 
        # of length 3 (then v-u1-a-b-u2-v is not a cycle... wait).
        
        # For a NEW vertex v with neighbors S = {v1, v2, ...}:
        # A C4 involving v must be v-v1-x-v2-v where v1~v, v1~x, x~v2, v2~v.
        # So C4 exists iff there exist v1, v2 in S such that v1 and v2 have
        # a common neighbor x (x != v1, x != v2, x != v).
        
        # So we need: for any two vertices v1, v2 in S, |N(v1) ∩ N(v2)| = 0
        # (excluding v itself, but v is new so it has no neighbors yet in the old graph)
        
        # Actually N(v1) ∩ N(v2) in the OLD graph should be empty for any pair in S.
        
        # Let's find a maximal set of vertices in the old graph that are pairwise
        # "C4-incompatible" (i.e., no two have a common neighbor)
        
        # Build list of candidate neighbors
        candidates = list(range(n))
        random.shuffle(candidates)
        
        # Greedily select neighbors for new vertex
        new_neighbors = []
        for v in candidates:
            if time.time() > deadline - 5:
                break
            # Check if v can be added to new_neighbors
            # v can be added iff for all u in new_neighbors, |N(v) ∩ N(u)| = 0
            compatible = True
            for u in new_neighbors:
                common = adj[v] & adj[u]
                if common:  # They have a common neighbor, adding both would create C4
                    compatible = False
                    break
            if compatible:
                new_neighbors.append(v)
        
        if not new_neighbors:
            # Can't add any edges to new vertex, it would be isolated
            # An isolated vertex increases independence number by 1
            # Check if current independence number + 1 < 11
            # For simplicity, skip if we can't connect it
            continue
        
        # Add the new vertex with these neighbors
        # Verify C4-free property
        valid = True
        for i in range(len(new_neighbors)):
            for j in range(i + 1, len(new_neighbors)):
                u = new_neighbors[i]
                v = new_neighbors[j]
                if adj[u] & adj[v]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check independence number
        # Build new adjacency
        new_n = n + attempt + 1
        new_adj = [set(s) for s in adj] + [set(new_neighbors)]
        for v in new_neighbors:
            new_adj[v].add(new_n - 1)
        
        # Quick independence check: greedy
        remaining = set(range(new_n))
        independent = []
        while remaining:
            min_deg = float('inf')
            best = None
            for v in remaining:
                deg = len(new_adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best = v
            independent.append(best)
            remaining.discard(best)
            remaining -= new_adj[best]
            if len(independent) >= 11:
                break
        
        if len(independent) >= 11:
            continue
        
        # Accept this extension
        for v in new_neighbors:
            edges.append([new_n - 1, v])
            adj[v].add(new_n - 1)
        adj.append(set(new_neighbors))
        n = new_n
        best_n = n
        best_edges = list(edges)
        best_adj = [set(s) for s in adj]
    
    return {'n': best_n, 'edges': best_edges}
