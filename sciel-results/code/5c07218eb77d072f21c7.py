import time
import random
import numpy as np
from collections import deque

def _pg2_5_graph():
    """Return the point graph of PG(2,5): 31 vertices, C4-free, alpha=7."""
    p = 5
    vertices = []
    seen = set()
    for x in range(p):
        for y in range(p):
            key = (x, y, 1)
            # normalize: gcd
            g = 1
            for a in key:
                while g < a:
                    if a % g == 0:
                        break
                g = a if g < a else g
            # simpler: just use tuples with leading nonzero
            # projective: (x, y, 1) for finite points, plus (x,1,0) and (1,0,0)
            vertices.append((x, y, 1))
    # add points at infinity: (x, 1, 0) for x in 0..4, and (1, 0, 0)
    for x in range(p):
        vertices.append((x, 1, 0))
    vertices.append((1, 0, 0))
    n = len(vertices)  # should be 31
    
    # normalize each vertex so first nonzero coord is 1
    norm = []
    for v in vertices:
        for i in range(3):
            if v[i] != 0:
                if v[i] != 1:
                    inv = pow(v[i], p - 2, p)
                    v = tuple((x * inv) % p for x in v)
                break
        norm.append(v)
    
    # build edges: two points connected if they lie on same line
    # line defined by ax + by + cz = 0
    edges_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            # check if there exists a nonzero (a,b,c) such that a*xi+bi*yi+ci*zi = 0 and a*xj+b*yj+c*zj = 0
            # This is a 2x3 system. Solutions exist iff the two vectors are linearly dependent in the sense that...
            # Actually, in PG(2,q), any two distinct points determine a unique line.
            # The line through points p and q is given by the cross product p × q = 0.
            # So they're always on a line. Wait, every pair of points is on a line in PG(2,q).
            # The graph is the complement of the collinearity graph? No.
            # In PG(2,q), the "point graph" where edges connect non-collinear... no.
            # Actually, for C4-free property, we want a graph where no two vertices have 2 common neighbors.
            # The standard construction: vertices are points of PG(2,q), two points adjacent if... 
            # Actually the standard C4-free graph from PG(2,q) is the polarity graph or similar.
            
            # Let me reconsider. The knowledge says PG(2,5) is a dense C4-free core with small alpha.
            # For PG(2,q), the point graph where vertices are points and edges connect points on the same line would be complete (all pairs share a line).
            # That's not right.
            
            # The standard C4-free construction from projective planes: take the incidence graph, or 
            # more commonly, the "orthogonal polarity" graph.
            # 
            # Actually, a well-known C4-free graph: vertices are points of PG(2,q), two points adjacent iff they are orthogonal under a suitable polarity.
            # For q=5, this gives a graph with 31 vertices.
            #
            # Simpler: just use the fact that the complement of the "line graph" works.
            # 
            # Let me use a different well-known construction: the (31, 10, 1) graph from PG(2,5).
            # Actually, let me just construct it properly.
            
            # In PG(2,q), define a polarity. Two points are adjacent if one lies on the polar line of the other.
            # For q odd, a suitable polarity exists.
            
            # Let me just use a known result: PG(2,5) gives a 31-vertex C4-free graph with alpha=7.
            # I'll construct it via the standard method.
            
            # Points: (x,y,z) with x^2 + y^2 + z^2 = 0 (or similar)
            # Actually, for C4-free, use the following:
            # Vertices: all points in PG(2,5)
            # Edge (p, q) iff p·q = 0 where · is the standard dot product, treating points as vectors.
            # This is the polarity graph.
            
            a = np.array(norm[i], dtype=int)
            b = np.array(norm[j], dtype=int)
            dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) % p
            if dot == 0:
                edges_set.add((i, j))
    
    edges = list(edges_set)
    return n, edges

def _check_c4_free(adj, n):
    """Check if graph is C4-free using adjacency sets."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                continue
            # count common neighbors
            common = 0
            for k in range(n):
                if adj[i][k] and adj[j][k]:
                    common += 1
                    if common > 1:
                        return False
    return True

def _independent_number_upper_bound(adj, n):
    """Greedy upper bound on independence number."""
    # Simple greedy: repeatedly pick vertex with max degree in remaining graph
    remaining = list(range(n))
    count = 0
    while remaining:
        # find vertex with max degree in subgraph induced by remaining
        best = -1
        best_deg = -1
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and not adj[v][u])
            if deg > best_deg:
                best_deg = deg
                best = v
        # Actually for independent set, we want to find large set with no edges
        # Greedy: pick vertex, remove it and its neighbors
        # But this gives lower bound, not upper.
        # For upper bound, use LP relaxation or just return n as trivial upper
        # Let's just do a simple greedy independent set (lower bound)
        pass
    
    # Simple greedy independent set (lower bound on alpha)
    remaining = set(range(n))
    indep = []
    while remaining:
        # pick vertex with min degree in remaining
        v = min(remaining, key=lambda x: sum(1 for y in remaining if y != x and not adj[x][y]))
        indep.append(v)
        remaining.discard(v)
        # remove neighbors of v from remaining (they can't be in same independent set)
        # wait, neighbors CAN be in independent set if they're not adjacent to each other
        # For greedy independent set: pick v, add to indep, remove v and all its non-neighbors? No.
        # Standard greedy: pick v, add to indep, remove v from remaining. All remaining vertices are still candidates.
        # But to ensure independence, we should remove neighbors of v.
        # Actually no - neighbors of v are adjacent to v, so they can't be in the independent set with v.
        # Non-neighbors of v CAN be in the independent set.
        for u in remaining:
            if adj[v][u]:
                remaining.discard(u)
    return len(indep)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s budget
    
    # Start with PG(2,5) graph
    n0, edges0 = _pg2_5_graph()
    adj = np.zeros((n0, n0), dtype=bool)
    for u, v in edges0:
        adj[u][v] = True
        adj[v][u] = True
    
    # Verify C4-free
    # For speed, precompute neighbor sets
    neigh = [set() for _ in range(n0)]
    for i in range(n0):
        for j in range(n0):
            if adj[i][j]:
                neigh[i].add(j)
    
    best_n = n0
    best_adj = adj.copy()
    best_neigh = [set(s) for s in neigh]
    
    def try_add_vertex(adj_cur, neigh_cur, n_cur, deadline):
        """Try to add a new vertex to the graph."""
        if time.time() > deadline:
            return None, None, n_cur, adj_cur, neigh_cur
        
        n = n_cur + 1
        new_v = n - 1
        
        # The new vertex can connect to a subset S of existing vertices
        # such that:
        # 1. No C4 is formed
        # 2. The condition |N(u) ∩ N(v)| ≤ 1 holds for all pairs
        
        # For the new vertex, it connects to some set S.
        # For C4-free:
        # - For any u, w in S: they must not have a common neighbor (other than new_v) 
        #   Wait, new_v is a common neighbor of u and w. So we need |N(u) ∩ N(w) \ {new_v}| ≤ 0
        #   i.e., N(u) ∩ N(w) = ∅ (in the old graph) for all u, w in S.
        #   This means S must be an independent set in the "conflict graph" where u~w if they share a neighbor.
        # - For u in S, w not in S: new_v is not a common neighbor, so old condition holds.
        # - For u in S, w in S: handled above.
        # - Also need to check pairs involving new_v: |N(new_v) ∩ N(u)| ≤ 1 for all u.
        #   N(new_v) = S. N(u) includes S ∩ N(u) plus possibly new_v.
        #   |S ∩ N(u)| ≤ 1 (since new_v ∈ N(u) if u ∈ S, but we're looking at common neighbors of new_v and u)
        #   Common neighbors of new_v and u: vertices in S that are also in N(u).
        #   So |S ∩ N(u)| ≤ 1 for all u.
        
        # So S must satisfy:
        # 1. For all u, w in S, u ≠ w: N(u) ∩ N(w) = ∅ (in old graph)
        # 2. For all u in V: |S ∩ N(u)| ≤ 1
        
        # Condition 2 implies condition 1? If u, w in S, then w ∈ S ∩ N(u) (if w ∈ N(u)) and u ∈ S ∩ N(w).
        # If u and w share a neighbor x, then x is not necessarily in S.
        # Condition 1 is about common neighbors in the OLD graph.
        
        # Let's find maximum such S.
        
        # Build conflict graph: u conflicts with w if N(u) ∩ N(w) ≠ ∅
        conflict = np.zeros((n_cur, n_cur), dtype=bool)
        for i in range(n_cur):
            for j in range(i + 1, n_cur):
                if neigh_cur[i] & neigh_cur[j]:
                    conflict[i][j] = True
                    conflict[j][i] = True
        
        # Also check condition 2: for each u, |S ∩ N(u)| ≤ 1
        # This means S can contain at most one neighbor of any vertex u.
        
        # Find max S satisfying both conditions.
        # This is like an independent set in a modified graph.
        
        # Greedy approach: try to build S
        S = set()
        # Sort vertices by some heuristic
        order = sorted(range(n_cur), key=lambda v: -len(neigh_cur[v]))
        
        for v in order:
            # Check if we can add v to S
            can_add = True
            # Check condition 1: v must not share a neighbor with any in S
            for u in S:
                if conflict[v][u]:
                    can_add = False
                    break
            if can_add:
                # Check condition 2: for each u, |S ∪ {v} ∩ N(u)| ≤ 1
                for u in range(n_cur):
                    if u == v:
                        continue
                    count = sum(1 for w in S if w in neigh_cur[u])
                    if v in neigh_cur[u]:
                        count += 1
                    if count > 1:
                        can_add = False
                        break
            if can_add:
                S.add(v)
        
        if not S:
            return None, None, n_cur, adj_cur, neigh_cur
        
        # Create new graph
        adj_new = np.zeros((n, n), dtype=bool)
        for i in range(n_cur):
            for j in range(n_cur):
                adj_new[i][j] = adj_cur[i][j]
        for u in S:
            adj_new[new_v][u] = True
            adj_new[u][new_v] = True
        
        neigh_new = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj_new[i][j]:
                    neigh_new[i].add(j)
        
        return S, adj_new, n, adj_new, neigh_new
    
    # Try to extend the graph
    attempts = 0
    max_attempts = 500
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        S, adj_try, n_try, adj_try2, neigh_try = try_add_vertex(adj, neigh, best_n, deadline)
        if S is not None and n_try > best_n:
            best_n = n_try
            best_adj = adj_try2.copy()
            best_neigh = [set(s) for s in neigh_try]
            # Reset attempts to try more
            attempts = 0
            if best_n >= 43:
                break
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
