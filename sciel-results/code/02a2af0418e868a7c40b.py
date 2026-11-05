"""Construct a K3,3-free graph on 20 vertices maximizing edges."""
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 100  # 100s safety margin
    
    n = 20
    
    # Build projective plane of order 5
    # Points: (x, y, z) in GF(6) with not all zero, up to scalar
    # GF(6) is not a field, so use a different construction
    # Actually, use the standard projective plane PG(2,5) which requires GF(5)
    
    # PG(2,5): 31 points, 31 lines, each line has 6 points, each point on 6 lines
    # We'll construct it and take 20 points
    
    def build_pg2_5():
        """Build projective plane of order 5 over GF(5)."""
        p = 5
        # Points: equivalence classes of (x,y,z) not all zero, under scalar multiplication
        points = []
        seen = set()
        for x in range(p):
            for y in range(p):
                for z in range(p):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize: make first nonzero coordinate = 1
                    if x != 0:
                        rep = (1, y * pow(x, p-2, p) % p, z * pow(x, p-2, p) % p)
                    elif y != 0:
                        rep = (0, 1, z * pow(y, p-2, p) % p)
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        
        # Lines: each line is a set of 6 points
        # A line is defined by ax + by + cz = 0 with (a,b,c) normalized
        lines = []
        seen_lines = set()
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    # Normalize line equation
                    if a != 0:
                        a_inv = pow(a, p-2, p)
                        a2, b2, c2 = 1, b * a_inv % p, c * a_inv % p
                    elif b != 0:
                        b_inv = pow(b, p-2, p)
                        a2, b2, c2 = a * b_inv % p, 1, c * b_inv % p
                    else:
                        a2, b2, c2 = 0, 0, 1
                    key = (a2, b2, c2)
                    if key not in seen_lines:
                        seen_lines.add(key)
                        # Find all points on this line
                        line_pts = []
                        for pt in points:
                            x, y, z = pt
                            if (a2 * x + b2 * y + c2 * z) % p == 0:
                                line_pts.append(pt)
                        lines.append(line_pts)
        
        return points, lines
    
    points, lines = build_pg2_5()
    
    # We have 31 points. Select 20 to form our graph.
    # Two points are adjacent if they lie on a common line.
    # In PG(2,5), any two points lie on exactly one line, so the full graph is K_31.
    # That's not useful. Instead, let's use a different construction.
    
    # Alternative: Use the incidence graph directly.
    # Take 20 vertices from the 31 points, and connect two points if they are on a line
    # but we need to be careful. Actually, in the full point graph, every pair is connected.
    
    # Better approach: Use a bipartite construction or random + local search.
    
    # Let me use a simpler, verified approach: start with a known K3,3-free structure
    # and do local search to add edges.
    
    def is_k33_free(adj, n):
        """Check if graph with adjacency bitmask list is K3,3-free.
        K3,3 exists if there are 3 vertices with >= 3 common neighbors,
        and those 3 common neighbors are not among the 3 vertices."""
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj[i] & adj[j] & adj[k]
                    mask = (1 << i) | (1 << j) | (1 << k)
                    valid = common & ~mask
                    if bin(valid).count('1') >= 3:
                        return False
        return True
    
    def count_edges(adj, n):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    # Start with a base K3,3-free graph
    # Use a random sparse graph and add edges greedily
    
    best_adj = [0] * n
    best_edges = 0
    
    # Try multiple random starts
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start with empty graph, add edges greedily
        adj = [0] * n
        edges = 0
        
        # Generate random edge order
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() > deadline - 2:
                break
            # Try adding edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges += 1
            # Check if still K3,3-free
            if not is_k33_free(adj, n):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges -= 1
            else:
                if edges > best_edges:
                    best_edges = edges
                    best_adj = adj[:]
        
        # Local search: try to add more edges
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 2:
                    break
                if adj[u] & (1 << v):
                    continue  # Edge already exists
                # Try adding edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges += 1
                if is_k33_free(adj, n):
                    improved = True
                    if edges > best_edges:
                        best_edges = edges
                        best_adj = adj[:]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges -= 1
    
    # Convert best_adj to edge list
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return edge_list
