import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    
    # Field F4 implementation
    # 0, 1, 2, 3
    # 2 is alpha, 3 is alpha+1
    # alpha^2 = alpha + 1 = 3
    # alpha^3 = 1
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def add(a, b):
        return (a + b) % 4 # Since char 4? No, char 2.
        # In F4, 1+1=0.
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=0, 2+3=1
        # 3+3=0
        # This is XOR on the bits if we map 0->00, 1->01, 2->10, 3->11?
        # 0(00)+1(01)=1(01). 1(01)+1(01)=0(00).
        # Yes, addition is XOR.
        return a ^ b

    def dot(u, v):
        return add(add(mul_table[u[0]][v[0]], mul_table[u[1]][v[1]]), mul_table[u[2]][v[2]])

    # Generate points of PG(2,4)
    points = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                vec = [a, b, c]
                if vec == [0, 0, 0]:
                    continue
                # Normalize: find first non-zero, divide by it (multiply by inverse)
                # In F4, non-zero elements are 1, 2, 3.
                # 1^-1 = 1
                # 2^-1 = 2 (since 2*2=3!=1? Wait. 2*2=3. 2*3=1. So 2^-1=3)
                # 3^-1 = 2 (since 3*2=1)
                first_nz = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nz = i
                        break
                if first_nz == -1:
                    continue
                inv = 1 if vec[first_nz] == 1 else (2 if vec[first_nz] == 3 else 3)
                norm_vec = [mul_table[vec[i]][inv] for i in range(3)]
                if norm_vec not in points:
                    points.append(norm_vec)
    
    # points list has 21 elements
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Construct adjacency matrix
    n = 21
    adj = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = True
                adj[j][i] = True
    
    # Remove one vertex to get 20.
    # Remove vertex 0.
    # New indices 0..19 correspond to old 1..20.
    # Let's just create a list of active vertices.
    active = list(range(1, 21))
    
    # Build edge list for active vertices
    edges = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if adj[active[i]][active[j]]:
                # Map back to 0..19
                u = i
                v = j
                edges.append([u, v])
    
    # Now, try to add edges.
    # Current graph on 20 vertices.
    # Check for K3,3 violations? The polarity graph is K3,3-free.
    # So we can try to add edges.
    
    # Helper: check if adding edge (u,v) creates K3,3
    # This is expensive if done naively.
    # But n=20 is small.
    
    # Build adjacency set for fast lookup
    adj_set = [set() for _ in range(20)]
    for u, v in edges:
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    def is_k33_free_with_addition(u, v):
        # If we add (u,v), check if there exist a,b,c in N(u) and x,y,z in N(v)
        # such that {a,b,c} and {x,y,z} are disjoint and form K3,3?
        # Actually, a K3,3 involves 3 vertices from one side and 3 from the other.
        # If we add (u,v), a new K3,3 must involve this edge.
        # So u and v are in the bipartition.
        # Let U_side contain u, and V_side contain v.
        # We need 2 other neighbors of u (call them u1, u2) and 2 other neighbors of v (v1, v2)
        # such that u1, u2 are connected to v1, v2.
        # AND u1, u2 are NOT connected to v? No, they can be.
        # The condition for K3,3 is that the 3x3 subgraph is complete bipartite.
        # So u, u1, u2 must be connected to v, v1, v2.
        # We already have u-v.
        # We need u1, u2 connected to v, v1, v2.
        # And u connected to v1, v2.
        
        # So:
        # 1. Pick u1, u2 from N(u).
        # 2. Pick v1, v2 from N(v).
        # 3. Check if u1, u2 are neighbors of v.
        # 4. Check if u is neighbor of v1, v2.
        # 5. Check if u1, u2 are neighbors of v1, v2.
        
        # This is O(deg(u)^2 * deg(v)^2).
        # deg is around 5. 25*25 = 625 checks. Very fast.
        
        N_u = list(adj_set[u])
        N_v = list(adj_set[v])
        
        # Optimization: u1, u2 must be in N(v) as well?
        # Yes, for the K3,3, u1 must be connected to v.
        # So u1, u2 must be in N(u) AND N(v).
        # Similarly, v1, v2 must be in N(u) AND N(v).
        
        common = [x for x in N_u if x in adj_set[v]]
        # Wait, v1, v2 are neighbors of v. They must be connected to u.
        # So v1, v2 must be in N(v) AND N(u).
        # So common neighbors of u and v are the candidates for the other 2 vertices on both sides.
        
        common_neighbors = [x for x in N_u if x in adj_set[v]]
        
        # We need to pick 2 from common_neighbors for the U side (u1, u2)
        # and 2 from common_neighbors for the V side (v1, v2)?
        # No.
        # The bipartition is {u, u1, u2} and {v, v1, v2}.
        # u1, u2 must be neighbors of v.
        # v1, v2 must be neighbors of u.
        # So u1, u2, v1, v2 are all common neighbors of u and v.
        
        if len(common_neighbors) < 4:
            return True # Cannot form K3,3 if fewer than 4 common neighbors?
            # Wait, u1, u2 are distinct from v1, v2?
            # Yes, vertices in K3,3 are distinct.
            # So we need 4 distinct common neighbors.
            # If we have < 4, we can't form K3,3.
        
        # If we have >= 4 common neighbors, check if any 4 form the structure.
        # We need u1, u2, v1, v2 such that:
        # u1, u2 connected to v1, v2.
        # (u1, v1), (u1, v2), (u2, v1), (u2, v2) must be edges.
        
        cn = common_neighbors
        for i in range(len(cn)):
            for j in range(i + 1, len(cn)):
                u1 = cn[i]
                u2 = cn[j]
                # Now pick v1, v2 from the remaining common neighbors?
                # Or from all common neighbors?
                # v1, v2 must be distinct from u1, u2.
                remaining = [x for x in cn if x != u1 and x != u2]
                for k in range(len(remaining)):
                    for l in range(k + 1, len(remaining)):
                        v1 = remaining[k]
                        v2 = remaining[l]
                        # Check edges
                        if (u1 in adj_set[v1] and u1 in adj_set[v2] and 
                            u2 in adj_set[v1] and u2 in adj_set[v2]):
                            return False # Found K3,3
        
        return True

    # Try to add edges
    improved = True
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        # Iterate over all non-edges
        for u in range(20):
            for v in range(u + 1, 20):
                if v not in adj_set[u]:
                    if is_k33_free_with_addition(u, v):
                        # Add edge
                        edges.append([u, v])
                        adj_set[u].add(v)
                        adj_set[v].add(u)
                        improved = True
    
    return edges

# EVOLVE-BLOCK-END
