import time
import random
from itertools import combinations

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    
    Strategy: Start with 8 disjoint K4s (n=32, alpha=8). Then attempt to
    add a 33rd vertex by connecting it to enough vertices in the K4s to
    keep alpha < 9, while verifying C5-freeness. If that fails, fall back
    to n=32.
    
    For the 33rd vertex v connected to k_b vertices in K4_b:
    - alpha including v: 1 + sum over b of (4 - k_b) choices... no,
      since K4_b is a clique, at most 1 vertex from K4_b in any independent set.
      So alpha_incl_v = 1 + 8 (one from each K4, avoiding neighbors) = 9 if
      v has at most 3 neighbors in each K4. To get alpha < 9, v must have
      4 neighbors in at least one K4 (so 0 choices there), giving alpha=8.
      But if v connects to all 4 in one K4, it's like merging, and we need
      to check C5-freeness.
    
    Actually, simpler: try n=33 with v connected to all 4 vertices of K4_0
    and to 0 vertices of other K4s. Then alpha = 1 (v) + 0 (K4_0, all connected) 
    + 7 (one from each of K4_1..K4_7) = 8 < 9. Good.
    C5 check: v connects to all of K4_0. A C5 through v would need v-a-b-c-d-v
    where a in K4_0. Since a is in K4_0 (clique), a connects to all in K4_0.
    b must be adjacent to a. If b in K4_0, then c,d also likely in K4_0 or elsewhere.
    This is complex - let's just verify computationally.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin

    def verify_c5_free(adj, n):
        """Check if graph has no C5 subgraph (not necessarily induced)."""
        # For each 5-tuple of distinct vertices, check if they form a C5
        # A C5 is a cycle of length 5: v0-v1-v2-v3-v4-v0
        # We check all combinations of 5 vertices and all ways to arrange them in a cycle
        for verts in combinations(range(n), 5):
            # Check all 12 distinct cycle arrangements (5!/10 = 12, but we can be smarter)
            # A cycle v0-v1-v2-v3-v4-v0: edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0)
            # Try all permutations of 5 vertices as a cycle
            # To avoid duplicates, fix v0 = min(verts) and try permutations of remaining 4
            v0 = min(verts)
            rest = [v for v in verts if v != v0]
            for perm in permutations(rest):
                v1, v2, v3, v4 = perm
                if (v1 in adj[v0] and v2 in adj[v1] and v3 in adj[v2] 
                    and v4 in adj[v3] and v0 in adj[v4]):
                    return False
        return True

    def verify_alpha_lt_9(adj, n):
        """Check if independence number is < 9 (i.e., no independent set of size 9)."""
        for verts in combinations(range(n), 9):
            independent = True
            for i in range(9):
                for j in range(i+1, 9):
                    if verts[j] in adj[verts[i]]:
                        independent = False
                        break
                if not independent:
                    break
            if independent:
                return False
        return True

    from itertools import permutations

    # Try n=33: 8 K4s plus vertex 32 connected to all of K4_0 (vertices 0-3)
    n = 33
    adj = [set() for _ in range(n)]
    edges = []
    
    # Construct 8 disjoint K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    # Connect vertex 32 to all vertices in K4_0 (0,1,2,3)
    for i in range(4):
        adj[32].add(i)
        adj[i].add(32)
        edges.append([32, i])
    
    # Verify
    if time.time() - start_time > deadline:
        # Fall back to n=32
        n = 32
        adj = [set() for _ in range(n)]
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    c5_ok = verify_c5_free(adj, n)
    alpha_ok = verify_alpha_lt_9(adj, n)
    
    if c5_ok and alpha_ok:
        return {'n': n, 'edges': edges}
    
    # If n=33 fails, try other connections for vertex 32
    # Try connecting to 4 vertices in K4_0 and 1 vertex in K4_1
    adj = [set() for _ in range(n)]
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    for i in range(4):
        adj[32].add(i)
        adj[i].add(32)
        edges.append([32, i])
    adj[32].add(4)
    adj[4].add(32)
    edges.append([32, 4])
    
    if time.time() - start_time > deadline:
        pass
    else:
        c5_ok = verify_c5_free(adj, n)
        alpha_ok = verify_alpha_lt_9(adj, n)
        if c5_ok and alpha_ok:
            return {'n': n, 'edges': edges}
    
    # Fall back to n=32
    n = 32
    adj = [set() for _ in range(n)]
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
