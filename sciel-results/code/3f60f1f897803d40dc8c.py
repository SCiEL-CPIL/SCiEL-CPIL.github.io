"""C6-free graph on 40 vertices maximizing edges via greedy + local search."""
import time
import random
from itertools import combinations

N = 40
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def construct():
    start = time.time()
    
    # Common-neighbor counts: cn[u][v] = number of vertices w adjacent to both u and v
    # For C6-freeness: for all non-adjacent (u,v), cn[u][v] <= 1
    # Actually C6-free means: no 6-cycle. A 6-cycle exists iff there are two
    # non-adjacent vertices u,v with at least 2 common neighbors AND those common
    # neighbors can form a path of length 3 between them avoiding u,v... 
    # 
    # Simpler characterization: A graph has a C6 iff there exist vertices u, v
    # (not necessarily non-adjacent) such that there are two internally disjoint
    # paths of length 3 between u and v. 
    #
    # Equivalent: For any two vertices u, v, if they have >= 2 common neighbors,
    # that alone doesn't create a C6 (that's a C4). A C6 requires a 6-cycle.
    #
    # Let me think more carefully. A 6-cycle is v1-v2-v3-v4-v5-v6-v1.
    # Opposite vertices: (v1,v4), (v2,v5), (v3,v6).
    # v1 and v4 have common neighbors v2, v3, v5, v6 (if all edges exist).
    # Actually v1's neighbors are v2, v6. v4's neighbors are v3, v5.
    # Common neighbors of v1 and v4: none directly.
    #
    # Better approach: check for C6 directly by looking for paths of length 3
    # between pairs of vertices that are at distance 3.
    #
    # Most practical: maintain adjacency as bitmasks, and when adding edge (u,v),
    # check if it creates a C6. An edge (u,v) creates a C6 if there exists a
    # path of length 5 from u to v (i.e., u - a - b - c - d - v) where a,b,c,d
    # are distinct and not u,v, and the edges (u,a),(a,b),(b,c),(c,d),(d,v) exist.
    #
    # That's expensive. Alternative: after building the graph, verify C6-freeness
    # by checking all 6-cycles.
    #
    # For the greedy approach: when considering adding edge (u,v), a C6 is created
    # iff there's a path of length 5 from u to v in the current graph (not using
    # edge (u,v) since it doesn't exist yet). Path of length 5: u-a-b-c-d-v.
    #
    # Let me use a different strategy: build the graph, then verify.
    # Use adjacency matrix as a list of sets or bitmasks.
    
    # Use bitmask representation for speed
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * N
    
    def has_c6():
        """Check if the current graph contains a C6."""
        # For each pair of vertices (i,j) that are NOT adjacent,
        # check if there are two internally disjoint paths of length 3.
        # Actually, let me just check all 6-cycles directly.
        # A C6: v0-v1-v2-v3-v4-v5-v0 with all distinct.
        # Fix v0=0, v1 in range(1,40), v2 in range(2,40) with v2 != v1,
        # etc. This is O(n^6) which is too slow for n=40.
        
        # Better: For each pair (i,j) with i<j, count paths of length 3 from i to j.
        # If there are >= 2 such paths that are internally vertex-disjoint, we have a C6.
        # Actually, if i and j are not adjacent and have >= 2 common neighbors, 
        # that gives a C4, not C6.
        
        # For C6: vertices i and j at distance 3. A path i-a-b-c-j.
        # Two such paths i-a-b-c-j and i-a'-b'-c'-j that are internally disjoint
        # give a C6: i-a-b-c-j-c'-b'-a'-i.
        #
        # Let me just do: for each ordered pair (i,j), find all length-3 paths.
        # If any pair has >= 2 internally disjoint length-3 paths, C6 exists.
        
        # Simpler check: for each 6-tuple... no, too slow.
        
        # Use: a C6 exists iff there exist i, j (non-adjacent) such that
        # the number of common neighbors of i and j is >= 2 AND... no that's C4.
        
        # Let me just check: for each pair (i,j), count the number of paths of
        # length exactly 3 from i to j. If >= 2, check if they can form a C6.
        # Actually, ANY two distinct paths of length 3 from i to j (where i,j 
        # non-adjacent) give a C6 if the internal vertices are all distinct.
        
        # Most efficient for n=40: 
        # For each i, for each j != i, compute paths of length 3.
        # paths3[i][j] = sum over a in N(i), b in N(a)\\{i}, c in N(b)\\{a,i}: c == j
        # = for each a in N(i), for each b in N(a) not i: if j in N(b) and j != a, j != i: count++
        
        # Let me precompute using adjacency bitmasks.
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    continue  # adjacent, skip
                # Count paths of length 3 from i to j
                count = 0
                # i - a - b - j
                for a in range(N):
                    if a == i or a == j:
                        continue
                    if not (adj[i] & (1 << a)):
                        continue
                    for b in range(N):
                        if b == i or b == j or b == a:
                            continue
                        if not (adj[a] & (1 << b)):
                            continue
                        if adj[b] & (1 << j):
                            count += 1
                            if count >= 2:
                                return True
        return False
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) creates a C6."""
        # After adding (u,v), check if C6 exists.
        # A new C6 must use edge (u,v).
        # So we need a path of length 5 from u to v in the current graph.
        # u - a - b - c - d - v where a,b,c,d distinct, not u,v.
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        result = has_c6()
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return not result
    
    # Greedy: try to add all possible edges in some order
    random.seed(42)
    
    # Order edges by some heuristic - try all pairs
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > DEADLINE - 5:
            break
        if adj[u] & (1 << v):
            continue
        if can_add_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Count edges
    edge_count = sum(bin(adj[i]).count('1') for i in range(N)) // 2
    
    # Local search: try edge swaps to add more edges
    # Try removing one edge and adding two others
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        # Try: remove edge (a,b), add edges (a,c) and (b,d) or similar
        # Simpler: try to add any edge that fits
        for (u, v) in all_edges:
            if time.time() > DEADLINE - 5:
                break
            if adj[u] & (1 << v):
                continue
            if can_add_edge(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
                break
    
    # Try edge swaps: remove (a,b), add (a,c) and (b,d)
    while time.time() < DEADLINE - 10:
        made_swap = False
        for (a, b) in all_edges:
            if time.time() > DEADLINE - 10:
                break
            if not (adj[a] & (1 << b)):
                continue
            # Try removing (a,b) and adding two edges
            for c in range(N):
                if c == a or c == b:
                    continue
                if adj[a] & (1 << c):
                    continue
                for d in range(N):
                    if d == a or d == b or d == c:
                        continue
                    if adj[b] & (1 << d):
                        continue
                    # Remove (a,b), add (a,c), (b,d)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    adj[a] |= (1 << c)
                    adj[c] |= (1 << a)
                    adj[b] |= (1 << d)
                    adj[d] |= (1 << b)
                    if not has_c6():
                        made_swap = True
                        break
                    # Undo
                    adj[a] &= ~(1 << c)
                    adj[c] &= ~(1 << a)
                    adj[b] &= ~(1 << d)
                    adj[d] &= ~(1 << b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                if made_swap:
                    break
            if made_swap:
                break
        if not made_swap:
            break
    
    # Final verification and build edge list
    if has_c6():
        # Fallback: return the 8xK5 construction
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges


# Quick test
if __name__ == "__main__":
    import time
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
    # Verify no C6
    N = 40
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check C6
    found_c6 = False
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                continue
            count = 0
            for a in range(N):
                if a == i or a == j:
                    continue
                if not (adj[i] & (1 << a)):
                    continue
                for b in range(N):
                    if b == i or b == j or b == a:
                        continue
                    if not (adj[a] & (1 << b)):
                        continue
                    if adj[b] & (1 << j):
                        count += 1
                        if count >= 2:
                            found_c6 = True
                            break
                if found_c6:
                    break
            if found_c6:
                break
        if found_c6:
            break
    print(f"C6 found: {found_c6}")
