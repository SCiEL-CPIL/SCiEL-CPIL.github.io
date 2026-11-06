import time
import itertools

def construct():
    deadline = time.time() + 140  # safety margin
    
    n = 16
    # Represent edges as a set of frozensets or a bitmask
    # Use a list of (u,v) with u<v
    
    # Precompute all pairs of 4-sets
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # For each 4-set, store as a tuple for fast lookup
    # We'll use a 2D structure: for each pair of 4-sets, we need to check edges
    
    # Better: represent the graph as a 16x16 adjacency matrix (bitmasks per vertex)
    # adj[i] is a bitmask of vertices adjacent to i
    
    # Strategy: greedy with backtracking, or local search
    # Start with a dense graph and remove edges to eliminate Q3s
    
    # Let's try: start with complete graph, find Q3s, remove edges
    # But that's slow. Instead, use a constructive approach.
    
    # Key insight: G[A,B] (A,B 4-sets) contains Q3 iff:
    #   - |A|=|B|=4, A∩B=∅
    #   - Number of edges between A and B is at least 12
    #   - The non-edges between A and B form a matching (i.e., at most 4 non-edges, each vertex in at most 1 non-edge)
    
    # Let's use a backtracking search: add edges one by one in a smart order
    # and check if adding an edge creates a Q3.
    
    # For efficiency, when adding edge (u,v), we only need to check 4-sets A containing u (or v) 
    # and 4-sets B containing v (or u) such that u∈A, v∈B or u∈B, v∈A.
    
    # Number of 4-sets containing a specific vertex: C(15,3) = 455
    # So for each edge addition, we check at most 455*455 = 207025 pairs? No, that's too many.
    
    # Better: for edge (u,v), Q3 containing this edge requires u and v to be in opposite parts.
    # So either u∈A, v∈B or u∈B, v∈A.
    # 4-sets containing u: C(15,3)=455. 4-sets containing v: C(15,3)=455.
    # But A and B must be disjoint.
    
    # This is still expensive. Let me use a different approach.
    
    # Approach: Local search / simulated annealing
    # Start with a random or structured graph, try to add edges or swap edges
    
    # Actually, let me try a simpler constructive approach first.
    # The observed best is ~54 edges. Let me try to get there.
    
    # Construction idea: Take K16 and remove a minimum edge set to destroy all Q3s.
    # This is a hitting set problem.
    
    # Alternative: Use the fact that if we partition 16 vertices into 4 groups of 4,
    # and make the graph complete between groups but with some structure...
    
    # Let me try: 4 groups of 4 vertices. Make all inter-group edges present, 
    # but within each group, no edges. This gives 4*3*C(4,4)*4 = wait.
    # Between each pair of groups: 4*4=16 edges. There are C(4,2)=6 pairs. Total: 6*16=96 edges.
    # But does this contain Q3? A Q3 needs 4+4 vertices. If we take 4 from group A and 4 from group B,
    # all 16 edges are present, which contains Q3 (since K4,4 contains Q3). So this doesn't work.
    
    # What if within each group we have a matching, and between groups we remove some edges?
    
    # Let me try a different approach: use the complement.
    # G is Q3-free iff for every 4+4 partition A,B, the bipartite graph G[A,B] is NOT 
    # a supergraph of Q3. I.e., G[A,B] has at most 11 edges, OR if it has 12+ edges, 
    # the non-edges don't form a matching.
    
    # Equivalently, in the complement H = K16 - G, for every disjoint 4-sets A,B:
    # H[A,B] must either have at least 4 edges, OR if it has exactly 3 or fewer edges, 
    # those edges must not be a matching (i.e., some vertex is incident to 2+ of them).
    # Wait, let me re-derive.
    
    # G[A,B] contains Q3 iff G[A,B] has all 16 edges except a perfect matching (4 edges).
    # So G[A,B] has exactly 12 edges and the 4 missing edges form a perfect matching.
    # OR G[A,B] has 13,14,15,16 edges - but wait, if it has 13 edges, it still contains Q3 
    # because we can find a perfect matching among the 3 missing edges? No, 3 missing edges 
    # can't form a perfect matching of size 4. So if G[A,B] has 13+ edges, it does NOT 
    # contain Q3 as a subgraph? 
    # 
    # Wait, I need to reconsider. Q3 as a subgraph means there exist 12 specific edges.
    # If G[A,B] has 13 edges, then there are 3 missing edges. Can we find a perfect matching 
    # among the missing edges? No, we need 4 missing edges forming a matching. With only 3 
    # missing, we can't. So G[A,B] with 13+ edges does NOT contain Q3? 
    # 
    # That's wrong. If G[A,B] = K4,4 (all 16 edges), it definitely contains Q3 as a subgraph 
    # (just pick any 12 edges forming Q3). The issue is: Q3 as a subgraph means there exists 
    # an injection from Q3's vertices to A∪B such that all Q3 edges are present.
    # 
    # So G[A,B] contains Q3 iff there exists a perfect matching M in K4,4 such that 
    # all edges NOT in M are present in G[A,B]. I.e., G[A,B] ⊇ (K4,4 - M) for some perfect matching M.
    # 
    # This means: G[A,B] contains Q3 iff the complement H[A,B] = K4,4 - G[A,B] is a subgraph 
    # of some perfect matching. I.e., H[A,B] has at most 4 edges and those edges form a matching.
    # 
    # So: G[A,B] contains Q3 iff H[A,B] is a subgraph of a perfect matching in K4,4.
    # Which means: H[A,B] has 0, 1, 2, 3, or 4 edges, and if it has k edges, those k edges 
    # form a matching (no two share a vertex).
    
    # So for G to be Q3-free: for all disjoint 4-sets A,B, H[A,B] is NOT a subgraph of 
    # a perfect matching. I.e., H[A,B] either has 5+ edges, OR has 2+ edges that share a vertex, 
    # OR has exactly 1 edge (which IS a matching, so that's bad).
    # 
    # Wait, 0 edges: empty set is a subgraph of any matching. So H[A,B] empty means G[A,B]=K4,4 
    # which contains Q3. So H[A,B] must NOT be a subgraph of a perfect matching.
    # 
    # H[A,B] is a subgraph of a perfect matching iff: H[A,B] is a matching (possibly empty).
    # So G is Q3-free iff for all disjoint 4-sets A,B, H[A,B] is NOT a matching.
    # I.e., H[A,B] either has 2+ edges sharing a vertex, or has 5+ edges.
    
    # This is a constraint on H (the complement). We want to minimize edges in H (to maximize G).
    # H must be such that for every disjoint 4-sets A,B, H[A,B] is not a matching.
    
    # This is a complex constraint. Let me just do a local search on G directly.
    
    # Local search approach:
    # 1. Start with a random graph or the seed
    # 2. Try to add edges; if it creates Q3, try removing a different edge
    # 3. Or: start dense, remove edges to fix Q3s
    
    # Let me implement a function to check if G contains Q3 using the H[A,B] approach.
    
    def contains_q3(adj):
        """adj[i] is a bitmask of neighbors of vertex i. Returns True if Q3 found."""
        # For each pair of disjoint 4-sets A, B, check if H[A,B] is a matching
        # H[A,B] = edges NOT in G between A and B
        # H[A,B] is a matching iff: for all u in A, v in B, if (u,v) not in G, 
        # then no other non-edge shares u or v.
        
        # More efficiently: count non-edges and check matching property.
        # For 4-sets A and B (disjoint), there are 16 possible edges.
        # non_edges = [(u,v) for u in A, v in B if not (adj[u] & (1<<v))]
        # Check if non_edges form a matching.
        
        for A in all_4sets:
            A_set = set(A)
            for B in all_4sets:
                if A_set & set(B):
                    continue
                # Count non-edges between A and B
                non_edges = []
                for u in A:
                    for v in B:
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if len(non_edges) > 4:
                    continue
                # Check if non_edges form a matching
                used_u = set()
                used_v = set()
                is_matching = True
                for u, v in non_edges:
                    if u in used_u or v in used_v:
                        is_matching = False
                        break
                    used_u.add(u)
                    used_v.add(v)
                if is_matching:
                    return True
        return False
    
    # This is too slow for repeated calls. Let me optimize.
    
    # Better: precompute for each vertex pair whether it's in the graph, and use bitmasks.
    
    # Let me try a different strategy: constructive with smart edge ordering.
    
    # Given the time constraints, let me use a known good construction.
    # The observed best is ~54 edges. Let me try to construct one.
    
    # Idea: Use a 4x4 grid structure. Label vertices (i,j) for i,j in 0..3.
    # Connect (i,j) to (i',j) if i≠i' (complete within rows) - no, that creates Q3s.
    
    # Alternative: Take the complement of a 4-regular graph with specific properties.
    
    # Let me just do a greedy add with efficient Q3 checking.
    
    # Efficient Q3 check when adding edge (u,v):
    # Only need to check 4-sets A containing u and 4-sets B containing v (disjoint),
    # and vice versa.
    
    # C(15,3) = 455 4-sets containing a given vertex.
    # For each such A, we need B disjoint from A containing v: C(12,3) = 220.
    # So 455 * 220 ≈ 100k checks per edge addition. With 120 edges max, that's 12M checks.
    # Each check is O(1) with bitmasks. Might be feasible in 140s.
    
    # Let me implement this.
    
    def is_q3_free(adj):
        """Check if graph with adjacency bitmasks is Q3-free."""
        for A in all_4sets:
            A_set = set(A)
            for B in all_4sets:
                if A_set & set(B):
                    continue
                # Check if H[A,B] is a matching
                non_edges = 0
                # Use bitmasks for efficiency
                for u in A:
                    for v in B:
                        if not (adj[u] & (1 << v)):
                            non_edges += 1
                if non_edges > 4:
                    continue
                # Check matching property
                used_u = set()
                used_v = set()
                is_match = True
                for u in A:
                    for v in B:
                        if not (adj[u] & (1 << v)):
                            if u in used_u or v in used_v:
                                is_match = False
                                break
                            used_u.add(u)
                            used_v.add(v)
                    if not is_match:
                        break
                if is_match:
                    return False
        return True
    
    # This is still slow. Let me just go with a known construction.
    
    # Based on the observed score of 1.3846 * 39 ≈ 54, let me try to get 54+.
    
    # Construction: Start with K16, remove edges to break all Q3s.
    # Use a greedy removal: find a Q3, remove one of its edges, repeat.
    
    # But finding Q3s is the bottleneck.
    
    # Let me try a completely different approach: use the structure of Q3.
    
    # Q3 is bipartite with parts of size 4. In any Q3, each vertex has degree 3.
    # So if I make the graph such that no 8 vertices form a 3-regular bipartite subgraph...
    
    # Practical approach: Use a SAT solver? No, not available.
    
    # Let me just do a local search with a timeout.
    
    import random
    
    def count_q3s(adj):
        """Count number of Q3 subgraphs. Returns 0 if none."""
        count = 0
        for A in all_4sets:
            A_set = set(A)
            for B in all_4sets:
                if A_set & set(B):
                    continue
                # Check if H[A,B] is a matching
                non_edges = []
                for u in A:
                    for v in B:
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if len(non_edges) > 4:
                    continue
                used_u = set()
                used_v = set()
                is_match = True
                for u, v in non_edges:
                    if u in used_u or v in used_v:
                        is_match = False
                        break
                    used_u.add(u)
                    used_v.add(v)
                if is_match:
                    count += 1
        return count
    
    # Start with a dense graph and remove edges
    # Or start sparse and add edges
    
    # Let me try: start with the seed (39 edges) and try to add edges.
    
    # Seed: K3,13
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    # Try to add edges in a random order
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Edges not in current graph
    missing_edges = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
    
    random.seed(42)
    random.shuffle(missing_edges)
    
    best_adj = adj[:]
    best_count = sum(bin(a).count('1') for a in adj) // 2
    
    for i, j in missing_edges:
        if time.time() > deadline:
            break
        # Try adding edge (i,j)
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        # Check if Q3-free
        if is_q3_free(adj):
            new_count = sum(bin(a).count('1') for a in adj) // 2
            if new_count > best_count:
                best_count = new_count
                best_adj = adj[:]
        else:
            # Remove the edge
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
