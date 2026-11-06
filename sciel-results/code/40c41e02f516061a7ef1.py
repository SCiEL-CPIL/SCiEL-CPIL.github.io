import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {}
    for idx, e in enumerate(all_edges):
        edge_to_idx[e] = idx
    idx_to_edge = all_edges  # 0-indexed
    
    # Precompute: for each edge, which other edges form a 4-cycle with it
    # A 4-cycle uses 4 edges. For edge (u,v), a 4-cycle containing it needs
    # two more edges forming a path of length 2 from u to v (or v to u)
    # Actually: 4-cycle = (u,v) + (v,w) + (w,x) + (x,u) where x != v, w != u
    # So for edge (u,v), the other 3 edges are (v,w), (w,x), (x,u)
    # where w is a common neighbor of v and x, and x is a common neighbor of u and v... 
    # Simpler: enumerate all 4-cycles directly.
    
    # Precompute all 4-cycles in K16
    # A 4-cycle is determined by 4 vertices {a,b,c,d} and the pairing.
    # For 4 vertices, there are 3 ways to form a 4-cycle (3 perfect matchings of the complement)
    # Actually for 4 vertices a<b<c<d, the 4-cycles are:
    # (a,b,c,d): edges ab,bc,cd,da
    # (a,b,d,c): edges ab,bd,dc,ca  
    # (a,c,b,d): edges ac,cb,bd,da
    # But we need to avoid double-counting. Each 4-cycle on 4 vertices is unique.
    # For vertices {a,b,c,d}, there are exactly 3 distinct 4-cycles (Hamiltonian cycles up to rotation/reflection).
    
    # Let's just enumerate: for each 4-vertex subset, there are 3 4-cycles
    # 4-cycle 1: a-b-c-d-a => edges (a,b),(b,c),(c,d),(d,a)
    # 4-cycle 2: a-b-d-c-a => edges (a,b),(b,d),(d,c),(c,a)
    # 4-cycle 3: a-c-b-d-a => edges (a,c),(c,b),(b,d),(d,a)
    
    four_cycles = []  # list of tuples of 4 edge indices
    for verts in itertools.combinations(range(n), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        e1 = (edge_to_idx.get((a,b)), edge_to_idx.get((b,c)), edge_to_idx.get((c,d)), edge_to_idx.get((d,a)))
        if all(x is not None for x in e1):
            four_cycles.append(e1)
        # Cycle 2: a-b-d-c-a
        e2 = (edge_to_idx.get((a,b)), edge_to_idx.get((b,d)), edge_to_idx.get((d,c)), edge_to_idx.get((c,a)))
        if all(x is not None for x in e2):
            four_cycles.append(e2)
        # Cycle 3: a-c-b-d-a
        e3 = (edge_to_idx.get((a,c)), edge_to_idx.get((c,b)), edge_to_idx.get((b,d)), edge_to_idx.get((d,a)))
        if all(x is not None for x in e3):
            four_cycles.append(e3)
    
    # For each edge, list of 4-cycle indices that contain it
    edge_to_cycles = defaultdict(list)
    for ci, cyc in enumerate(four_cycles):
        for ei in cyc:
            edge_to_cycles[ei].append(ci)
    
    # For each pair of 4-cycles, precompute their intersection pattern
    # Two 4-cycles form a Q3 if they share exactly 2 edges that are opposite in both cycles
    # In a 4-cycle (e0,e1,e2,e3) in order, opposite pairs are (e0,e2) and (e1,e3)
    # Two 4-cycles form a Q3 if their intersection is exactly one opposite pair from each
    
    # Precompute for each pair of 4-cycles whether they form a Q3
    # This is expensive: ~2800 4-cycles, so ~4M pairs. Let's be smarter.
    
    # Actually, let's count Q3s by: for each pair of 4-cycles, check if they share
    # exactly 2 edges, and those 2 edges are opposite in both 4-cycles.
    
    # Precompute opposite pairs for each 4-cycle
    cycle_opposite_pairs = []
    for cyc in four_cycles:
        # cyc = (e0, e1, e2, e3) in cyclic order
        # Opposite pairs: (e0,e2) and (e1,e3)
        cycle_opposite_pairs.append([(cyc[0], cyc[2]), (cyc[1], cyc[3])])
    
    # For fast Q3 counting, I'll use a different approach:
    # Count Q3s by iterating over pairs of "opposite edge pairs" 
    # A Q3 has 3 pairs of opposite edges (in the cube sense).
    # Actually, in Q3, there are 3 directions, each with 4 parallel edges.
    # The 12 edges of Q3 can be partitioned into 3 sets of 4 parallel edges.
    
    # Let me use the 4-cycle approach more carefully.
    # A Q3 contains exactly 6 4-cycles (one for each pair of opposite faces).
    # Wait, Q3 has 6 faces, each a 4-cycle. So each Q3 contains 6 4-cycles.
    # But different Q3s can share 4-cycles.
    
    # For counting purposes: number of Q3s = (number of pairs of 4-cycles that form a Q3) / 6
    # because each Q3 has 6 pairs of "opposite" 4-cycles... no, each Q3 has 6 4-cycles
    # and each pair of 4-cycles in a Q3 that are "opposite faces" gives one Q3.
    # Actually, in Q3, two 4-cycles (faces) are either adjacent (share an edge) or opposite (disjoint).
    # There are 3 pairs of opposite faces in Q3. So each Q3 contributes 3 pairs of disjoint 4-cycles.
    # And each pair of disjoint 4-cycles that form a Q3 corresponds to exactly one Q3.
    
    # So: count pairs of 4-cycles that are disjoint (no shared edges) and whose union forms a Q3.
    # Two 4-cycles form a Q3 iff:
    # 1. They share no edges
    # 2. The 8 vertices are all distinct
    # 3. The 12 edges form a Q3
    
    # This is getting complex. Let me use a simpler but still fast approach:
    # For the SA, I'll use a penalty based on the number of Q3s, computed incrementally.
    # When adding/removing an edge, only 4-cycles containing that edge are affected.
    
    # Let me implement a Q3 counter that can be updated incrementally.
    
    # State: set of present edges
    # For each 4-cycle, track how many of its 4 edges are present
    # A 4-cycle is "complete" if all 4 edges are present
    
    # Q3 count = number of pairs of complete 4-cycles that form a Q3
    
    # Precompute: for each pair of 4-cycles, whether they form a Q3
    # Two 4-cycles form a Q3 if:
    # - They share no edges
    # - The 8 vertices are distinct
    # - The 12 edges form a connected graph with the right structure
    
    # Actually, two 4-cycles on 8 distinct vertices with no shared edges always form a Q3
    # if and only if the 12 edges form a 3-regular graph on 8 vertices (which is Q3).
    # Two disjoint 4-cycles on 8 vertices: the 8 vertices have degree 2 within each cycle.
    # For the union to be 3-regular, we need 4 more edges connecting the two cycles
    # such that each vertex gets degree 3. But we only have the 8 edges of the two cycles.
    # So two disjoint 4-cycles on 8 vertices have 8 edges total, each vertex degree 2.
    # That's not 3-regular, so it's not a Q3!
    
    # I made an error. Let me reconsider.
    # A Q3 has 12 edges and 8 vertices, each of degree 3.
    # A 4-cycle has 4 edges and 4 vertices, each of degree 2.
    # Two 4-cycles sharing an edge: 7 edges, 7 vertices... no.
    
    # Let me just directly count Q3s by checking all 8-vertex subsets, but optimize.
    # C(16,8) = 12870 subsets. For each, check if the induced subgraph contains a Q3.
    # This is what the original code does, but it's slow.
    
    # Alternative: use the fact that Q3 is 3-regular on 8 vertices.
    # For an 8-vertex subset, check if the induced subgraph has a 3-regular spanning subgraph
    # that is isomorphic to Q3.
    
    # For the SA approach, let me use a different penalty:
    # Penalty = number of 8-vertex subsets where the induced subgraph contains a Q3.
    # But computing this is expensive.
    
    # Let me go back to the greedy approach but with a much faster Q3 check.
    # The key insight: when adding edge (u,v), only 8-sets containing both u and v need checking.
    # C(14,6) = 3003 subsets. For each, build the induced edge set and check against Q3 patterns.
    
    # Precompute all Q3 edge patterns as bitmasks over the 28 edges of K8
    # K8 has 28 edges. Index them 0..27.
    
    # For 8 vertices v0..v7, edge (vi,vj) has index: 
    # Let's use a fixed ordering: for 0<=i<j<=7, index = i*7 + (j-1) ... no
    # Standard: for i in 0..7, for j in i+1..7: index = i*(15-i)//2 + (j-i-1)
    # Or simpler: index = i*8 + j for i<j, but that's 64 slots. Let's use a dict.
    
    # Let me precompute Q3 patterns as frozensets of edge indices in K8
    def gen_q3_patterns():
        base = []
        for i in range(8):
            for j in range(i+1, 8):
                if (i ^ j) in (1, 2, 4):
                    base.append((i, j))
        
        patterns = set()
        for perm in itertools.permutations(range(8)):
            pat = frozenset(
                (min(perm[i], perm[j]), max(perm[i], perm[j])) 
                for (i, j) in base
            )
            patterns.add(pat)
        return list(patterns)
    
    # This takes a while to compute but only once
    q3_patterns = gen_q3_patterns()
    
    # For fast checking, convert patterns to a set of frozensets
    q3_pattern_set = set(q3_patterns)
    
    def check_q3_in_subset(vertices, adj):
        """Check if induced subgraph on vertices contains a Q3."""
        # Map vertices to 0..7
        vmap = {v: i for i, v in enumerate(vertices)}
        induced = frozenset()
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    induced = induced | frozenset([(i, j)])
        # Check if any Q3 pattern is a subset of induced
        # This is slow. Better: check if induced contains any pattern.
        # Since patterns have 12 edges, we can check:
        # For each pattern, check if all 12 edges are in induced.
        # 840 patterns * 12 edges = 10080 checks per subset.
        # 3003 subsets * 10080 = 30M checks. Too slow.
        
        # Better: build induced as a bitmask over 28 edges, and precompute patterns as bitmasks.
        # 28 edges in K8. Index: for i<j, idx = i*7 + (j-1) if j>i... 
        # Let's use: idx = i*(15-i)//2 + (j-i-1) for 0<=i<j<=7
        # i=0: j=1..7, idx=0..6
        # i=1: j=2..7, idx=7..12
        # i=2: j=3..7, idx=13..17
        # i=3: j=4..7, idx=18..21
        # i=4: j=5..7, idx=22..24
        # i=5: j=6..7, idx=25..26
        # i=6: j=7, idx=27
        
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            # Count edges before row i: sum_{k=0}^{i-1} (7-k) = 7i - i(i-1)/2
            before = 7*i - i*(i-1)//2
            return before + (j - i - 1)
        
        mask = 0
        for i in range(8):
            u = vertices[i]
            for j in range(i+1, 8):
                v = vertices[j]
                if adj[u] & (1 << v):
                    mask |= (1 << edge_idx(i, j))
        
        # Precomputed pattern masks
        # We need to precompute these once
        return mask
    
    # Precompute pattern masks
    def edge_idx_global(i, j):
        if i > j:
            i, j = j, i
        before = 7*i - i*(i-1)//2
        return before + (j - i - 1)
    
    pattern_masks = []
    for pat in q3_patterns:
        mask = 0
        for (i, j) in pat:
            mask |= (1 << edge_idx_global(i, j))
        pattern_masks.append(mask)
    
    # Deduplicate pattern masks
    unique_pattern_masks = list(set(pattern_masks))
    
    def has_q3_fast(adj, n=16, new_edge=None):
        """Check if graph contains a Q3. If new_edge given, only check subsets containing it."""
        if new_edge is not None:
            u, v = new_edge
            others = [x for x in range(n) if x != u and x != v]
            for subset in itertools.combinations(others, 6):
                vertices = list(subset) + [u, v]
                mask = 0
                for i in range(8):
                    a = vertices[i]
                    for j in range(i+1, 8):
                        b = vertices[j]
                        if adj[a] & (1 << b):
                            mask |= (1 << edge_idx_global(i, j))
                for pm in unique_pattern_masks:
                    if (mask & pm) == pm:
                        return True
        else:
            for vertices in itertools.combinations(range(n), 8):
                mask = 0
                for i in range(8):
                    a = vertices[i]
                    for j in range(i+1, 8):
                        b = vertices[j]
                        if adj[a] & (1 << b):
                            mask |= (1 << edge_idx_global(i, j))
                for pm in unique_pattern_masks:
                    if (mask & pm) == pm:
                        return True
        return False
    
    # Initialize with K3,13
    adj = [0] * n
    edges = set()
    for a in range(3):
        for b in range(3, n):
            edges.add((a, b))
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    best_edges = set(edges)
    best_count = len(edges)
    
    # Candidates
    candidates = [e for e in all_edges if e not in edges]
    
    # Greedy addition
    while time.time() < deadline:
        random.shuffle(candidates)
        improved = False
        for e in candidates[:]:
            if time.time() > deadline - 5:
                break
            u, v = e
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if has_q3_fast(adj, n, e):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                edges.add(e)
                if e in candidates:
                    candidates.remove(e)
                best_count = len(edges)
                improved = True
        
        if not improved:
            break
    
    # Try some random restarts with different initial structures
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        adj2 = [0] * n
        edges2 = set()
        random.shuffle(all_edges)
        for e in all_edges[:25]:
            u, v = e
            adj2[u] |= (1 << v)
            adj2[v] |= (1 << u)
            edges2.add(e)
        
        # Check if Q3-free, if not, remove edges
        while has_q3_fast(adj2, n):
            # Remove a random edge
            if not edges2:
                break
            rem = random.choice(list(edges2))
            edges2.remove(rem)
            u, v = rem
            adj2[u] &= ~(1 << v)
            adj2[v] &= ~(1 << u)
        
        # Greedy addition
        cands2 = [e for e in all_edges if e not in edges2]
        while time.time() < deadline - 10:
            random.shuffle(cands2)
            imp = False
            for e in cands2[:]:
                if time.time() > deadline - 10:
                    break
                u, v = e
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
                if has_q3_fast(adj2, n, e):
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)
                else:
                    edges2.add(e)
                    if e in cands2:
                        cands2.remove(e)
                    imp = True
            if not imp:
                break
        
        if len(edges2) > best_count:
            best_edges = edges2
            best_count = len(edges2)
            adj = adj2
            edges = edges2
    
    return [list(e) for e in best_edges]
