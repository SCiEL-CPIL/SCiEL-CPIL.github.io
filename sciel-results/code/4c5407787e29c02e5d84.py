import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Precompute Q3 subgraph isomorphism check using bitmasks
    # Q3 has 8 vertices and 12 edges
    # Q3 vertices: 0..7, edge between i,j if Hamming distance = 1
    q3_edges = set()
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.add((i, j))
    
    # All 8-vertex subsets of 16 vertices
    all_8sets = list(combinations(range(16), 8))
    
    # For each 8-set, precompute all 8! = 40320 permutations is too many
    # Instead, use a faster approach: for each 8-set, check if the induced subgraph
    # contains Q3 as a subgraph using backtracking with strong pruning
    
    # Better approach: represent graph as a 16x16 adjacency matrix (bitmasks)
    # For speed, use integer bitmasks for adjacency
    
    def has_q3(adj_mask):
        """Check if graph with adjacency bitmask contains Q3 as subgraph.
        adj_mask[i] is a 16-bit integer with bit j set if edge (i,j) exists.
        """
        # For each 8-vertex subset, check if it contains Q3
        # Use backtracking with the structure of Q3
        
        # Q3 adjacency (as bitmask for vertices 0..7)
        q3_adj = [0] * 8
        for a, b in q3_edges:
            q3_adj[a] |= (1 << b)
            q3_adj[b] |= (1 << a)
        
        # Precompute: for each 8-set, we need to check if there's an embedding
        # of Q3 into that 8-set's induced subgraph
        
        # Strategy: for each 8-set, try to map Q3 vertices to the 8 vertices
        # Use backtracking with constraint propagation
        
        # Order Q3 vertices by degree (all degree 3, so use fixed order)
        # But we can use a smarter order based on the structure
        
        # Actually, let's use a different approach:
        # Q3 is the graph of a cube. Two vertices in Q3 are at distance 0,1,2,or 3.
        # We can use the fact that Q3 has specific properties.
        
        # Fastest approach: precompute all "Q3 patterns" on 8 labeled vertices
        # There are 8! = 40320 ways to label, but many are equivalent.
        # Actually, the automorphism group of Q3 has 48 elements,
        # so there are 40320/48 = 840 distinct "shapes" of Q3 on 8 labeled vertices.
        
        # Let's precompute: for each permutation of 8 vertices, check if it's a Q3 embedding
        # Then for each 8-set in our graph, check if any of these patterns matches.
        
        # Precompute all valid Q3 embeddings as edge sets
        # For a permutation p of [0..7], the edges of Q3 under this permutation are:
        # (p[i], p[j]) for each (i,j) in q3_edges
        
        # But we need to check if the induced subgraph on 8 vertices contains Q3
        # as a (not necessarily induced) subgraph.
        
        # Key insight: Q3 has 12 edges. If the induced subgraph on 8 vertices has
        # fewer than 12 edges, it can't contain Q3. If it has >= 12 edges, we need
        # to check if Q3 is a subgraph.
        
        # For speed, let's use a different strategy:
        # 1. Quick filter: count edges in induced subgraph, if < 12, skip
        # 2. Use backtracking with strong pruning
        
        for vs in all_8sets:
            # Quick filter: count edges in induced subgraph
            edge_count = 0
            for i in range(8):
                for j in range(i + 1, 8):
                    u, w = vs[i], vs[j]
                    if adj_mask[u] & (1 << w):
                        edge_count += 1
            if edge_count < 12:
                continue
            
            # Build local adjacency
            # local_adj[i] = bitmask of neighbors within vs
            local_adj = [0] * 8
            for i in range(8):
                for j in range(8):
                    if i != j and adj_mask[vs[i]] & (1 << vs[j]):
                        local_adj[i] |= (1 << j)
            
            # Check if Q3 is a subgraph using backtracking
            # Assign Q3 vertex 0 to some vertex in vs, then propagate
            # Q3: vertices 0-7, edges where Hamming distance = 1
            
            # Use backtracking with ordering
            # Try to assign Q3 vertices in order 0,1,2,...,7
            # But we can use symmetry breaking: fix Q3 vertex 0 to vs[0]
            
            # Actually, let's try all 8 choices for where Q3 vertex 0 maps,
            # but with symmetry breaking we can fix it to vs[0] and try all
            # permutations of the remaining 7 for Q3 vertices 1-7.
            
            # Better: use a recursive backtracking with strong pruning.
            
            assignment = [-1] * 8  # q3_vertex -> local_vertex
            used = [False] * 8  # local_vertex -> assigned?
            
            # Precompute Q3 adjacency for quick lookup
            # q3_neighbors[v] = list of Q3 vertices adjacent to v
            q3_neighbors = [[] for _ in range(8)]
            for a, b in q3_edges:
                q3_neighbors[a].append(b)
                q3_neighbors[b].append(a)
            
            def backtrack(qv):
                if qv == 8:
                    return True
                for lv in range(8):
                    if used[lv]:
                        continue
                    # Check if lv is compatible with already-assigned neighbors
                    ok = True
                    for nb in q3_neighbors[qv]:
                        if nb < qv:  # already assigned
                            if not (local_adj[lv] & (1 << assignment[nb])):
                                ok = False
                                break
                    if not ok:
                        continue
                    assignment[qv] = lv
                    used[lv] = True
                    if backtrack(qv + 1):
                        return True
                    used[lv] = False
                    assignment[qv] = -1
                return False
            
            # Symmetry breaking: fix Q3 vertex 0 to local vertex 0
            # Actually, we can't fix it because Q3 vertex 0 could map to any of the 8
            # Let's just try all 8 starting points
            
            # To speed up, try the most constrained first
            # All Q3 vertices have degree 3, so no difference
            
            # Try with symmetry breaking: Q3 vertex 0 maps to local vertex 0
            assignment[0] = 0
            used[0] = True
            if backtrack(1):
                return True
            assignment[0] = -1
            used[0] = False
            
            # Also try Q3 vertex 0 mapping to other local vertices
            # But this is expensive. Let's just do the above first and see.
            
            # If that didn't work, try other starting points
            for start_lv in range(1, 8):
                assignment[0] = start_lv
                used[start_lv] = True
                if backtrack(1):
                    return True
                used[start_lv] = False
                assignment[0] = -1
        
        return False
    
    # This is still slow. Let's use a much faster approach.
    # Precompute all 840 distinct Q3 edge patterns on 8 labeled vertices.
    # For each pattern, it's a set of 12 edges (as pairs of 0..7).
    # For each 8-set in our graph, check if any pattern is a subgraph.
    
    # Generate all distinct Q3 embeddings
    # Two embeddings are equivalent if they differ by an automorphism of Q3
    # Automorphisms of Q3: generated by coordinate swaps and bit flips
    
    # Actually, let's just generate all 40320 permutations and deduplicate
    # by the resulting edge set (after sorting).
    
    # For a permutation p of [0..7], the Q3 edges become:
    # {(p[i], p[j]) for (i,j) in q3_edges}
    # Normalize: sort each edge, then sort the list of edges
    
    q3_patterns = set()
    from itertools import permutations
    
    # 40320 permutations is a lot. Let's use the automorphism group to reduce.
    # Aut(Q3) has 48 elements. So there are 40320/48 = 840 distinct patterns.
    
    # Generate automorphisms of Q3
    # Q3 vertices are 3-bit strings. Automorphisms are:
    # - Permute coordinates (6 ways)
    # - Flip any subset of coordinates (8 ways)
    # Total: 6 * 8 = 48
    
    def apply_automorphism(v, perm, flips):
        """Apply coordinate permutation and bit flips to vertex v (3-bit)."""
        bits = [(v >> i) & 1 for i in range(3)]
        # Permute
        new_bits = [bits[perm[i]] for i in range(3)]
        # Flip
        for i in range(3):
            if flips[i]:
                new_bits[i] ^= 1
        result = 0
        for i in range(3):
            result |= new_bits[i] << i
        return result
    
    # Generate all 48 automorphisms
    autos = []
    for perm in permutations(range(3)):
        for flips in range(8):
            flip_bits = [(flips >> i) & 1 for i in range(3)]
            autos.append((perm, flip_bits))
    
    # Generate one representative pattern, then apply all automorphisms
    # to get all 48 equivalent patterns. Then for each permutation of the
    # "orbit representative", generate more patterns.
    
    # Actually, simpler: just generate all 40320 and deduplicate.
    # 40320 is manageable.
    
    # But we need to be careful about time. Let's precompute once.
    
    # For each permutation p of [0..7]:
    #   pattern = frozenset of sorted edges
    # Store in a set to deduplicate.
    
    # This takes some time but only once.
    
    # Actually, let's use a smarter approach. The number of distinct Q3
    # subgraphs on 8 labeled vertices is 840. Let's just hardcode the check
    # using a different method.
    
    # Alternative fast check:
    # Q3 is bipartite with parts of size 4. In any Q3 embedding,
    # the 8 vertices split into two sets of 4, with edges only between parts.
    # Each vertex has degree 3 in Q3.
    
    # For a given 8-set with induced subgraph:
    # 1. Find all bipartitions into two sets of 4
    # 2. For each, check if the bipartite graph between the parts is Q3
    
    # Number of ways to choose 4 from 8: C(8,4) = 70
    # For each, check if the bipartite graph matches Q3's bipartite structure.
    
    # Q3 bipartite structure:
    # Part A (even parity): {0, 2, 4, 6}
    # Part B (odd parity): {1, 3, 5, 7}
    # Edges: (0,1),(0,3),(0,5), (2,1),(2,3),(2,7), (4,5),(4,7),(4,1), (6,7),(6,5),(6,3)
    # Wait, let me recompute.
    # 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
    # Edges (Hamming distance 1):
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    
    # So the bipartite graph has specific structure.
    
    # This approach: for each 8-set, for each of 70 bipartitions,
    # check if the bipartite graph is isomorphic to Q3's bipartite graph.
    
    # Q3's bipartite graph: 4+4 vertices, 12 edges, each vertex degree 3.
    # The bipartite graph is actually the 3x3 grid minus corners... no.
    # It's a 4-regular... no, 3-regular bipartite graph on 4+4 vertices.
    # There's only one 3-regular bipartite graph on 4+4 vertices (up to isomorphism),
    # which is K_{4,4} minus a perfect matching.
    
    # Wait, is that right? Let me verify.
    # Q3 bipartite: A={0,2,4,6}, B={1,3,5,7}
    # 0 connects to 1,2,4 -> but 2 and 4 are in A! That's wrong.
    # 0=000, neighbors: 001(1), 010(2), 100(4). 1 is odd, 2 is even, 4 is even.
    # So 0 connects to 1 (B), 2 (A), 4 (A). That's not bipartite with parity!
    
    # I made an error. Q3 is bipartite, but the bipartition is not by parity of
    # the 3-bit representation. Let me reconsider.
    
    # Q3 is bipartite. The bipartition is by the sum of bits mod 2.
    # 0=000 (sum 0, even), 1=001 (sum 1, odd), 2=010 (sum 1, odd),
    # 3=011 (sum 2, even), 4=100 (sum 1, odd), 5=101 (sum 2, even),
    # 6=110 (sum 2, even), 7=111 (sum 3, odd)
    
    # So A (even sum): {0, 3, 5, 6}
    # B (odd sum): {1, 2, 4, 7}
    
    # Edges:
    # 0-1 (A-B), 0-2 (A-B), 0-4 (A-B)
    # 3-1 (A-B), 3-2 (A-B), 3-7 (A-B)
    # 5-4 (A-B), 5-7 (A-B), 5-1 (A-B)
    # 6-2 (A-B), 6-4 (A-B), 6-7 (A-B)
    
    # Yes! This is a proper bipartition. Each vertex has degree 3.
    # The bipartite graph is 3-regular on 4+4 vertices.
    
    # Now, is there only one 3-regular bipartite graph on 4+4 vertices?
    # K_{4,4} has 16 edges. Removing a perfect matching (4 edges) gives 12 edges,
    # and each vertex has degree 3. So yes, it's K_{4,4} minus a perfect matching.
    
    # So the check is: for each 8-set, for each bipartition into A and B (|A|=|B|=4),
    # check if the bipartite graph between A and B is K_{4,4} minus a perfect matching.
    
    # K_{4,4} minus a perfect matching: each vertex in A is connected to exactly
    # 3 of the 4 vertices in B, and the missing edges form a perfect matching.
    
    # This is much faster to check!
    
    # Let me implement this approach.
    
    def has_q3_fast(adj_mask):
        """Fast check using bipartite structure of Q3."""
        for vs in all_8sets:
            # Build local adjacency bitmask
            # local_adj[i] has bit j set if vs[i]-vs[j] is an edge
            local_adj = [0] * 8
            for i in range(8):
                for j in range(8):
                    if i != j and adj_mask[vs[i]] & (1 << vs[j]):
                        local_adj[i] |= (1 << j)
            
            # Count edges
            edge_count = sum(bin(local_adj[i]).count('1') for i in range(8)) // 2
            if edge_count < 12:
                continue
            
            # Try all bipartitions into two sets of 4
            # C(8,4) = 70, but we can halve by symmetry (A,B) ~ (B,A)
            # So 35 unique bipartitions
            from itertools import combinations as comb
            for A in comb(range(8), 4):
                B = [i for i in range(8) if i not in A]
                # Check if bipartite graph between A and B is K_{4,4} minus perfect matching
                # For each vertex in A, it should have exactly 3 neighbors in B
                # And the missing neighbors should form a perfect matching
                
                # Check degrees
                ok = True
                missing = {}  # a -> b that is missing
                for a in A:
                    neighbors_in_B = [b for b in B if local_adj[a] & (1 << b)]
                    if len(neighbors_in_B) != 3:
                        ok = False
                        break
                    missing[a] = [b for b in B if not (local_adj[a] & (1 << b))][0]
                
                if not ok:
                    continue
                
                # Check that missing edges form a perfect matching
                # i.e., all missing[b] are distinct and cover all of B
                missing_B = [missing[a] for a in A]
                if len(set(missing_B)) != 4:
                    continue
                
                # Also verify the reverse: each b in B has exactly one missing edge from A
                # This is guaranteed by the above if missing_B is a permutation of B
                
                # Verify it's actually a subgraph (not just bipartite)
                # We need to ensure that all 12 edges of Q3 are present
                # In our check, we verified that each a in A has 3 neighbors in B,
                # and the missing edges form a perfect matching. This means the
                # bipartite graph is exactly K_{4,4} minus a perfect matching,
                # which is isomorphic to Q3's bipartite graph.
                
                # But we also need to make sure that there are no edges within A
                # or within B that would make it not a subgraph... wait, we're checking
                # for Q3 as a subgraph, not induced subgraph. So edges within A or B
                # don't matter. Q3 has no edges within A or within B, so we just need
                # the bipartite edges to match.
                
                # Actually, we need to verify that the specific bipartite graph is
                # isomorphic to Q3's bipartite graph. Since both are 3-regular
                # bipartite on 4+4 vertices, and we've shown it's K_{4,4} minus
                # a perfect matching, and Q3's bipartite graph is also K_{4,4}
                # minus a perfect matching, they are isomorphic.
                
                # Wait, I need to double-check that Q3's bipartite graph is
                # indeed K_{4,4} minus a perfect matching.
                # A={0,3,5,6}, B={1,2,4,7}
                # 0 connects to 1,2,4 (missing 7)
                # 3 connects to 1,2,7 (missing 4)
                # 5 connects to 1,4,7 (missing 2)
                # 6 connects to 2,4,7 (missing 1)
                # Missing: 0-7, 3-4, 5-2, 6-1
                # Is this a perfect matching? {0-7, 3-4, 5-2, 6-1}
                # Each vertex appears exactly once. Yes!
                
                return True
        
        return False
    
    # Now, let's use this fast check for our search.
    
    # Strategy:
    # 1. Start with a random dense graph
    # 2. Use local search to add edges while maintaining Q3-freeness
    # 3. Use simulated annealing or greedy with random restarts
    
    # For the local search, when we try to add an edge (u,v),
    # we only need to check 8-sets that contain both u and v.
    # Number of such 8-sets: C(14, 6) = 3003
    
    # Precompute for each pair (u,v) the list of 8-sets containing both
    pair_to_8sets = {}
    for i in range(16):
        for j in range(i + 1, 16):
            remaining = [x for x in range(16) if x != i and x != j]
            pair_to_8sets[(i, j)] = list(combinations(remaining, 6))
    
    def check_edge_addition(adj_mask, u, v):
        """Check if adding edge (u,v) creates a Q3."""
        # Only check 8-sets containing both u and v
        for other6 in pair_to_8sets[(min(u,v), max(u,v))]:
            vs = tuple(sorted([u, v] + list(other6)))
            # Build local adjacency with the new edge
            local_adj = [0] * 8
            vertex_to_local = {vs[i]: i for i in range(8)}
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    a, b = vs[i], vs[j]
                    if a == u and b == v or a == v and b == u:
                        local_adj[i] |= (1 << j)
                    elif adj_mask[a] & (1 << b):
                        local_adj[i] |= (1 << j)
            
            # Quick edge count
            edge_count = sum(bin(local_adj[i]).count('1') for i in range(8)) // 2
            if edge_count < 12:
                continue
            
            # Check bipartite structure
            from itertools import combinations as comb
            for A in comb(range(8), 4):
                B = [i for i in range(8) if i not in A]
                ok = True
                missing = {}
                for a in A:
                    neighbors_in_B = [b for b in B if local_adj[a] & (1 << b)]
                    if len(neighbors_in_B) != 3:
                        ok = False
                        break
                    missing[a] = [b for b in B if not (local_adj[a] & (1 << b))][0]
                
                if not ok:
                    continue
                
                missing_B = [missing[a] for a in A]
                if len(set(missing_B)) != 4:
                    continue
                return True  # Q3 found
        
        return False  # No Q3 created
    
    # Main search
    random.seed(42)
    
    best_adj = 0  # bitmask representation
    best_count = 0
    
    # Start with a few random initializations
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        
        # Random initial graph
        p = random.uniform(0.5, 0.8)
        adj = [0] * 16
        edge_count = 0
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edge_count += 1
        
        # Local search: try to add edges
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            # Try all missing edges in random order
            missing_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                            if not (adj[i] & (1 << j))]
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if time.time() > deadline - 10:
                    break
                if check_edge_addition(adj, u, v):
                    continue
                # Add the edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                changed = True
        
        if edge_count > best_count:
            best_count = edge_count
            best_adj = list(adj)
    
    # Final greedy pass on the best solution
    if best_adj and time.time() < deadline - 5:
        adj = list(best_adj)
        edge_count = best_count
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            missing_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                            if not (adj[i] & (1 << j))]
            random.shuffle(missing_edges)
            for u, v in missing_edges:
                if time.time() > deadline - 3:
                    break
                if check_edge_addition(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                changed = True
        if edge_count > best_count:
            best_count = edge_count
            best_adj = adj
    
    # Convert to edge list
    result = []
    for i in range(16):
        for j in range(i + 1, 16):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
