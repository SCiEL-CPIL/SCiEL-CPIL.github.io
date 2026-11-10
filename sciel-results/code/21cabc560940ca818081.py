import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    n = 16
    
    # Q3 structure: 8 vertices.
    # In Q3, every vertex has degree 3.
    # Opposite pairs (distance 3): (0,7), (1,6), (2,5), (3,4) assuming standard labeling 000..111.
    # Let's define the Q3 edges and non-edges for a canonical set of 8 vertices.
    # Vertices: 0..7
    # Edges in Q3 (12 edges):
    q3_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    # Non-edges in Q3 (16 edges among 8 vertices, 12 are edges, 4 are opposite, 16-12-4=0? No.
    # Total pairs 28. 12 edges. 4 opposite (dist 3). 12 dist 2.
    # Wait, dist 2 pairs are NOT edges in Q3.
    # So non-edges are the 16 pairs that are not in q3_edges.
    
    # To check if a set of 8 vertices induces a Q3, we need to check if there is an isomorphism.
    # A faster check: 
    # 1. Pick a vertex v0.
    # 2. Its neighbors in the induced subgraph must be exactly 3 (if it's a Q3).
    # Actually, we are looking for a SUBGRAPH isomorphic to Q3, not necessarily induced.
    # "containing no subgraph isomorphic to the 3-cube Q3"
    # This means we cannot have a set of 8 vertices and a mapping to Q3 such that all 12 edges of Q3 exist in our graph.
    # The other edges (non-edges of Q3) can exist or not.
    
    # So, to find a Q3 subgraph:
    # We need to find 8 vertices {v0..v7} and a bijection to {0..7} of Q3
    # such that for every (u,v) in q3_edges, (map(u), map(v)) is an edge in G.
    
    # Strategy for finding a Q3 in G:
    # Iterate over all 8-subsets? C(16,8) = 12870.
    # For each subset, check if it contains a Q3.
    # Checking if a specific 8-vertex induced subgraph contains a Q3 as a subgraph is still hard if done naively.
    # However, we can use the "opposite pair" structure.
    # In any Q3, there are 4 pairs of opposite vertices.
    # Let the opposite pairs be (a,b), (c,d), (e,f), (g,h).
    # The vertices are partitioned into these 4 pairs.
    # For a candidate set of 8 vertices, we can try to partition them into 4 pairs.
    # There are (8-1)!! = 105 ways to partition 8 vertices into 4 pairs.
    # For each partition, we check if it's consistent with Q3 structure.
    # In Q3, if (a,b) is an opposite pair, then a and b are NOT adjacent.
    # Also, any neighbor of a is adjacent to exactly 2 neighbors of b?
    # Let's use a simpler necessary condition:
    # In Q3, the graph is bipartite.
    # So the 8 vertices must be 2-colorable.
    # If the induced subgraph on the 8 vertices is not bipartite, it cannot contain a Q3?
    # No, the Q3 subgraph itself is bipartite, but the induced subgraph might have extra edges that make it non-bipartite.
    # However, the edges of the Q3 subgraph must form a bipartite graph.
    
    # Let's try a different approach:
    # Iterate over all 4-cycles?
    # A Q3 contains many 4-cycles.
    # Actually, the most efficient way for n=16 is likely to precompute all possible Q3 embeddings in K16?
    # Number of embeddings: Choose 8 vertices (12870) * Number of automorphisms of Q3 (48) / (something)?
    # Actually, we can just iterate over all 8-subsets and check if they contain a Q3.
    # 12870 subsets is small.
    # For each subset, we need to check if there exists a labeling of the 8 vertices as Q3 vertices such that all 12 edges exist.
    # We can fix one vertex to be 0 to break symmetry.
    # Then its 3 neighbors in Q3 must be neighbors in G.
    # Let the neighbors of 0 in G (within the subset) be N0.
    # If |N0| < 3, no Q3.
    # If |N0| >= 3, pick 3 neighbors n1, n2, n3.
    # In Q3, n1, n2, n3 are pairwise non-adjacent?
    # In Q3, neighbors of 0 are 1, 2, 4.
    # 1-2? No. 1-4? No. 2-4? No.
    # So n1, n2, n3 must form an independent set in G (within the subset).
    # If they don't, this choice of neighbors is invalid.
    # Then, the remaining 4 vertices are determined by the structure.
    # This might be fast enough.

    def creates_q3_in_subset(subset, adj):
        # subset is a tuple of 8 vertex indices
        # adj is a list of bitmasks
        # Check if subset contains a Q3 subgraph
        
        # Map subset vertices to 0..7 for local indexing? No, use original indices.
        # Fix vertex s[0] to be Q3 vertex 0.
        # Q3 neighbors of 0: 1, 2, 4.
        # In G, s[0] must have at least 3 neighbors in the subset.
        # Let's find neighbors of s[0] in the subset.
        s0 = subset[0]
        neighbors_in_subset = []
        for v in subset[1:]:
            if adj[s0] & (1 << v):
                neighbors_in_subset.append(v)
        
        if len(neighbors_in_subset) < 3:
            return False
            
        # We need to choose 3 neighbors from neighbors_in_subset to map to 1, 2, 4.
        # They must be pairwise non-adjacent in G.
        # Let's iterate over combinations of 3 neighbors.
        from itertools import combinations
        for n1, n2, n4 in combinations(neighbors_in_subset, 3):
            # Check pairwise non-adjacency
            if (adj[n1] & (1 << n2)) or (adj[n1] & (1 << n4)) or (adj[n2] & (1 << n4)):
                continue
                
            # If we map s[0]->0, n1->1, n2->2, n4->4
            # Then the rest of the Q3 vertices are determined:
            # 3 = 1+2 (in Q3 logic, 3 is neighbor of 1 and 2)
            # 5 = 1+4
            # 6 = 2+4
            # 7 = 1+2+4 (opposite of 0)
            
            # We need to find vertices v3, v5, v6, v7 in the remaining 4 vertices of the subset
            # such that:
            # v3 is adjacent to n1 and n2
            # v5 is adjacent to n1 and n4
            # v6 is adjacent to n2 and n4
            # v7 is adjacent to v3, v5, v6 (and not to s0, n1, n2, n4? No, v7 is opposite to s0, so not adjacent to s0.
            # v7 is adjacent to 3,5,6. Is v7 adjacent to 1,2,4? No.
            # In Q3:
            # 1 is 001, 2 is 010, 4 is 100.
            # 3 is 011, 5 is 101, 6 is 110, 7 is 111.
            # 7 is adjacent to 3,5,6.
            # 7 is NOT adjacent to 0,1,2,4.
            
            remaining = [v for v in subset[1:] if v not in [n1, n2, n4]]
            # remaining has 4 vertices
            
            # Find v3: neighbor of n1 and n2 in remaining
            v3_candidates = [v for v in remaining if (adj[n1] & (1 << v)) and (adj[n2] & (1 << v))]
            if not v3_candidates:
                continue
                
            # Find v5: neighbor of n1 and n4 in remaining
            v5_candidates = [v for v in remaining if (adj[n1] & (1 << v)) and (adj[n4] & (1 << v))]
            if not v5_candidates:
                continue
                
            # Find v6: neighbor of n2 and n4 in remaining
            v6_candidates = [v for v in remaining if (adj[n2] & (1 << v)) and (adj[n4] & (1 << v))]
            if not v6_candidates:
                continue
            
            # Try all combinations of v3, v5, v6
            for v3 in v3_candidates:
                for v5 in v5_candidates:
                    for v6 in v6_candidates:
                        # v3, v5, v6 must be distinct
                        if v3 == v5 or v3 == v6 or v5 == v6:
                            continue
                        # The last remaining vertex is v7
                        v7_candidates = [v for v in remaining if v not in [v3, v5, v6]]
                        if not v7_candidates:
                            continue
                        v7 = v7_candidates[0]
                        
                        # Check edges for v7:
                        # v7 must be adjacent to v3, v5, v6
                        if not (adj[v7] & (1 << v3)): continue
                        if not (adj[v7] & (1 << v5)): continue
                        if not (adj[v7] & (1 << v6)): continue
                        
                        # Check non-edges for v7 (optional, but good for pruning if we were checking induced, 
                        # but for subgraph, we only care about existing edges. 
                        # However, if v7 is adjacent to s0, n1, n2, or n4, that's fine for subgraph containment, 
                        # as long as the Q3 edges exist.
                        # Wait, the definition is "subgraph isomorphic to Q3". 
                        # This means there is a set of 8 vertices and a set of 12 edges forming a Q3.
                        # Extra edges don't matter.
                        
                        # So if all 12 Q3 edges exist, we found a Q3.
                        return True
        return False

    def find_q3(adj):
        """
        Find any Q3 subgraph in the graph represented by adj.
        Returns a list of 8 vertices or None.
        """
        # Iterate over all 8-subsets
        from itertools import combinations
        for subset in combinations(range(n), 8):
            if creates_q3_in_subset(subset, adj):
                return list(subset)
        return None

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    # Start with K16
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_count = count_edges(adj)
    best_adj = adj[:]
    best_count = current_count

    # We need to remove edges until no Q3 exists.
    # Strategy: 
    # While there is a Q3:
    #   Find a Q3.
    #   Remove an edge from it.
    #   Which edge? 
    #   To maximize remaining edges, we should remove an edge that is part of the "most" Q3s?
    #   Or just remove a random one?
    #   Or remove the edge that appears in the most Q3s? That's expensive.
    #   Let's try: Remove a random edge from the found Q3.
    #   But this might be slow if we remove one at a time.
    
    # Better: 
    # Find a Q3.
    # Identify all edges in this Q3.
    # For each edge, check if removing it breaks this specific Q3 (yes).
    # But does it help break other Q3s?
    
    # Let's try a heuristic: 
    # 1. Find a Q3.
    # 2. Pick the edge in this Q3 with the highest "centrality" or just random.
    # 3. Remove it.
    # 4. Repeat.
    
    # Optimization: 
    # If we remove an edge, we don't need to re-scan from scratch immediately if we track Q3s.
    # But n=16 is small.
    
    # Let's improve the search:
    # Instead of removing one edge at a time, we can do a local search on the set of removed edges.
    # But the "remove one at a time" is a greedy descent.
    
    # To speed up, we can use the fact that if we remove an edge (u,v), 
    # any Q3 containing (u,v) is destroyed.
    
    # Let's run the greedy removal.
    
    while time.time() < deadline - 5:
        q3 = find_q3(adj)
        if q3 is None:
            break
            
        # We found a Q3. It has 12 edges.
        # Let's pick an edge to remove.
        # Heuristic: Remove the edge that is shared by the most Q3s?
        # We don't know that.
        # Let's just pick a random edge from the Q3.
        # Or, pick the edge with the highest degree sum?
        
        # Let's try to remove an edge that is "central".
        # For each of the 12 edges in the Q3, count how many other vertices are connected to both ends?
        # No, just random is fine for now, but let's try to be smart.
        
        # Let's collect the 12 edges of the found Q3.
        # We need to reconstruct the edges from the subset and the mapping.
        # My find_q3 doesn't return the mapping.
        # Let's modify creates_q3_in_subset to return the edges if found.
        
        # Actually, let's just pick a random pair of vertices from the Q3 subset that are adjacent in G.
        # There are at least 12 such pairs.
        
        q3_verts = set(q3)
        candidate_edges = []
        for u in q3:
            for v in q3:
                if u < v and (adj[u] & (1 << v)):
                    candidate_edges.append((u, v))
                    
        if not candidate_edges:
            # Should not happen
            break
            
        # Remove a random edge
        u, v = random.choice(candidate_edges)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        current_count -= 1
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]

    # Local search on the removed edges?
    # We have a Q3-free graph.
    # Can we add edges back?
    # If we add an edge, we might create a Q3.
    # Let's try to add edges that don't create a Q3.
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        We can use the same logic as before, but optimized.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # We need to find a Q3 containing (u,v).
        # In Q3, (u,v) can be:
        # 1. An edge (distance 1)
        # 2. Distance 2
        # 3. Distance 3 (opposite)
        
        # Case 1: (u,v) is an edge in Q3.
        # Then u and v are adjacent in Q3.
        # Let's assume u=0, v=1.
        # Neighbors of 0 in Q3: 1, 2, 4.
        # Neighbors of 1 in Q3: 0, 3, 5.
        # So we need to find a2 (neighbor of u, not v) and a3 (neighbor of v, not u).
        # a2 must be 2 or 4. a3 must be 3 or 5.
        # In Q3, 2 is adjacent to 3, 6, 0.
        # 4 is adjacent to 5, 6, 0.
        # 3 is adjacent to 1, 2, 7.
        # 5 is adjacent to 1, 4, 7.
        # 6 is adjacent to 2, 4, 7.
        # 7 is adjacent to 3, 5, 6.
        
        # So if u=0, v=1:
        # a2 in {2,4}, a3 in {3,5}.
        # If a2=2, then a3 must be adjacent to 2? No, a3 is neighbor of v=1.
        # The other vertices are 3,4,5,6,7.
        # 2 is adjacent to 3, 6.
        # 4 is adjacent to 5, 6.
        # 3 is adjacent to 2, 7.
        # 5 is adjacent to 4, 7.
        # 6 is adjacent to 2, 4, 7.
        # 7 is adjacent to 3, 5, 6.
        
        # So, if we pick a2 (neighbor of u), and a3 (neighbor of v):
        # We need to find a6 (neighbor of a2 and a4? No.)
        # Let's use the structure:
        # If (u,v) is an edge, say u=0, v=1.
        # We need to find a2, a4 (neighbors of u other than v) and a3, a5 (neighbors of v other than u).
        # Actually, u has 3 neighbors in Q3: v, a2, a4.
        # v has 3 neighbors in Q3: u, a3, a5.
        # a2 is adjacent to a3 and a6?
        # Let's stick to the previous `creates_q3_fast` which was working.
        # It checked for Q3 creation by iterating over neighbors.
        
        # I will reuse the logic from the seed program's `creates_q3_fast` as it was effective.
        # But I need to ensure it's correct for the "add edge" case.
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    # Now, try to add edges back.
    # We have best_adj.
    # Let's try to add missing edges.
    
    for _ in range(50): # Iterations
        if time.time() > deadline - 2:
            break
        
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (best_adj[u] & (1 << v))]
        random.shuffle(missing)
        
        added = 0
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            if not creates_q3_fast(u, v, best_adj):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_count += 1
                added += 1
        if added == 0:
            break
            
    return edges_from_adj(best_adj)

# Test
if __name__ == "__main__":
    # Quick test to ensure it runs
    edges = construct()
    print(f"Edges: {len(edges)}")
