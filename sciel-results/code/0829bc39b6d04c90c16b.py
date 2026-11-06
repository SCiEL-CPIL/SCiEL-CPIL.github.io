import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit
    
    n = 16
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)
    
    # Precompute adjacency lists for pairs for faster intersection?
    # Actually, for n=16, we can represent the graph as a bitmask of 120 bits.
    # But checking Q3 requires finding 8 specific vertices and 12 edges.
    # A Q3 is a bipartite graph K_{4,4} minus perfect matching? No, Q3 is the 3-cube.
    # Q3 has 8 vertices and 12 edges.
    # It is a bipartite graph with partitions of size 4.
    # Let's enumerate all subsets of 8 vertices from 16. C(16,8) = 12870.
    # For each subset, check if it contains a Q3 subgraph.
    # This is still too slow for every step of a search.
    
    # Better approach: Incremental validity check.
    # Maintain adjacency matrix as list of ints (bitmasks).
    # When adding edge (u,v), check if it creates a Q3.
    # A new edge (u,v) creates a Q3 if there exists a Q3 in G + (u,v) that uses (u,v).
    # This is equivalent to finding a Q3 in G where u and v are non-adjacent,
    # and adding (u,v) completes the cycle/structure.
    # Specifically, Q3 is the cube. It has 6 faces (C4s).
    # If we add (u,v), we might create a C4, which might be part of a Q3.
    # Actually, simpler: Just check if G + (u,v) contains Q3.
    # Since n=16 is small, we can precompute all Q3 embeddings?
    # Number of Q3 embeddings in K16 is large.
    
    # Alternative: Use the fact that Q3 is a specific graph.
    # Let's use a randomized local search with a fast validity checker.
    # Validity checker:
    # 1. Build adjacency matrix.
    # 2. Iterate over all 8-subsets of vertices.
    # 3. For each 8-subset, check if the induced subgraph contains a Q3.
    #    Checking if an 8-vertex graph contains Q3:
    #    Q3 is bipartite. We can check if the 8-vertex graph has a bipartition (A,B) of size 4
    #    such that all edges between A and B in Q3 are present.
    #    There are C(8,4)/2 = 35 bipartitions.
    #    For a fixed bipartition, there are 4! ways to map Q3 vertices to A and B?
    #    Actually, Q3 is vertex-transitive? No.
    #    Let's just hardcode the edges of Q3 relative to a canonical labeling.
    #    Q3 vertices: 0..7. Edges: (i,j) if popcount(i xor j) == 1.
    #    For a given set of 8 vertices S and a bijection f: {0..7} -> S,
    #    check if all Q3 edges are present in G[S].
    #    Number of bijections: 8! = 40320.
    #    12870 subsets * 40320 bijections * 12 edge checks = ~6 * 10^9 ops. Too slow.
    
    # Optimization:
    # Only check bijections that preserve the bipartite structure.
    # Q3 is bipartite with parts of size 4.
    # If S does not have a bipartite induced subgraph, skip.
    # Also, we can use backtracking to find Q3 in the 8-vertex graph.
    # Or, simpler:
    # Precompute all "Q3 patterns" as bitmasks of edges on 8 vertices.
    # There are 8! / (automorphisms of Q3) distinct patterns?
    # Automorphism group of Q3 has size 48.
    # 40320 / 48 = 840 distinct edge patterns.
    # So for each 8-subset, we only need to check 840 patterns.
    # 12870 * 840 * 12 checks ~ 1.3 * 10^8. Still too slow for every move.
    
    # We need a faster check.
    # Let's use the property: Q3 is C4-free? No, Q3 has C4s.
    # Q3 is bipartite.
    
    # Let's try a different strategy:
    # Start with a known good Q3-free graph.
    # The Turan graph T(16,2) is K8,8 which has 64 edges.
    # Does K8,8 contain Q3?
    # Q3 is bipartite with parts of size 4.
    # K8,8 contains K4,4. Q3 is a subgraph of K4,4?
    # Q3 has 12 edges. K4,4 has 16 edges.
    # Q3 is a subgraph of K4,4 if we can map Q3 to K4,4.
    # Yes, Q3 is a subgraph of K4,4 (remove a perfect matching from K4,4).
    # So K8,8 contains Q3.
    
    # We need a graph that is Q3-free.
    # Random dense graphs are likely to contain Q3.
    # We need to avoid Q3.
    
    # Let's use a simulated annealing approach with a fast local check.
    # Fast local check for adding edge (u,v):
    # A Q3 using (u,v) must have u and v at distance 1 in Q3.
    # In Q3, every edge is in 2 faces (C4s).
    # So (u,v) is in 2 C4s in the Q3.
    # Let the neighbors of u in Q3 be u1, u2 (and v).
    # Let the neighbors of v in Q3 be v1, v2 (and u).
    # The C4s are u-v-v1-u1-u and u-v-v2-u2-u.
    # This implies u1 and v1 are connected? No, u1 is connected to u and v1?
    # In Q3, if u-v is an edge, and u-u1 is an edge, then u1 is not connected to v.
    # u1 is connected to u and u1' (opposite).
    # Let's trace Q3 structure:
    # Vertices: 000, 001, 010, 011, 100, 101, 110, 111.
    # Edge 000-001.
    # Neighbors of 000: 001, 010, 100.
    # Neighbors of 001: 000, 011, 101.
    # Common neighbors of 000 and 001? None. (Distance 1).
    # So u and v have no common neighbors in Q3.
    # The C4s containing edge (u,v):
    # 000-001-011-010-000. Edges: (000,001), (001,011), (011,010), (010,000).
    # So u=000, v=001, u1=010, v1=011.
    # Edges: (u,u1), (v,v1), (u1,v1).
    # So adding (u,v) creates a Q3 if there exist u1, v1 such that:
    # (u,u1) in G, (v,v1) in G, (u1,v1) in G, AND the rest of the Q3 exists.
    # This is getting complicated.
    
    # Given the time limit, I will use a simpler heuristic:
    # 1. Start with a random graph.
    # 2. Find all Q3s and remove edges to break them.
    # 3. Add edges back greedily.
    # To find Q3s fast, I will use a precomputed list of all 840 patterns on 8 vertices.
    # For each 8-subset, check if any pattern is present.
    
    # Precomputation
    def get_q3_patterns():
        # Generate all permutations of 8 vertices
        import itertools
        patterns = set()
        # Q3 edges in canonical form (0..7)
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    q3_edges.append((i, j))
        
        # We need to check if a graph on 8 vertices contains Q3.
        # Instead of checking all 840 patterns, we can just check if the 8-vertex graph
        # has a Q3 subgraph via backtracking or bipartite matching?
        # Q3 is bipartite.
        # Let's just use a simple check:
        # For a given 8-vertex graph G_S, does it contain Q3?
        # We can try to embed Q3 into G_S.
        # Since Q3 is small, we can use a simple backtracking search.
        # Or, we can precompute the 840 bitmasks of edges on 8 vertices that form Q3.
        # Let's do that.
        
        base_edges = q3_edges
        perms = itertools.permutations(range(8))
        seen = set()
        for p in perms:
            # p is a mapping from canonical vertex i to actual vertex p[i]
            # Create edge set
            edges = set()
            for u, v in base_edges:
                eu = p[u]
                ev = p[v]
                if eu > ev:
                    eu, ev = ev, eu
                edges.add((eu, ev))
            # Canonical representation of edge set
            key = tuple(sorted(edges))
            if key not in seen:
                seen.add(key)
                patterns.add(key)
        return list(patterns)
    
    # This precomputation might take a few seconds.
    # Let's estimate: 40320 perms. 12 edges each.
    # It should be fast enough (< 1s).
    
    q3_patterns = get_q3_patterns()
    
    # Map 8-subset to index? No, just iterate.
    # To speed up checking a specific 8-subset S:
    # Build a 8x8 adjacency matrix or bitmask for the induced subgraph.
    # Then check if any pattern matches.
    
    def is_q3_free(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Iterate over all 8-subsets
        # C(16,8) = 12870
        # For each subset, build local adjacency
        # Check patterns
        
        import itertools
        for subset in itertools.combinations(range(n), 8):
            # Build local adjacency bitmask (64 bits)
            # Map subset vertices to 0..7
            idx_map = {v: i for i, v in enumerate(subset)}
            local_adj = [0] * 8
            for i, u in enumerate(subset):
                for j, v in enumerate(subset):
                    if i != j and (adj[u] >> v) & 1:
                        local_adj[i] |= (1 << j)
            
            # Check if any pattern is present
            # Pattern is a set of 12 edges.
            # Represent pattern as a 64-bit mask?
            # Edge (i,j) in 8x8 corresponds to bit i*8+j.
            # Let's precompute pattern masks.
            pass
            
        return True # Placeholder, need to implement properly
    
    # This is getting too complex to implement correctly and fast in this context.
    # I will fall back to a robust random restart local search with a simpler validity check
    # that might be slower but correct, or use a known construction.
    
    # Known construction:
    # The graph formed by the 16 vertices of a 4D hypercube Q4?
    # Q4 has 16 vertices and 32 edges. It is Q3-free?
    # Q4 contains Q3 as a subgraph (fix one bit). So Q4 is NOT Q3-free.
    
    # What about the complement of Q4?
    # Q4 has 32 edges. Complement has 120 - 32 = 88 edges.
    # Does complement of Q4 contain Q3?
    # Likely yes.
    
    # Let's try a random search with a fast approximate check, then verify.
    
    def check_q3_fast(adj):
        # Check all 8-subsets
        import itertools
        for subset in itertools.combinations(range(n), 8):
            # Quick check: Q3 is bipartite.
            # If the induced subgraph is not bipartite, it can't contain Q3?
            # No, Q3 is bipartite, but a non-bipartite graph can contain a bipartite subgraph.
            # So that doesn't help.
            
            # Check if the induced subgraph has at least 12 edges.
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj[subset[i]] >> subset[j]) & 1:
                        edge_count += 1
            if edge_count < 12:
                continue
                
            # Detailed check:
            # Try to find Q3 in this 8-vertex graph.
            # Use backtracking or precomputed patterns.
            # For speed, let's just say "maybe" and let the main loop handle it?
            # No, we need certainty.
            
            # Let's use the precomputed patterns.
            # We need to map the 8 vertices to 0..7 and check against patterns.
            # This is O(840 * 12) per subset.
            # 12870 * 10000 = 1.2 * 10^8. Too slow.
            
        return True
    
    # Given the constraints, I will use a simpler strategy:
    # 1. Start with a random graph.
    # 2. Use a local search that removes edges from found Q3s.
    # 3. To find Q3s, I will use a randomized check:
    #    Pick a random 8-subset. Check if it contains Q3.
    #    If yes, remove a random edge from it.
    #    This is a Las Vegas algorithm.
    
    def find_q3_in_subset(adj, subset):
        # Check if the induced subgraph on 'subset' contains a Q3.
        # Returns a list of edges in the Q3, or None.
        import itertools
        # Map subset to 0..7
        idx_map = {v: i for i, v in enumerate(subset)}
        local_adj = [0] * 8
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and (adj[u] >> v) & 1:
                    local_adj[i] |= (1 << j)
        
        # Check all 840 patterns
        # Precompute pattern masks as 64-bit integers
        # Edge (i,j) -> bit i*8+j
        for pattern in q3_patterns:
            mask = 0
            for u, v in pattern:
                mask |= (1 << (u * 8 + v))
                mask |= (1 << (v * 8 + u))
            
            # Check if local_adj contains this mask
            # local_adj is a list of 8 ints.
            # We can combine them into one 64-bit int?
            # local_mask = sum(local_adj[i] << (i*8) for i in range(8))
            local_mask = 0
            for i in range(8):
                local_mask |= (local_adj[i] << (i * 8))
                
            if (local_mask & mask) == mask:
                # Found a Q3
                # Return the edges in terms of original vertices
                edges = []
                for u, v in pattern:
                    edges.append((subset[u], subset[v]))
                return edges
        return None
    
    # Precompute pattern masks
    pattern_masks = []
    for pattern in q3_patterns:
        mask = 0
        for u, v in pattern:
            mask |= (1 << (u * 8 + v))
            mask |= (1 << (v * 8 + u))
        pattern_masks.append(mask)
        
    def find_q3(adj):
        import itertools
        # Randomly sample 8-subsets
        # To be thorough, we should check all, but that's slow.
        # Let's check all 12870 subsets.
        # 12870 * 840 checks = 10.8M checks.
        # Each check is a bitwise AND.
        # This should be fast enough in Python if optimized?
        # Maybe not. 10M iterations in Python is ~1s.
        # So 12870 * 840 = 10.8M. ~1s per full scan.
        # We have 115s. So we can do ~100 full scans.
        
        for subset in itertools.combinations(range(n), 8):
            idx_map = {v: i for i, v in enumerate(subset)}
            local_mask = 0
            for i, u in enumerate(subset):
                for j, v in enumerate(subset):
                    if i != j and (adj[u] >> v) & 1:
                        local_mask |= (1 << (i * 8 + j))
            
            for mask in pattern_masks:
                if (local_mask & mask) == mask:
                    # Found Q3
                    # Extract edges
                    edges = []
                    # We need to map back to original vertices
                    # The pattern is on 0..7.
                    # We need to find which pattern it was?
                    # We just found a mask match.
                    # We can reconstruct the edges from the mask and subset.
                    # But we need the specific edges to remove.
                    # Let's just return the subset and let the caller remove a random edge from the induced subgraph?
                    # No, we must remove an edge that is in the Q3.
                    # Since we know the mask, we can find the edges.
                    # But the mask is generic.
                    # Let's return the subset. The caller can pick a random edge from the induced subgraph?
                    # That might not break the Q3 if the edge is not in the Q3.
                    # So we need the specific edges.
                    
                    # Let's find the edges from the mask.
                    # The mask corresponds to a specific pattern.
                    # We can iterate through the pattern edges and check if they are in the mask.
                    # But we don't know which pattern it was.
                    # However, the mask is unique to a pattern?
                    # No, different patterns might have the same mask?
                    # No, the mask is the set of edges.
                    # So we can just extract the edges from the mask.
                    edges = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            bit = (1 << (i * 8 + j)) | (1 << (j * 8 + i))
                            if (mask & bit) == bit:
                                edges.append((subset[i], subset[j]))
                    return edges
        return None
    
    best_edges = 0
    best_graph = set()
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Start with random graph p=0.6
        current_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.6:
                    current_edges.add((u, v))
        
        # Remove Q3s
        while time.time() < deadline:
            adj = [0] * n
            for u, v in current_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            edge_to_remove = random.choice(q3_edges)
            current_edges.remove(edge_to_remove)
            
        # Add edges back
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            all_pairs_list = list(all_pairs)
            random.shuffle(all_pairs_list)
            for u, v in all_pairs_list:
                if (u, v) in current_edges:
                    continue
                # Try adding
                new_edges = current_edges | {(u, v)}
                adj = [0] * n
                for uu, vv in new_edges:
                    adj[uu] |= (1 << vv)
                    adj[vv] |= (1 << uu)
                    
                if find_q3(adj) is None:
                    current_edges = new_edges
                    added_any = True
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_graph = set(current_edges)
            
    return [[u, v] for u, v in best_graph]

construct()
