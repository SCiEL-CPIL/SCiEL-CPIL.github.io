import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 135.0
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # 1. Generate all edges in K_16
    all_edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
            
    # 2. Identify all Q3 subgraphs in K_16
    # A Q3 is defined by 8 vertices and 12 edges.
    # We need to build a conflict graph on the 120 edge-nodes.
    # Conflict: Edge A and Edge B are connected if they are both in some Q3.
    
    # Precompute Q3 structure
    # Q3 vertices are 3-bit numbers 0..7. Edges connect if Hamming distance is 1.
    Q3_EDGES = []
    for u in range(8):
        for v in range(u + 1, 8):
            if bin(u ^ v).count('1') == 1:
                Q3_EDGES.append((u, v))
                
    # To speed up conflict detection, we can iterate over all 8-vertex subsets of {0..15}
    # For each subset, check if it induces a Q3 (or contains a Q3).
    # Actually, we just need to know if a specific set of 8 vertices *can* be mapped to Q3.
    # Since we are looking for subgraphs (not induced), a subset S of 8 vertices
    # contributes to conflicts if the subgraph induced by S in K_16 (which is K_8)
    # contains Q3. K_8 always contains Q3.
    # So, ANY 8 vertices in K_16 form a "potential Q3 host".
    # Wait, the definition of Q3-free is "no subgraph isomorphic to Q3".
    # If we pick any 8 vertices, the edges between them in our candidate graph must NOT
    # form a Q3.
    # In the conflict graph formulation for MIS:
    # Node = Edge e in K_16.
    # Two nodes e1, e2 are connected if there exists a set of 8 vertices V8 such that
    # e1 and e2 are both in V8, and the specific pattern of edges required for Q3
    # is "threatened".
    
    # Actually, a simpler and more direct conflict definition:
    # e1 and e2 are in conflict if they are part of the same Q3 *instance*.
    # A Q3 instance is defined by a bijection from Q3_vertices to Graph_vertices.
    # Let's iterate over all 8-subsets S of {0..15}.
    # For each S, there are 8! ways to map Q3 vertices to S.
    # For each mapping, we get 12 specific edges in the graph.
    # All pairs of these 12 edges are in conflict.
    
    # This is too many iterations (12870 subsets * 40320 mappings).
    # Optimization: 
    # The conflict graph is vertex-transitive-ish? No.
    # Let's use a faster check.
    # Two edges (u,v) and (x,y) are in conflict if they can be part of a Q3.
    # This happens if the 4 distinct vertices {u,v,x,y} (or 3, or 2) can be embedded in Q3.
    # Actually, the simplest robust way for n=16 is to precompute the conflict matrix
    # by iterating over all possible Q3 embeddings.
    
    # Number of 8-subsets is 12870.
    # For each subset S, we can check if S contains a Q3.
    # But we need to mark conflicts between specific edges.
    
    # Let's try a different perspective:
    # The conflict graph H has 120 nodes.
    # H is the union of all "Q3 edge-cliques".
    # A Q3 on vertices S={v0..v7} creates a clique on the 12 edges of the Q3.
    # So, if we find all sets of 12 edges that form a Q3, we add a clique to H.
    
    # Efficiently finding all Q3s:
    # Iterate over all 8-subsets S.
    # For a given S, does it contain a Q3?
    # Yes, K_8 contains many Q3s.
    # Actually, we don't need to iterate mappings.
    # We can just say: If edge e1 and e2 are in conflict, it means there is a Q3 containing both.
    
    # Let's precompute the conflict bitmasks.
    # conflict_mask[i] is a bitmask of 120 bits, where bit j is 1 if edge i and edge j are in conflict.
    
    conflict_masks = [0] * total_possible_edges
    
    # To avoid iterating 12870 * 40320, we can use the fact that Q3 is small.
    # Two edges are in conflict if their endpoints can be mapped to edges in Q3.
    # Let's iterate over all pairs of edges? 120*120 = 14400 pairs.
    # For each pair, check if they can be part of a Q3.
    # This check:
    # Given edges (u,v) and (x,y).
    # Can we extend this to a Q3?
    # This is still complex.
    
    # Alternative: Iterate over all 8-subsets S.
    # For each S, find all Q3s inside S.
    # Since S is small (8 vertices), we can brute force check all 8! mappings? No, 40320 is fine for 12870 subsets? 
    # 12870 * 40320 = 519 million operations. Too slow in Python.
    
    # Better: 
    # For each 8-subset S, the subgraph induced is K_8.
    # The edges of K_8 are 28 edges.
    # We need to find all Q3s in K_8.
    # A Q3 in K_8 corresponds to a labeling of the 8 vertices with 0..7 such that 
    # the edges of the Q3 (12 specific pairs) are a subset of the 28 edges of K_8.
    # Since K_8 has all edges, ANY labeling of the 8 vertices of S to 0..7 creates a valid Q3 embedding.
    # So, for each 8-subset S, and for each permutation of S (mapping 0..7 to S), 
    # the 12 edges of the Q3 are present.
    # This means that for ANY 8-subset S, ANY two edges within S are in conflict?
    # No. Only if they belong to the SAME Q3 instance.
    # If edge e1 and e2 are in S, is there a Q3 in S containing both?
    # In K_8, yes. Any two edges in K_8 can be part of some Q3?
    # Let's check. Q3 has 12 edges. K_8 has 28.
    # If we pick any 2 edges in K_8, can we embed them in Q3?
    # If the two edges are disjoint (matching), yes.
    # If they share a vertex, yes.
    # Actually, Q3 is 3-regular. 
    # If we pick two edges that are "far apart" in Q3, do they co-occur?
    # Yes, as long as they are in the same Q3.
    # The question is: For a fixed set of 8 vertices S, and two specific edges e1, e2 in S,
    # is there a bijection f: Q3_V -> S such that f(Q3_E) contains e1 and e2?
    # This is equivalent to: Do e1 and e2 form a subgraph of Q3?
    # Since e1 and e2 are just edges in the complete graph on S, they don't have "Q3 structure" yet.
    # We are asking if the pair of edges {e1, e2} is a subgraph of Q3.
    # Any two edges in a graph are a subgraph of another graph if the other graph contains that pattern.
    # Q3 contains two disjoint edges? Yes.
    # Q3 contains two adjacent edges? Yes.
    # So, for ANY 8-subset S, and ANY pair of edges e1, e2 within S,
    # there exists a Q3 in S containing both e1 and e2?
    # Let's verify.
    # Take Q3. Pick any two edges. They are in Q3.
    # So if we map the vertices of Q3 to S, those two edges map to e1 and e2?
    # No, we need the mapping to map the specific edges of Q3 to e1 and e2.
    # This requires that the "shape" of e1 and e2 (relative to each other) matches a shape in Q3.
    # Shapes:
    # 1. Disjoint edges: Q3 has disjoint edges.
    # 2. Adjacent edges: Q3 has adjacent edges.
    # Are there any other shapes? 
    # With 2 edges, the only invariants are: do they share a vertex?
    # If yes, they form a path of length 2. Q3 has paths of length 2.
    # If no, they are disjoint. Q3 has disjoint edges.
    # So, YES. For any 8-subset S, and any two edges e1, e2 in S, they are in conflict.
    
    # Wait, this implies that the conflict graph H is such that any two edges sharing the same 8-vertex superset are connected?
    # No. e1 and e2 are in conflict if there EXISTS a Q3 containing them.
    # The Q3 must be on 8 vertices.
    # If e1=(u,v) and e2=(x,y).
    # Let V_e = {u,v} U {x,y}.
    # If |V_e| <= 8, we can extend V_e to a set S of size 8.
    # Does S contain a Q3 that includes e1 and e2?
    # As established above, if the relative structure of e1 and e2 (adjacent or disjoint) is realizable in Q3, then yes.
    # Both are realizable.
    # So, e1 and e2 are in conflict IF AND ONLY IF |{u,v,x,y}| <= 8?
    # No, the Q3 has exactly 8 vertices.
    # If e1 and e2 use 4 distinct vertices, we can pick 4 more vertices to make 8.
    # If e1 and e2 use 3 distinct vertices (share one), we can pick 5 more.
    # If e1 and e e2 use 2 distinct vertices (same edge), trivial.
    # So, essentially, ANY two distinct edges in K_16 are in conflict?
    # Let's check the constraint: "subgraph isomorphic to Q3".
    # If e1 and e2 are in conflict, it means we cannot have both in our final graph, because if we did,
    # we could potentially complete the rest of the Q3 edges?
    # NO. The definition of Q3-free is that the graph does NOT contain a Q3.
    # If we include e1 and e2, we are not FORCED to include the other 10 edges of the Q3.
    # We only get a Q3 if ALL 12 edges are present.
    
    # My previous MIS formulation was:
    # "Two edge-nodes are adjacent if their union is contained within the vertex set of any Q3-inducing subgraph"
    # This interpretation is WRONG for standard subgraph avoidance.
    # If we just want to avoid having a Q3, we can have many edges.
    # The conflict is: We cannot have a SET of 12 edges that forms a Q3.
    # This is a Hypergraph Independent Set problem (or hitting set).
    # Nodes: 120 edges.
    # Hyperedges: Sets of 12 edges that form a Q3.
    # We want the largest set of nodes that contains NO hyperedge.
    
    # This is equivalent to: Find the maximum number of edges such that no Q3 is fully present.
    # This is NOT a standard MIS on a graph where edges are conflicts.
    # It is a "Maximal Subgraph without Forbidden Configuration".
    
    # However, we can transform it.
    # If we select a set of edges E', it is valid iff for every Q3 Q, E' does NOT contain all edges of Q.
    # This is the complement of a Hitting Set problem? No.
    
    # Let's stick to the heuristic approach but make it much faster and smarter.
    # The previous heuristic was: Start dense, remove edges from found Q3s.
    # Why did it score 1.15? Because it's slow and gets stuck.
    # The inspiration scored 1.20.
    # The observed child scored 1.33.
    
    # Let's try a "Local Search" on the edge set.
    # 1. Start with a high-quality initial guess.
    #    What is a good initial guess?
    #    Complete graph K_16 has 120 edges. It contains many Q3s.
    #    We need to remove edges to break all Q3s.
    #    This is equivalent to finding the minimum number of edges to remove to destroy all Q3s.
    #    Let R be the set of removed edges. We want to minimize |R| such that (K_16 \ R) is Q3-free.
    #    Then the answer is 120 - |R|.
    
    #    This is a Minimum Hitting Set problem on the hypergraph of Q3s.
    #    Nodes: 120 edges.
    #    Hyperedges: 12-sets of edges forming a Q3.
    #    We want the smallest set of nodes hitting all hyperedges.
    
    #    Since n=16 is small, we can use a greedy hitting set algorithm with local search (2-opt/3-opt swaps).
    
    #    Steps:
    #    1. Enumerate all Q3s in K_16.
    #       How many?
    #       Number of 8-subsets: 12870.
    #       For each 8-subset, how many Q3s?
    #       A Q3 is a specific set of 12 edges on 8 vertices.
    #       In K_8, the number of Q3s is the number of isomorphic copies of Q3 in K_8.
    #       This is a fixed number, let's call it C.
    #       Total Q3s = 12870 * C.
    #       C is the number of ways to map Q3 to 8 labeled vertices.
    #       Q3 has 8 vertices. Aut(Q3) has size 48.
    #       Number of labelings = 8! / 48 = 40320 / 48 = 840.
    #       So there are 840 distinct Q3s in K_8.
    #       Total Q3s in K_16 = 12870 * 840 = 10,810,800.
    #       This is ~10 million Q3s.
    #       Storing 10 million bitmasks of 120 bits is feasible (10M * 16 bytes = 160 MB).
    #       But iterating over 10M Q3s in Python to build the hypergraph might be slow.
    
    #    2. Greedy Hitting Set:
    #       - Start with empty set R (removed edges).
    #       - While there exists a Q3 Q that is not hit by R (i.e., all edges of Q are present):
    #           - Pick a Q3 Q.
    #           - Add an edge from Q to R.
    #           - Which edge? The one that hits the most other "unhit" Q3s.
    
    #    3. Local Search:
    #       - Try to remove an edge from R (i.e., put it back in the graph).
    #       - If this creates a new Q3, try to swap it with another edge in R to maintain the property and reduce size.
    
    #    Implementation details:
    #    - We need to efficiently check if a Q3 is "active" (all its edges are in the current graph).
    #    - We need to efficiently find active Q3s.
    
    #    Let's represent the current graph as a bitmask of 120 edges.
    #    active_q3s = [True] * num_q3s
    #    
    #    Precomputation:
    #    - Generate all Q3s (list of 120-bit masks).
    #    - This is the bottleneck. 10M masks.
    #    - Generating 10M masks in Python might take > 10s.
    #    - We can optimize by generating the 840 masks for K_8 once, and then applying permutations of vertices 0..15 to map them to the 12870 subsets.
    
    #    Let's try this approach.
    
    def generate_q3_masks():
        # Returns a list of 120-bit integers representing all Q3s in K_16
        # 1. Get all 840 Q3 masks for vertices 0..7
        base_masks = []
        # We need all permutations of 0..7 that represent distinct Q3 embeddings.
        # Actually, we can just iterate over all 8! permutations of range(8) and generate the mask.
        # 40320 permutations.
        # For each perm p, the Q3 edges are (p[u], p[v]) for (u,v) in Q3_EDGES.
        # We need to map these to edge indices in K_8 (0..7).
        
        # Edge index in K_8 (0..7):
        # Total edges 28.
        # We need a mapping from (u,v) to 0..27.
        k8_edge_idx = {}
        idx = 0
        for u in range(8):
            for v in range(u+1, 8):
                k8_edge_idx[(u,v)] = idx
                idx += 1
        
        seen_masks = set()
        for p in itertools.permutations(range(8)):
            mask = 0
            valid = True
            for u, v in Q3_EDGES:
                eu, ev = p[u], p[v]
                if eu > ev: eu, ev = ev, eu
                if (eu, ev) not in k8_edge_idx:
                    valid = False
                    break
                idx8 = k8_edge_idx[(eu, ev)]
                mask |= (1 << idx8)
            if valid:
                seen_masks.add(mask)
        
        # There should be 840 unique masks.
        base_masks = list(seen_masks)
        
        # 2. Map these to K_16
        # For each 8-subset S of 0..15:
        #   For each base mask:
        #     Apply the permutation S to the mask.
        #     The base mask has bits 0..27 set corresponding to edges in K_8.
        #     We need to map edge i in K_8 to the corresponding edge index in K_16.
        
        # Precompute mapping from K_8 edge index to (u,v) in 0..7
        k8_edges_list = []
        for u in range(8):
            for v in range(u+1, 8):
                k8_edges_list.append((u,v))
                
        # For a given subset S (sorted list of 8 vertices), and a permutation pi of 0..7,
        # the vertex at position i in K_8 is mapped to S[pi[i]]?
        # No, the base mask was generated by permuting the labels 0..7.
        # Let's say the base mask corresponds to a specific labeling L: Q3_V -> {0..7}.
        # To map this to S, we need a bijection M: {0..7} -> S.
        # The edges in the Q3 are (L(u), L(v)).
        # In the graph on S, the edges are (M(L(u)), M(L(v))).
        # So we need to map the edge index of (L(u), L(v)) in K_8 to the edge index of (M(L(u)), M(L(v))) in K_16.
        
        # This is getting complicated.
        # Simpler:
        # For each 8-subset S:
        #   Generate all 840 Q3 masks directly on S.
        #   How?
        #   Iterate over all 8! permutations of S.
        #   For each perm p (where p[i] is the vertex in S corresponding to Q3 vertex i):
        #     Build the 120-bit mask for K_16.
        
        all_q3_masks = []
        seen_all = set()
        
        # Optimization: 
        # Instead of 12870 * 40320 = 500M iterations, 
        # we can use the fact that the structure is invariant.
        # But 500M is too slow.
        
        # Alternative: 
        # We don't need to store ALL Q3s.
        # We can use a "Lazy" approach.
        # We maintain the current graph G.
        # We need to know if G contains a Q3.
        # We can check this by iterating over 8-subsets and checking if they induce a Q3.
        # This is what the current program does.
        
        # The problem with the current program is that it's SLOW.
        # Why? 
        # 1. It iterates 12870 subsets.
        # 2. For each, it does backtracking.
        # 3. It does this in a loop until no Q3 is found.
        
        # Let's optimize the "Check if G contains Q3" function.
        # We can precompute the list of 840 "patterns" of edges in K_8.
        # For a given 8-subset S, we extract the subgraph of G induced by S.
        # We get a 28-bit mask of which edges are present in S.
        # We check if this 28-bit mask is a SUPERSET of any of the 840 base Q3 masks.
        # This is a fast bitwise operation!
        
        # Precompute the 840 base masks (28-bit integers).
        base_q3_masks_28 = []
        for p in itertools.permutations(range(8)):
            mask = 0
            for u, v in Q3_EDGES:
                eu, ev = p[u], p[v]
                if eu > ev: eu, ev = ev, eu
                idx8 = k8_edge_idx[(eu, ev)]
                mask |= (1 << idx8)
            if mask not in base_q3_masks_28:
                base_q3_masks_28.append(mask)
                
        # There are 840 masks.
        
        # Function to check if G contains a Q3:
        def contains_q3(adj_mask_16):
            # adj_mask_16: list of 16 ints (bitmasks of neighbors)
            # Or better: a single 120-bit int representing the edge set.
            # Let's convert adj_mask_16 to a 120-bit int.
            
            # Actually, for speed, let's keep the 16-bit adjacency list.
            # For each 8-subset S:
            #   Build the 28-bit mask of edges present in S.
            #   Check if (mask & base) == base for any base in base_q3_masks_28.
            
            # To speed up, we can precompute the 120-bit mask for each 8-subset?
            # No, the graph changes.
            
            # We can precompute the list of (u,v) pairs for each of the 28 edges in K_8.
            # And for a given S (sorted tuple), we can map K_8 edge i to the actual edge in G.
            
            # Let's precompute for each 8-subset S:
            #   A list of 28 indices into the 120-edge array.
            
            subset_edge_indices = {}
            for S in itertools.combinations(range(n), 8):
                S_set = set(S)
                # Map local index 0..7 to global vertex
                # S is already sorted.
                # Local edge (i, j) corresponds to global edge (S[i], S[j]).
                # We need the index of (S[i], S[j]) in the 120-edge array.
                indices = []
                for i in range(8):
                    for j in range(i+1, 8):
                        u, v = S[i], S[j]
                        idx = edge_to_idx[(u, v)]
                        indices.append(idx)
                subset_edge_indices[S] = indices
                
            # Now, contains_q3 can be:
            def _contains_q3(edge_mask_120):
                # edge_mask_120 is a 120-bit int
                for S, indices in subset_edge_indices.items():
                    # Extract the 28-bit mask for this subset
                    local_mask = 0
                    for k, idx in enumerate(indices):
                        if (edge_mask_120 >> idx) & 1:
                            local_mask |= (1 << k)
                    
                    # Check against base masks
                    # Optimization: 
                    # If local_mask has fewer than 12 bits, skip.
                    if local_mask.bit_count() < 12:
                        continue
                            
                    for bmask in base_q3_masks_28:
                        if (local_mask & bmask) == bmask:
                            return True, S, bmask
                return False, None, None
            
            return _contains_q3(edge_mask_120)
            
        # This check is O(12870 * 840) = ~10M operations per check.
        # In Python, 10M ops is ~0.5s.
        # If we do 100 checks, that's 50s.
        # We need to be faster.
        
        # Optimization: 
        # We can precompute for each 28-bit local mask, whether it contains a Q3.
        # There are 2^28 = 268M possible masks. Too many to precompute.
        
        # Alternative: 
        # Use the "Greedy Removal" but with a faster "Find One Q3".
        # The current program's `find_one_q3_fast` is slow because of the backtracking.
        # The new `contains_q3` above is faster (bitwise AND).
        
        # Let's refine the strategy:
        # 1. Start with K_16 (120 edges).
        # 2. While contains_q3(G):
        #      Find a Q3.
        #      Remove an edge from it.
        #      Which edge? 
        #      To maximize remaining edges, we should remove an edge that is in the MOST Q3s.
        #      This is the "Most Constrained Variable" heuristic.
        
        # 3. After getting a Q3-free graph, try to ADD edges back.
        #      For each missing edge, try adding it. If it creates a Q3, don't add.
        
        # This is essentially the same as before, but with faster Q3 detection.
        
        # Let's implement this.
        
        # Precomputation time:
        # 12870 subsets * 28 edges = 360k operations. Fast.
        
        # Runtime:
        # Each `find_q3` takes ~0.5s (10M ops).
        # We might need 20-30 removals. 30 * 0.5s = 15s.
        # Then adding edges back: 120 candidates * 0.5s = 60s.
        # Total ~75s. Fits in 135s.
        
        # Let's code this.
        
        # 1. Precompute subset_edge_indices
        subset_edge_indices = {}
        for S in itertools.combinations(range(n), 8):
            indices = []
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = S[i], S[j]
                    idx = edge_to_idx[(u, v)]
                    indices.append(idx)
            subset_edge_indices[S] = indices
            
        # 2. Precompute base_q3_masks_28
        base_q3_masks_28 = []
        for p in itertools.permutations(range(8)):
            mask = 0
            for u, v in Q3_EDGES:
                eu, ev = p[u], p[v]
                if eu > ev: eu, ev = ev, eu
                idx8 = k8_edge_idx[(eu, ev)]
                mask |= (1 << idx8)
            if mask not in base_q3_masks_28:
                base_q3_masks_28.append(mask)
                
        # 3. Helper to convert 16-adj-mask to 120-edge-mask
        def adj_to_edge_mask(adj):
            mask = 0
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        idx = edge_to_idx[(u,v)]
                        mask |= (1 << idx)
            return mask
            
        # 4. Helper to convert 120-edge-mask to adj list
        def edge_mask_to_adj(mask):
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if (mask >> edge_to_idx[(u,v)]) & 1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            return adj
            
        # 5. Find one Q3
        def find_one_q3(edge_mask):
            # Returns (True, list_of_edge_indices_in_q3) or (False, [])
            for S, indices in subset_edge_indices.items():
                local_mask = 0
                for k, idx in enumerate(indices):
                    if (edge_mask >> idx) & 1:
                        local_mask |= (1 << k)
                        
                if local_mask.bit_count() < 12:
                    continue
                    
                for bmask in base_q3_masks_28:
                    if (local_mask & bmask) == bmask:
                        # Found it. Extract the 12 edge indices.
                        q3_edges = []
                        for k in range(28):
                            if (bmask >> k) & 1:
                                q3_edges.append(indices[k])
                        return True, q3_edges
            return False, []
            
        # 6. Main Loop
        # Start with K_16
        initial_mask = (1 << total_possible_edges) - 1
        current_mask = initial_mask
        
        # Greedy Removal
        # We want to remove edges that break the most Q3s.
        # But we don't know the counts.
        # Let's just remove a random edge from the found Q3.
        # To improve, we can try removing each edge in the Q3 and see which one
        # results in the fewest "new" Q3s? Too expensive.
        
        # Let's use a simple random removal first, then local search.
        
        rng = random.Random(42)
        
        max_removals = 100
        for _ in range(max_removals):
            if time.time() > deadline - 10.0:
                break
                
            found, q3_edges = find_one_q3(current_mask)
            if not found:
                break
                
            # Pick a random edge from the Q3 to remove
            # Heuristic: Remove the edge that is "central"?
            # For now, random.
            idx_to_remove = rng.choice(q3_edges)
            current_mask &= ~(1 << idx_to_remove)
            
        # Now current_mask is Q3-free.
        # Try to add edges back.
        
        # Get list of missing edges
        missing_edges = [i for i in range(total_possible_edges) if not (current_mask >> i) & 1]
        rng.shuffle(missing_edges)
        
        for idx in missing_edges:
            if time.time() > deadline - 2.0:
                break
                
            # Try adding edge idx
            new_mask = current_mask | (1 << idx)
            found, _ = find_one_q3(new_mask)
            if not found:
                current_mask = new_mask
                
        # Convert back to edge list
        final_adj = edge_mask_to_adj(current_mask)
        final_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (final_adj[u] >> v) & 1:
                    final_edges.append([u, v])
                    
        return final_edges

    # Run the main logic
    # Note: The precomputation of subset_edge_indices takes a bit of time.
    # 12870 * 28 = 360k lookups. Fast.
    
    # Let's execute the logic inside construct to ensure variables are local.
    
    # Re-implementing inside construct for clarity and closure access.
    
    # 1. Precompute
    subset_edge_indices = {}
    for S in itertools.combinations(range(n), 8):
        indices = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = S[i], S[j]
                idx = edge_to_idx[(u, v)]
                indices.append(idx)
        subset_edge_indices[S] = indices
        
    base_q3_masks_28 = []
    for p in itertools.permutations(range(8)):
        mask = 0
        for u, v in Q3_EDGES:
            eu, ev = p[u], p[v]
            if eu > ev: eu, ev = ev, eu
            idx8 = k8_edge_idx[(eu, ev)]
            mask |= (1 << idx8)
        if mask not in base_q3_masks_28:
            base_q3_masks_28.append(mask)
            
    def adj_to_edge_mask(adj):
        mask = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    idx = edge_to_idx[(u,v)]
                    mask |= (1 << idx)
        return mask
        
    def edge_mask_to_adj(mask):
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if (mask >> edge_to_idx[(u,v)]) & 1:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
        
    def find_one_q3(edge_mask):
        for S, indices in subset_edge_indices.items():
            local_mask = 0
            for k, idx in enumerate(indices):
                if (edge_mask >> idx) & 1:
                    local_mask |= (1 << k)
                    
            if local_mask.bit_count() < 12:
                continue
                
            for bmask in base_q3_masks_28:
                if (local_mask & bmask) == bmask:
                    q3_edges = []
                    for k in range(28):
                        if (bmask >> k) & 1:
                            q3_edges.append(indices[k])
                    return True, q3_edges
        return False, []
        
    # Start with K_16
    current_mask = (1 << total_possible_edges) - 1
    rng = random.Random(123)
    
    # Greedy Removal
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        found, q3_edges = find_one_q3(current_mask)
        if not found:
            break
        idx_to_remove = rng.choice(q3_edges)
        current_mask &= ~(1 << idx_to_remove)
        
    # Add edges back
    missing_edges = [i for i in range(total_possible_edges) if not (current_mask >> i) & 1]
    rng.shuffle(missing_edges)
    
    for idx in missing_edges:
        if time.time() > deadline - 2.0:
            break
        new_mask = current_mask | (1 << idx)
        found, _ = find_one_q3(new_mask)
        if not found:
            current_mask = new_mask
            
    final_adj = edge_mask_to_adj(current_mask)
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (final_adj[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges

