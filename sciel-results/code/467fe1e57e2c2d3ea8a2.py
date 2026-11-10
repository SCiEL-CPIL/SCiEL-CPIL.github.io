import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a K3,3-free graph on 20 vertices maximizing the number of edges.
    
    Strategy:
    1. Use Simulated Annealing with edge-swap moves.
    2. Represent the graph as a list of 20 64-bit integers (bitmasks) for adjacency.
    3. The K3,3 check is optimized using bitwise operations.
       A graph contains K3,3 if there exist disjoint sets A, B of size 3 such that
       all edges between them exist.
       
       Efficient check for if a graph is K3,3-free:
       For every pair of vertices (u, v), let C = N(u) & N(v).
       If |C| >= 3, we must check if there are 3 vertices in C that are mutually 
       non-adjacent to each other? No, that's for K3,3 where u,v are in one part.
       
       Actually, the standard efficient check for K3,3 in a graph G:
       K3,3 exists iff there are two disjoint sets of 3 vertices, A and B, 
       such that every vertex in A is connected to every vertex in B.
       
       Let's re-evaluate the "incremental" check used in the previous best program 
       (which scored ~2.33).
       
       The previous program used a greedy approach. Local search with swaps is 
       structurally different because it can trade one edge for another to 
       reach a denser configuration that is not reachable by just adding edges.
       
       To make this fast, we need a fast `is_k33_free(adj)` check or an incremental 
       violation count.
       
       Given n=20, we can precompute all pairs of disjoint triples (A, B).
       Number of ways to choose 3 from 20 is 1140.
       Number of ways to choose another 3 from the remaining 17 is 680.
       Total pairs of disjoint triples ~ 1140 * 680 / 2 (since A,B unordered? No, 
       in K3,3 A and B are distinct parts, but the graph is undirected. 
       However, the set of edges is the same. So we just need to check if 
       there exists ANY pair of disjoint triples (A, B) such that all 9 edges exist.
       
       Total such pairs: C(20,3) * C(17,3) / 2 ? 
       Actually, if we fix A, B is any 3-set from V\A.
       C(20,3) = 1140.
       C(17,3) = 680.
       Total combinations = 1140 * 680 = 775,200.
       This is small enough to check in a tight loop if we use bitmasks.
       
       For each pair (A, B), we can check if the edges exist by:
       For each a in A:
           if (adj[a] & mask_B) != mask_B: fail.
       
       This is 3 checks per pair. 775,200 * 3 = ~2.3M operations.
       In Python, this might be too slow for *every* step of simulated annealing 
       if we do 10,000 steps. 2.3M * 10k = 23B ops. Too slow.
       
       We need an incremental violation counter or a faster check.
       
       Alternative: 
       Maintain a count of "potential K3,3" violations.
       A violation is a pair of disjoint triples (A,B) fully connected.
       
       When we add an edge (u,v):
       Which violations are created?
       A new violation must include the edge (u,v).
       So (u,v) must be part of the complete bipartite graph K3,3.
       This means u is in A, v is in B (or vice versa).
       Let's assume u in A, v in B.
       Then A = {u, a1, a2} and B = {v, b1, b2}.
       The other 8 edges must already exist.
       
       So, to check if adding (u,v) creates a K3,3:
       Iterate over all pairs (a1, a2) from V \ {u, v} such that:
       - a1, a2 are NOT connected to v? No, in K3,3, vertices in A are not connected to each other? 
         Wait, K3,3 is a bipartite graph. The subgraph induced by A and B has edges 
         ONLY between A and B. 
         However, the definition of "contains K3,3" usually means "contains a subgraph isomorphic to K3,3". 
         It does NOT require the induced subgraph to be exactly K3,3. 
         It just requires the 9 edges to be present. 
         Edges within A or within B are irrelevant to the existence of the subgraph.
         
         So, the condition for A={u,a1,a2}, B={v,b1,b2} to form a K3,3 subgraph is:
         1. u-v edge exists (the one we are adding).
         2. u-b1, u-b2 exist.
         3. a1-v, a1-b1, a1-b2 exist.
         4. a2-v, a2-b1, a2-b2 exist.
         
         So, for a fixed edge (u,v) being added:
         We need to find if there exist a1, a2, b1, b2 distinct from u,v and each other 
         such that:
         - a1, a2 are connected to v.
         - b1, b2 are connected to u.
         - a1, a2 are connected to b1, b2.
         
         Let N(u) be neighbors of u.
         Let N(v) be neighbors of v.
         Candidates for b1, b2: N(u) \ {v}.
         Candidates for a1, a2: N(v) \ {u}.
         
         For each pair (a1, a2) in N(v) \ {u}:
             Let C = N(a1) & N(a2) & N(u) \ {v}.
             If |C| >= 2, then we can pick b1, b2 from C.
             This forms a K3,3.
             
         This is the same check as in the greedy program!
         Complexity: O(|N(v)|^2 * bit_ops).
         |N(v)| <= 19. So ~190 checks. Very fast.
         
         So we can use this check to determine if an edge can be added.
         
         For Simulated Annealing:
         Move: Remove edge (u1, v1), Add edge (u2, v2).
         Net change in edges: 0 (if we just swap) or we can try to add an edge if possible.
         
         Actually, a better move set for maximizing edges:
         1. Try to add a random missing edge. If valid, accept.
         2. If no missing edge is valid (locally maximal), try a swap: 
            Remove a random edge, then try to add a random missing edge.
            If the new graph is valid, accept (edge count stays same, but we might 
            unlock new additions).
            To improve edge count, we need to find a sequence of swaps that allows 
            adding an edge.
            
         Simpler approach: 
         Use "Edge Exchange" moves where we remove one edge and add another. 
         This keeps edge count constant. 
         Then, periodically, try to add an edge.
         
         Even better: 
         Use a penalty function.
         Score = EdgeCount - Penalty * Violations.
         But we want strictly K3,3-free.
         
         Let's stick to:
         1. Randomly generate a valid graph using the greedy method (fast).
         2. Run Local Search:
            - While time remains:
              - Pick a random missing edge (u,v).
              - If adding (u,v) is valid, add it. (Edge count increases)
              - Else, try to "fix" the conflict by removing one of the conflicting edges.
                The conflicting edges are those that prevent (u,v) from being added.
                From the check: 
                Conflict arises if there exist a1, a2 in N(v) such that |N(u) & N(a1) & N(a2) \ {v}| >= 2.
                Let S = N(u) & N(a1) & N(a2) \ {v}.
                If we remove one edge from the set of edges required for the K3,3, the conflict is resolved.
                The edges in the K3,3 are: (u,v) [missing], (u,b1), (u,b2), (a1,v), (a1,b1), (a1,b2), (a2,v), (a2,b1), (a2,b2).
                We can remove one of the existing edges: (u,b1), (u,b2), (a1,v), (a1,b1), (a1,b2), (a2,v), (a2,b1), (a2,b2).
                Removing any one of these breaks the K3,3.
                After removal, we can try to add (u,v).
                
                This is a "repair" move.
                
         This is structurally different from pure greedy because it allows "undoing" 
         previous choices to make room for new edges, potentially reaching a global 
         maximum that greedy misses.
    """
    n = 20
    deadline = time.time() + 145
    
    # Precompute all pairs of vertices
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def can_add_edge(adj, u, v):
        """
        Returns True if adding edge (u,v) to graph with adjacency sets adj 
        does NOT create a K3,3.
        """
        # Check if there exist a1, a2 in N(v) \ {u} such that
        # |N(u) & N(a1) & N(a2) \ {v}| >= 2
        
        # Optimization: If deg(u) + deg(v) is small, it's likely valid.
        # But let's just do the set intersection.
        
        # Get candidates for a1, a2
        # They must be neighbors of v.
        # Note: u is NOT a neighbor of v yet (we are checking before adding).
        # So N(v) does not contain u.
        
        # We need to check pairs from N(v).
        # To speed up, we can iterate over N(v).
        
        # If |N(v)| < 2, no pair exists.
        if len(adj[v]) < 2:
            return True
            
        # Iterate over pairs in N(v)
        # To avoid O(n^2) in the worst case if N(v) is large, 
        # but n=20 is small.
        
        # We can also check from the other side:
        # There exist b1, b2 in N(u) \ {v} such that
        # |N(b1) & N(b2) & N(v) \ {u}| >= 2
        # This is symmetric.
        
        # Let's stick to the first one.
        # We can prune: if len(adj[u]) < 2, return True (need 2 common neighbors in N(u)).
        if len(adj[u]) < 2:
            return True
            
        # Iterate over a1 in N(v)
        # To make it faster, we can pick the smaller set to iterate over?
        # No, the structure is fixed.
        
        # Let's try to be smart:
        # If there are many common neighbors, it's harder.
        
        for a1 in adj[v]:
            # a1 is in N(v)
            # We need a2 in N(v), a2 != a1
            # And |N(u) & N(a1) & N(a2) \ {v}| >= 2
            
            # Precompute common of u and a1
            # common_ua1 = adj[u] & adj[a1]
            # Then for each a2 in N(v) \ {a1}, check if |common_ua1 & adj[a2] \ {v}| >= 2
            
            common_ua1 = adj[u] & adj[a1]
            # Remove v if present (v is in adj[a1], is v in adj[u]? No, not yet. 
            # Is v in common_ua1? 
            # v is in adj[a1]. 
            # v is NOT in adj[u]. 
            # So v is NOT in common_ua1.
            # So we don't need to remove v.
            
            # Now check for a2
            for a2 in adj[v]:
                if a2 == a1:
                    continue
                # Check if there are at least 2 vertices in common_ua1 & adj[a2]
                # These vertices are in N(u), N(a1), N(a2).
                # They are distinct from u, v, a1, a2?
                # They are in N(u), so != u.
                # They are in N(a1), so != a1.
                # They are in N(a2), so != a2.
                # Are they v?
                # v is not in N(u). So v is not in the intersection.
                # So the vertices are valid b1, b2 candidates.
                
                common = common_ua1 & adj[a2]
                if len(common) >= 2:
                    return False
                    
        return True

    def is_valid_graph(adj):
        """
        Checks if the entire graph is K3,3-free.
        Used for verification of the final result.
        """
        # Check all pairs of disjoint triples?
        # Or just check all missing edges? No, that's expensive.
        # We can use the same logic:
        # A graph is K3,3-free iff no K3,3 exists.
        # We can check all pairs of vertices (u,v) that are NOT connected?
        # No, K3,3 can exist even if some edges are missing? No, K3,3 requires 9 edges.
        
        # Actually, we can just check all pairs of triples (A, B).
        # But that's 775k checks.
        
        # Better: Iterate over all pairs of vertices (u, v).
        # If u and v are connected, they can be in different parts.
        # If u and v are NOT connected, they can be in the same part?
        # No, in K3,3, all edges between parts exist.
        
        # Let's just use the property:
        # For every pair of disjoint sets A, B of size 3, 
        # not all 9 edges exist.
        
        # We can do this by iterating over all 3-subsets A.
        # For each A, let N(A) be the common neighbors of all vertices in A?
        # No, we need vertices in B that are connected to ALL vertices in A.
        # Let C = intersection of N(a) for a in A.
        # If |C \ A| >= 3, then there are 3 vertices connected to all in A.
        # Let B be any 3 from C \ A.
        # Do we need B to be independent? No.
        # Do we need B to be disjoint from A? Yes, C \ A ensures that.
        # So if |C \ A| >= 3, we have a K3,3?
        # Wait. If B is a subset of C, then all vertices in B are connected to all vertices in A.
        # This forms a K3,3 subgraph (with parts A and B).
        # Yes!
        
        # So the check is:
        # For every 3-subset A of V:
        #   C = intersection of N(a) for a in A
        #   if len(C \ A) >= 3:
        #       return False
        # return True
        
        # Number of 3-subsets: C(20,3) = 1140.
        # For each, 3 set intersections.
        # 1140 * 3 = 3420 set ops. Very fast.
        
        vertices = range(n)
        # Precompute all 3-subsets?
        # Or just iterate.
        import itertools
        for A in itertools.combinations(vertices, 3):
            # Calculate common neighbors
            common = adj[A[0]] & adj[A[1]] & adj[A[2]]
            # Remove A itself
            common.difference_update(A)
            if len(common) >= 3:
                return False
        return True

    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # 1. Greedy Initialization
        adj = [set() for _ in range(n)]
        edges = []
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                
        current_count = len(edges)
        
        # 2. Local Search with Repairs
        # We maintain a list of missing edges to try adding.
        # We also allow "swaps" if adding fails.
        
        # To speed up, we don't re-scan all missing edges every time.
        # We just pick random missing edges.
        
        # Precompute missing edges? 
        # It changes as we add/remove.
        
        # Simple strategy:
        # Loop until no improvement or time limit.
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add random missing edges
            # Generate a list of missing edges
            # Since n=20, total edges 190.
            # Missing = 190 - current_count.
            
            # To be efficient, we can just iterate through all_pairs and check if missing.
            # But that's 190 checks.
            
            # Better: Maintain a set of missing edges?
            # Or just pick random pairs and check if they are missing.
            
            attempts = 0
            max_attempts = 500 # per pass
            
            while attempts < max_attempts and not improved:
                attempts += 1
                u, v = random.choice(all_pairs)
                if u in adj[v]:
                    continue # already exists
                
                # Try to add
                if can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    current_count += 1
                    improved = True
                else:
                    # Repair: try to remove an edge to make it valid
                    # Find a conflicting K3,3 witness
                    # We need to find a1, a2, b1, b2
                    # Re-run the check logic to find the witness
                    found_witness = False
                    for a1 in adj[v]:
                        common_ua1 = adj[u] & adj[a1]
                        for a2 in adj[v]:
                            if a2 == a1:
                                continue
                            common = common_ua1 & adj[a2]
                            if len(common) >= 2:
                                # Found conflict
                                # The K3,3 involves u, v, a1, a2 and 2 from common
                                # We can remove one of the edges:
                                # (u, b1), (u, b2), (a1, v), (a1, b1), (a1, b2), (a2, v), (a2, b1), (a2, b2)
                                # Let's pick one randomly from the existing edges in this K3,3
                                # Existing edges in the K3,3 (excluding u-v which is missing):
                                candidates_to_remove = []
                                # b1, b2 are in common
                                # Pick two from common
                                # common is a set
                                b_list = list(common)
                                b1, b2 = b_list[0], b_list[1]
                                
                                # Edges to consider for removal:
                                # (u, b1), (u, b2)
                                # (a1, v), (a1, b1), (a1, b2)
                                # (a2, v), (a2, b1), (a2, b2)
                                
                                # Check if they exist (they should, by definition of the witness)
                                # u-b1: u in N(b1)? b1 in N(u)? Yes, b1 in common which is in N(u).
                                # a1-v: a1 in N(v)? Yes, a1 in adj[v].
                                # a1-b1: a1 in N(b1)? b1 in N(a1)? Yes, b1 in common which is in N(a1).
                                
                                candidates_to_remove = [
                                    (u, b1), (u, b2),
                                    (a1, v), (a1, b1), (a1, b2),
                                    (a2, v), (a2, b1), (a2, b2)
                                ]
                                
                                # Remove a random one
                                rem_u, rem_v = random.choice(candidates_to_remove)
                                adj[rem_u].remove(rem_v)
                                adj[rem_v].remove(rem_u)
                                edges.remove([rem_u, rem_v]) # O(N) remove, but N is small
                                current_count -= 1
                                
                                # Now try to add (u, v) again
                                # It should be valid now
                                # But let's verify
                                if can_add_edge(adj, u, v):
                                    adj[u].add(v)
                                    adj[v].add(u)
                                    edges.append((u, v))
                                    current_count += 1
                                    # Net change: 0 edges (removed 1, added 1)
                                    # But we might have unlocked other additions?
                                    # We don't count this as "improved" for the outer loop 
                                    # because count didn't increase.
                                    # But we changed the graph.
                                    # We should continue searching for additions.
                                    # So we don't set improved=True here, 
                                    # but we break the inner attempt loop to 
                                    # start a new pass of trying to add.
                                    break
                                else:
                                    # Should not happen
                                    pass
                                found_witness = True
                                break
                        if found_witness:
                            break
                    if found_witness:
                        # We performed a swap. 
                        # We should restart the attempt loop to try adding more.
                        # Reset attempts? 
                        # Or just break to outer while to try more additions.
                        break

        # Verify the best found so far
        if current_count > best_count:
            # Verify validity
            if is_valid_graph(adj):
                best_count = current_count
                best_edges = [[u, v] for u, v in edges]
            else:
                # Should not happen if can_add_edge is correct
                pass

    return best_edges
# EVOLVE-BLOCK-END
