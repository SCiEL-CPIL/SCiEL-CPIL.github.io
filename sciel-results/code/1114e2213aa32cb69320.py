import random
import itertools
import time

def construct():
    n = 16
    # Q3 edges in a 0-7 labeling
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    # Edges connect if hamming distance 1
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Precompute all 8-vertex subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    
    def contains_q3(adj, subset):
        # Check if subgraph induced by subset contains Q3
        # Map subset vertices to 0..7
        # Try all isomorphisms?
        # Q3 is symmetric. We can fix vertex 0 of Q3 to subset[0].
        # Then try permutations of the rest?
        # Or simpler: check if there is a bijection f: {0..7} -> subset
        # such that for all (u,v) in q3_edges, adj[f[u]][f[v]] is True.
        
        # Optimization: 
        # We know the degrees in Q3 are all 3.
        # In the subgraph, vertices must have degree >= 3 within the subset?
        # Not necessarily, but the induced subgraph must contain Q3.
        
        # Simple backtracking search for isomorphism
        # Since 8 is small, we can just iterate permutations? 8! = 40320.
        # 12870 subsets * 40320 is too slow.
        
        # Better: 
        # Check if the bipartition sizes match? Q3 is bipartite 4-4.
        # If the subgraph is not bipartite, it can still contain Q3.
        
        # Let's use a faster check:
        # For each subset, build the induced adjacency matrix (8x8).
        # Then check if Q3 is a subgraph.
        
        # We can precompute the list of all 8! mappings? No.
        
        # Efficient check:
        # 1. Identify the 4 vertices of one part of Q3.
        # 2. In Q3, the two parts are independent sets.
        # 3. So we need to find 4 vertices in subset with no edges between them? No, 
        #    in the *subgraph* they must have no edges between them corresponding to Q3 edges.
        #    But they might have other edges in G.
        #    Wait, "contains Q3 as a subgraph" means there exist 8 vertices v1..v8 
        #    such that for every edge (u,v) in Q3, (f(u),f(v)) is in G.
        #    Edges in G that are not in Q3 are allowed.
        
        # So we just need to check if there is a mapping f such that 
        # all 12 edges of Q3 are present in G.
        
        # We can iterate over all ways to pick 4 vertices for the first part of Q3.
        # C(8,4) = 70 ways.
        # For each choice of 4 vertices A, the other 4 are B.
        # In Q3, every a in A is connected to 3 vertices in B.
        # So for the mapping to work, for each a in A, there must be at least 3 neighbors in B in G.
        # And the specific 3 neighbors must form the correct pattern.
        
        # Actually, just checking if the 12 edges exist is enough.
        
        # Let's use a randomized check or a simple loop.
        # Given the small size, even a somewhat slow check is okay if we don't do it too many times.
        
        # Let's try:
        # For each permutation of the 8 vertices (40320):
        #   Check if all 12 edges match.
        # This is 40320 * 12 checks. ~500k ops.
        # 12870 subsets * 500k = 6 billion ops. Too slow.
        
        # We need a faster "contains Q3" check.
        
        # Observation: Q3 is the only 3-regular bipartite graph on 8 vertices?
        # No, but it's unique in some sense?
        # Actually, we can check if the subgraph contains a 3-regular bipartite subgraph on 8 vertices.
        # If it does, is it Q3?
        # There are two 3-regular bipartite graphs on 8 vertices: Q3 and the prism graph?
        # No, the prism graph C4 x K2 is Q3.
        # Are there others?
        # The number of 3-regular bipartite graphs on 4+4 vertices is small.
        
        # Let's just use the fact that we are adding/removing edges one by one.
        # In the "remove" phase, we find ALL Q3s in the current graph.
        # We can do this by iterating subsets and checking.
        # To make it fast, we can prune.
        
        # Let's implement a fast check:
        # 1. If the subgraph has fewer than 12 edges, return False.
        # 2. If the subgraph is not bipartite, it can still contain Q3.
        # 3. Try all 70 partitions into A, B of size 4.
        #    For a fixed A, B:
        #    Check if there is a bijection f: A->Part1, B->Part2 of Q3
        #    such that edges match.
        #    This is a bipartite matching problem? No, it's isomorphism.
        #    But since Q3 is 3-regular, each vertex in A must connect to exactly 3 in B.
        #    So the subgraph induced by A, B must be 3-regular?
        #    No, it must *contain* the 3-regular structure.
        #    So for each a in A, it must have at least 3 neighbors in B.
        #    If any a has < 3 neighbors in B, this partition fails.
        #    If all have >= 3, we can try to match.
        
        # This pruning is strong.
        
        pass

    # Given time constraints, I will implement the "Random Start + Remove + Add" strategy
    # with a reasonably fast Q3 check.

    def has_q3_in_subset(adj, subset):
        # subset is a tuple of 8 vertices
        # Check if G[subset] contains Q3
        # 1. Count edges in subset
        e_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    e_count += 1
        if e_count < 12:
            return False
            
        # 2. Try all 70 ways to split into A (size 4) and B (size 4)
        #    A is the first 4 indices of the permutation? No, we iterate combinations of 4 from 8.
        for A_idx in itertools.combinations(range(8), 4):
            B_idx = tuple(i for i in range(8) if i not in A_idx)
            A_verts = [subset[i] for i in A_idx]
            B_verts = [subset[i] for i in B_idx]
            
            # Check degrees
            valid = True
            for a in A_verts:
                deg = sum(1 for b in B_verts if adj[a][b])
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            for b in B_verts:
                deg = sum(1 for a in A_verts if adj[a][b])
                if deg < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            # If degrees are ok, check if there is an isomorphism
            # This is a small CSP.
            # We can just brute force the 4! = 24 mappings of A to Q3's Part 1
            # and check if B follows.
            
            # Q3 Part 1: 0, 3, 5, 6 (even parity? 000, 011, 101, 110)
            # Q3 Part 2: 1, 2, 4, 7 (odd parity? 001, 010, 100, 111)
            # Edges from 0: 1, 2, 4
            # Edges from 3: 1, 2, 7
            # Edges from 5: 1, 4, 7
            # Edges from 6: 2, 4, 7
            
            # Map A_verts to [0, 3, 5, 6]
            for perm in itertools.permutations([0, 3, 5, 6]):
                # perm[i] is the Q3 vertex for A_verts[i]
                # Check if edges from A_verts to B_verts match
                # For each a_idx, a_vert in enumerate(A_verts):
                #   q3_v = perm[a_idx]
                #   neighbors_in_q3 = set of B-side Q3 vertices connected to q3_v
                #   neighbors_in_graph = set of B_verts connected to a_vert
                #   The mapping of B_verts must be consistent.
                
                # Let's build the required adjacency from A to B in Q3
                req_adj = {}
                for i, a_v in enumerate(A_verts):
                    q3_v = perm[i]
                    # Get neighbors in Q3 Part 2
                    nbrs = []
                    for b_q3 in [1, 2, 4, 7]:
                        if (q3_v, b_q3) in q3_edges or (b_q3, q3_v) in q3_edges:
                            nbrs.append(b_q3)
                    req_adj[a_v] = set(nbrs)
                    
                # Now we need to map B_verts to [1, 2, 4, 7]
                # such that for all a in A, b in B:
                # adj[a][b] is True iff b's q3 label is in req_adj[a]
                
                # Try all 24 permutations of B
                for b_perm in itertools.permutations([1, 2, 4, 7]):
                    ok = True
                    for i, b_v in enumerate(B_verts):
                        b_q3_label = b_perm[i]
                        for a_v in A_verts:
                            is_edge = adj[a_v][b_v]
                            should_be_edge = b_q3_label in req_adj[a_v]
                            if is_edge != should_be_edge:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        return True
        return False

    # Main logic
    best_edges = 0
    best_adj = None
    
    start_time = time.time()
    timeout = 110  # Safety margin
    
    for trial in range(100):
        if time.time() - start_time > timeout:
            break
            
        # 1. Generate random graph
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    adj[i][j] = adj[j][i] = True
                    
        # 2. Remove edges to eliminate Q3
        # Find all Q3s
        q3s_found = []
        for subset in all_subsets:
            if has_q3_in_subset(adj, subset):
                q3s_found.append(subset)
                
        # Greedy removal
        while q3s_found:
            if time.time() - start_time > timeout:
                break
            subset = q3s_found.pop()
            # Remove a random edge from this subset
            # Find an edge in this subset
            edge_removed = False
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = subset[i], subset[j]
                    if adj[u][v]:
                        adj[u][v] = adj[v][u] = False
                        edge_removed = True
                        break
                if edge_removed:
                    break
            
            # Re-check for Q3s in affected subsets?
            # Simpler: just re-scan all subsets. It's fast enough.
            q3s_found = []
            for sub in all_subsets:
                if has_q3_in_subset(adj, sub):
                    q3s_found.append(sub)
            
        # 3. Try to add edges back
        # Iterate random non-edges
        for _ in range(500):
            if time.time() - start_time > timeout:
                break
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or adj[u][v]:
                continue
            
            # Tentatively add
            adj[u][v] = adj[v][u] = True
            
            # Check if creates Q3
            # Only need to check subsets containing u and v
            # Precomputed list of subsets containing u and v?
            # Just iterate all_subsets and filter
            conflict = False
            for subset in all_subsets:
                if u in subset and v in subset:
                    if has_q3_in_subset(adj, subset):
                        conflict = True
                        break
            
            if conflict:
                adj[u][v] = adj[v][u] = False
            else:
                # Keep edge
                pass

        # Count edges
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
            
    # Construct result
    edges = []
    if best_adj:
        for i in range(n):
            for j in range(i+1, n):
                if best_adj[i][j]:
                    edges.append([i, j])
    else:
        # Fallback to seed
        edges = [[a, b] for a in range(3) for b in range(3, 16)]
        
    return edges
