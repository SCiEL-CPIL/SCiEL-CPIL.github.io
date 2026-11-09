import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    n = 16
    # Initialize with a random graph
    # We want to maximize edges, so start somewhat dense
    # But ensure no Q3 initially? Or just start random and repair.
    
    # Let's start with the known good seed: K3,13 + K3 on part A
    # Edges: (0,1), (0,2), (1,2) and all (a,b) for a in [0,1,2], b in [3..15]
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if adj[u][v]: return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u,v), max(u,v)))
        
    def remove_edge(u, v):
        if not adj[u][v]: return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u,v), max(u,v)))
        
    # Initialize with seed structure
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)
    
    best_edges = len(edges)
    best_adj = [row[:] for row in adj]
    best_edge_set = set(edges)
    
    # Q3 Detection:
    # A Q3 is an 8-vertex subgraph isomorphic to the cube.
    # The cube Q3 has 12 edges.
    # We can check for Q3 by looking for induced subgraphs? No, "subgraph isomorphic".
    # This means we look for a set of 8 vertices and a bijection to Q3 vertices such that
    # all 12 edges of Q3 are present in G.
    # Note: It does NOT require the subgraph to be induced. Extra edges are allowed.
    # So we need to find 8 vertices that contain *at least* the edges of Q3.
    
    # Efficient check:
    # Precompute all 8-vertex subsets? C(16,8) = 12870.
    # For each subset, check if it contains a Q3.
    # How to check if a graph on 8 vertices contains Q3?
    # Q3 is bipartite (4,4).
    # We can iterate over all 8! bijections? Too slow.
    # We can iterate over all partitions of the 8 vertices into two sets of 4 (bipartition).
    # There are C(8,4)/2 = 35 such partitions.
    # For a specific bipartition (A, B) of the 8 vertices, check if the bipartite graph between A and B
    # contains K_{4,4} minus... wait.
    # Q3 is the bipartite graph where each vertex in A is connected to 3 vertices in B.
    # Specifically, Q3 is the incidence graph of the Fano plane? No.
    # Q3 is the graph of the 3-dimensional hypercube.
    # Vertices are 3-bit strings. Edges connect strings differing in 1 bit.
    # It is 3-regular.
    # So, for a subset of 8 vertices to contain Q3, there must exist a labeling of the 8 vertices
    # as 000, 001, 010, 011, 100, 101, 110, 111 such that all 12 hypercube edges are present in G.
    
    # Optimization:
    # We can precompute the 12 edges of Q3 in the abstract index space 0..7.
    # q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    # Note: This depends on the labeling.
    # Standard Q3 labeling:
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    # Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    # Total 12.
    
    q3_abstract_edges = [
        (0,1), (0,2), (0,4),
        (1,3), (1,5),
        (2,3), (2,6),
        (3,7),
        (4,5), (4,6),
        (5,7),
        (6,7)
    ]
    
    # To check if a set of 8 vertices S contains Q3:
    # We need to find a permutation p of S (mapping abstract index i to vertex v_i)
    # such that for all (i,j) in q3_abstract_edges, adj[v_i][v_j] == 1.
    
    # This is a subgraph isomorphism problem for a fixed small pattern.
    # We can use backtracking.
    
    def contains_q3_in_subset(vertices):
        # vertices is a tuple/list of 8 distinct vertex IDs
        # Map abstract index 0..7 to these vertices
        # We want to assign vertices to abstract indices 0..7
        # such that all q3_abstract_edges are satisfied.
        
        # To speed up, we can fix vertex 0 to abstract index 0? No, symmetry.
        # But we can just iterate permutations? 8! = 40320.
        # 12870 subsets * 40320 perms * 12 checks = ~6 billion ops. Too slow.
        
        # Better: Backtracking.
        # Order abstract vertices by degree or constraint.
        # All have degree 3.
        # Let's try to assign abstract vertices 0..7 to the set of vertices.
        # We maintain a mapping abstract_idx -> vertex.
        
        # Heuristic: Start with abstract vertex 0.
        # Actually, we can check if the induced subgraph on the 8 vertices has at least 12 edges?
        # If it has < 12 edges, it can't contain Q3.
        edge_count = 0
        v_list = list(vertices)
        for i in range(8):
            for j in range(i+1, 8):
                if adj[v_list[i]][v_list[j]]:
                    edge_count += 1
        if edge_count < 12:
            return False, []
            
        # Now check isomorphism.
        # We can use a simple backtracking search.
        # To make it faster, we can pre-filter based on degrees in the induced subgraph?
        # Q3 is 3-regular. But the induced subgraph can have higher degrees.
        # However, the *image* of Q3 vertices must have degree at least 3 *within the Q3 structure*.
        
        # Let's just do a quick randomized check or a bounded backtracking.
        # Since we are in a loop, we need this to be very fast.
        
        # Alternative: Precompute all Q3s in the complete graph K8?
        # There are many automorphisms.
        
        # Let's try a different approach:
        # Instead of checking every subset of 8 vertices from scratch,
        # we can maintain a list of "candidate Q3s" or just check recently added edges.
        
        # For now, let's implement a fast check for a specific set of 8 vertices.
        # We will use a bitmask or list of neighbors.
        
        # Backtracking solver
        # abstract_nodes = [0, 1, ..., 7]
        # We assign them to vertices in v_list.
        
        # To speed up, sort v_list by degree in the current graph?
        # No, we are checking a specific subset.
        
        # Let's just iterate permutations of v_list for abstract indices 0..7
        # but prune early.
        
        # We'll use itertools.permutations but break early.
        from itertools import permutations
        
        # Optimization: Check if any vertex in v_list has degree < 3 in the induced subgraph?
        # If so, it cannot be part of the Q3 core (since Q3 vertices have degree 3).
        # Wait, the Q3 is a subgraph, not induced. So a vertex in the Q3 must have at least 3 neighbors in the Q3.
        # So if a vertex in v_list has < 3 neighbors *within v_list*, it cannot be in the Q3.
        # If more than 0 vertices have < 3 neighbors, return False.
        
        deg_in_sub = [0]*8
        for i in range(8):
            for j in range(8):
                if i != j and adj[v_list[i]][v_list[j]]:
                    deg_in_sub[i] += 1
                    deg_in_sub[j] += 1 # wait, this counts edges twice if we do this loop?
                    # No, deg_in_sub[i] is the degree of v_list[i] in the induced subgraph.
                    # Let's recalculate cleanly.
        
        deg_in_sub = [0]*8
        for i in range(8):
            for j in range(8):
                if i != j and adj[v_list[i]][v_list[j]]:
                    deg_in_sub[i] += 1
        
        # If any deg < 3, impossible.
        if any(d < 3 for d in deg_in_sub):
            return False, []
            
        # Now, we need to find a bijection f: {0..7} -> {0..7} (indices in v_list)
        # such that for all (u,v) in q3_abstract_edges, adj[v_list[f(u)]][v_list[f(v)]] == 1.
        
        # We can try all 8! permutations?
        # 40320 * 12 checks = ~500k ops per subset.
        # 12870 subsets * 500k = 6.4 billion. Too slow for Python in 100s.
        
        # We need a faster check.
        # We can use the fact that Q3 is bipartite.
        # Q3 has two parts of 4.
        # Let's find all possible bipartitions of the 8 vertices in v_list that *could* support a Q3.
        # A bipartition (A, B) of v_list is valid if:
        # 1. No edges within A, no edges within B? No, Q3 is a subgraph, so edges within A are allowed in G, just not in Q3.
        #    But the Q3 edges must go between A and B.
        # 2. Each vertex in A must have at least 3 neighbors in B (in G).
        # 3. Each vertex in B must have at least 3 neighbors in A (in G).
        # 4. There must exist a 3-regular bipartite subgraph between A and B that is isomorphic to Q3.
        
        # There are C(8,4)/2 = 35 bipartitions.
        # For each bipartition (A, B):
        #   Check if min(deg_A_to_B) >= 3 and min(deg_B_to_A) >= 3.
        #   If yes, check if the bipartite graph between A and B contains Q3.
        #   Since |A|=4, |B|=4, and Q3 is 3-regular on 4+4,
        #   the bipartite graph must contain a 3-regular spanning subgraph.
        #   Actually, Q3 is the *only* 3-regular bipartite graph on 4+4 vertices?
        #   Let's check.
        #   K_{4,4} has 16 edges. Q3 has 12.
        #   Are there other 3-regular bipartite graphs on 4+4?
        #   The complement of a 1-regular bipartite graph (perfect matching) in K_{4,4} is 3-regular.
        #   K_{4,4} has 4! = 24 perfect matchings.
        #   So there are 24 such graphs?
        #   Are they all isomorphic to Q3?
        #   Q3 is the incidence graph of the 3-cube.
        #   Yes, all 3-regular bipartite graphs on 4+4 vertices are isomorphic to Q3?
        #   Let's verify.
        #   A 3-regular bipartite graph on 4+4 is determined by the missing edges (a perfect matching).
        #   The automorphism group of K_{4,4} acts transitively on perfect matchings?
        #   Yes, S4 x S4 acts transitively on perfect matchings.
        #   So yes, any 3-regular bipartite graph on 4+4 vertices is isomorphic to Q3.
        
        # Therefore, the check is:
        # Does there exist a bipartition (A, B) of the 8 vertices such that
        # the bipartite graph between A and B contains a perfect matching M where
        # the edges of G between A and B include all edges of K_{4,4} \ M?
        # Wait. Q3 = K_{4,4} \ M.
        # So we need the edges of G between A and B to be a superset of (K_{4,4} \ M) for some perfect matching M.
        # Equivalently: The edges of G between A and B must NOT miss all edges of any perfect matching?
        # No. We need G[A,B] to contain *at least* the edges of Q3.
        # So G[A,B] must contain K_{4,4} \ M for some M.
        # This means: For every perfect matching M, if G[A,B] misses any edge in M, that's fine?
        # No. We need G[A,B] to contain a specific set of 12 edges.
        # Let E_G be the edges of G between A and B.
        # We need E_G to contain K_{4,4} \ M for some perfect matching M.
        # This is equivalent to: The complement of E_G in K_{4,4} (let's call it E_miss)
        # must be a subset of some perfect matching M.
        # Since M is a perfect matching (4 disjoint edges), E_miss must be a subset of a perfect matching.
        # This implies:
        # 1. E_miss must be a matching (no two edges share a vertex).
        # 2. |E_miss| <= 4.
        
        # So the algorithm for a specific subset of 8 vertices:
        # 1. Iterate over all 35 bipartitions (A, B).
        # 2. Compute E_miss: edges in K_{4,4} between A and B that are NOT in G.
        # 3. Check if E_miss is a matching (no shared vertices).
        # 4. If yes, then G contains Q3 (specifically, the Q3 corresponding to the matching M that extends E_miss).
        #    Wait, if E_miss is a matching, we can extend it to a perfect matching M.
        #    Then K_{4,4} \ M is a subgraph of G?
        #    Yes, because E_miss subset M, so (K_{4,4} \ M) subset (K_{4,4} \ E_miss) = E_G.
        #    So yes.
        
        # This check is very fast!
        # For each bipartition:
        #   Count missing edges.
        #   Check if they form a matching.
        
        for mask in range(1, 1<<8, 2): # Iterate over subsets of size 4?
            # Actually, we can just iterate over combinations of 4 indices for A.
            pass
            
        # Let's use itertools.combinations for A (size 4).
        from itertools import combinations
        indices = list(range(8))
        for A_indices in combinations(indices, 4):
            B_indices = tuple(i for i in indices if i not in A_indices)
            
            # Check if missing edges form a matching
            # Missing edge (i, j) where i in A_indices, j in B_indices
            # such that adj[v_list[i]][v_list[j]] == 0
            
            # Represent A and B as sets of indices for fast lookup?
            # Or just lists.
            A_set = set(A_indices)
            B_set = set(B_indices)
            
            # List of missing edges
            missing_edges = []
            for i in A_indices:
                for j in B_indices:
                    if not adj[v_list[i]][v_list[j]]:
                        missing_edges.append((i, j))
                        
            # Check if missing_edges is a matching
            # No two edges can share a vertex.
            used_vertices = set()
            is_matching = True
            for u, v in missing_edges:
                if u in used_vertices or v in used_vertices:
                    is_matching = False
                    break
                used_vertices.add(u)
                used_vertices.add(v)
                
            if is_matching:
                return True, list(missing_edges)
                
        return False, []
    
    # Main Loop: Local Search
    # We will try to add random missing edges.
    # If it creates a Q3, we remove an edge from the found Q3 (or just revert).
    # To improve, if we revert, we can try to "swap" edges?
    # For now, simple Hill Climbing:
    # 1. Pick a random missing edge (u, v).
    # 2. Add it.
    # 3. Check if any Q3 is created.
    #    - We only need to check Q3s that *use* the new edge (u, v).
    #    - A Q3 containing (u, v) must include u and v.
    #    - We can find all 8-vertex subsets containing u and v.
    #    - C(14, 6) = 3003 subsets.
    #    - For each, check if it contains Q3.
    #    - This is still potentially expensive if done naively.
    
    # Optimization:
    # When we add edge (u, v), we only introduce new Q3s that contain (u, v).
    # We can check if (u, v) is part of a Q3 by looking at the neighborhood.
    # However, the "bipartition check" on subsets is robust.
    # Let's limit the search:
    # Only check subsets that contain u and v.
    
    def check_q3_with_edge(u, v):
        # Find all 6 other vertices from 0..15 excluding u, v
        other_vertices = [i for i in range(n) if i != u and i != v]
        # We need to choose 6 from 14.
        # C(14, 6) = 3003.
        # For each combination of 6, form the set of 8 vertices: {u, v} + combo.
        # Check if this set contains a Q3 that uses edge (u, v).
        # Note: The previous `contains_q3_in_subset` checks if the set contains *any* Q3.
        # If it returns True, we need to know if the Q3 found uses (u, v).
        # Actually, if the set contained a Q3 *before* adding (u, v), it would have been detected earlier.
        # So if `contains_q3_in_subset` returns True *now*, it might be a new Q3 using (u, v)
        # or an old one (if we didn't remove it properly).
        # Assuming we maintain a Q3-free state, any new Q3 must use (u, v).
        
        from itertools import combinations
        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            # Check if this set contains a Q3
            found, _ = contains_q3_in_subset(vertices)
            if found:
                return True
        return False

    # To make it faster, we can cache or use a different strategy.
    # Given the time limit, we will do a limited number of iterations.
    
    iterations = 0
    max_iterations = 5000 # Adjust based on speed
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Find a random missing edge
        # We can maintain a list of missing edges or just pick random u, v
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v or adj[u][v]:
            continue
            
        # Try to add (u, v)
        add_edge(u, v)
        
        # Check for Q3
        if check_q3_with_edge(u, v):
            # Revert
            remove_edge(u, v)
            # Optional: Try to remove an edge from the Q3 to allow other edges?
            # For now, just revert.
        else:
            # Success
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_adj = [row[:] for row in adj]
                best_edge_set = set(edges)
                
    # Return the best found
    return [list(e) for e in best_edge_set]

# Run construct to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res))
