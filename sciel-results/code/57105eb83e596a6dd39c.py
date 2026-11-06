"""Find a dense Q3-free graph on 16 vertices via local search with bitmasks."""
import time
import random

# Deadline guard
_DEADLINE = 140.0  # safety margin before 150s

# Precompute Q3 structure
# Q3 vertices: 0..7, edges between i and i^bit for bit in {0,1,2}
Q3_EDGES = []
for i in range(8):
    for bit in [0, 1, 2]:
        j = i ^ (1 << bit)
        if i < j:
            Q3_EDGES.append((i, j))

# Q3 adjacency as bitmasks for each vertex (within 0..7)
Q3_ADJ = [0] * 8
for i in range(8):
    mask = 0
    for bit in [0, 1, 2]:
        j = i ^ (1 << bit)
        mask |= (1 << j)
    Q3_ADJ[i] = mask

# Q3 degree is 3 for all vertices
# Parity: even parity vertices are 0,3,5,6 (popcount even), odd are 1,2,4,7
Q3_EVEN = [0, 3, 5, 6]
Q3_ODD = [1, 2, 4, 7]


def _check_q3(adj_masks, n=16):
    """Check if graph with adjacency bitmasks has any Q3 subgraph.
    adj_masks[v] is a bitmask of neighbors of v.
    Returns True if Q3-free, False otherwise.
    """
    # For each 8-vertex subset, check if Q3 is a subgraph
    # Use a faster approach: enumerate all 8-subsets, but with early pruning
    
    # Precompute: for each vertex, its neighbor mask
    # For a Q3 to exist, we need 8 vertices where the induced subgraph
    # contains all 12 Q3 edges.
    
    # Strategy: iterate over all ways to choose 4 vertices for the "even" part
    # and 4 for the "odd" part, then check if the bipartite connections
    # match Q3's bipartite structure.
    
    # Actually, let's use a direct approach with bitmasks.
    # For each combination of 8 vertices, check if Q3 embeds.
    # But that's C(16,8)=12870 subsets, and for each we need to check
    # if Q3 is a subgraph.
    
    # Better: use the fact that Q3 is bipartite. 
    # In Q3, each even vertex connects to 3 odd vertices, and vice versa.
    # The bipartite adjacency is a 4x4 matrix with row/col sums 3.
    # Specifically, it's the complement of the identity in a 4x4 grid... 
    # Let me think: Q3 even={0,3,5,6}, odd={1,2,4,7}
    # 0(000) connects to 1(001),2(010),4(100) - not 7(111)
    # 3(011) connects to 1(001),6(110),7(111) - not 2(010)
    # 5(101) connects to 1(001),4(100),7(111) - not 2(010)
    # 6(110) connects to 2(010),4(100),7(111) - not 1(001)
    
    # So the bipartite adjacency (even x odd) is:
    #        1  2  4  7
    # 0:     1  1  1  0
    # 3:     1  0  1  1
    # 5:     1  0  1  1
    # 6:     0  1  1  1
    
    # This is a specific 4x4 0-1 matrix. We need to find if there exist
    # 4 vertices E and 4 vertices O such that the bipartite adjacency
    # between E and O is isomorphic to this matrix.
    
    # Enumerate all 4-subsets for E and all 4-subsets for O (disjoint).
    # That's C(16,4)*C(12,4) = 1820*495 = 900900 pairs. For each, check
    # if the 4x4 bipartite adjacency is isomorphic to Q3's.
    
    # To check isomorphism of the 4x4 bipartite graph: try all 4! = 24
    # permutations of O (or E) and check if the matrix matches.
    # 900900 * 24 = ~21.6M checks, each O(1) with bitmasks. Should be fast.
    
    # Precompute the target bipartite adjacency as bitmasks for each even vertex
    # Target: for even vertex index e (0..3), which odd vertices (0..3) it connects to
    # Even 0 -> odd 0,1,2 (not 3) => mask 0b0111 = 7
    # Even 1 (vertex 3) -> odd 0,2,3 (not 1) => mask 0b1011 = 11
    # Even 2 (vertex 5) -> odd 0,2,3 (not 1) => mask 0b1011 = 11
    # Even 3 (vertex 6) -> odd 1,2,3 (not 0) => mask 0b1110 = 14
    TARGET_MASKS = [7, 11, 11, 14]
    
    # Precompute all permutations of [0,1,2,3]
    import itertools
    all_perms = list(itertools.permutations(range(4)))
    
    # Enumerate all 4-subsets of 0..15
    from itertools import combinations
    four_subsets = list(combinations(range(16), 4))
    
    for E in four_subsets:
        E_set = set(E)
        E_mask = 0
        for v in E:
            E_mask |= (1 << v)
        
        # For each vertex in E, compute which other 4-subset vertices it connects to
        # We need to check against all O subsets disjoint from E
        for O in four_subsets:
            # Check disjoint
            O_mask = 0
            for v in O:
                O_mask |= (1 << v)
            if E_mask & O_mask:
                continue
            
            # Build the 4x4 bipartite adjacency
            # For each even vertex E[i], compute bitmask of which O[j] it connects to
            bip_adj = [0, 0, 0, 0]
            for i in range(4):
                e = E[i]
                adj_e = adj_masks[e]
                for j in range(4):
                    o = O[j]
                    if adj_e & (1 << o):
                        bip_adj[i] |= (1 << j)
            
            # Check if bip_adj is a permutation of TARGET_MASKS
            # Try all 24 permutations of the rows
            found = False
            for perm in all_perms:
                ok = True
                for i in range(4):
                    if bip_adj[perm[i]] != TARGET_MASKS[i]:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if found:
                return False
    
    return True


def construct():
    start_time = time.time()
    n = 16
    
    # Start with K16
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
    
    edge_count = n * (n - 1) // 2  # 120
    
    rng = random.Random(42)
    
    best_adj = adj_masks[:]
    best_count = edge_count
    
    # Local search: hill climbing with random restarts
    # Move: add an edge (if not present) or remove an edge
    # Actually, since we start from K16 and want to remove edges, let's do:
    # Start from K16, greedily remove edges to destroy Q3s.
    # Then try to add edges back if possible.
    
    # Phase 1: Greedy removal from K16
    current_adj = adj_masks[:]
    current_count = edge_count
    
    # Find Q3s and remove edges
    removed_edges = []
    
    max_time = 130.0
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        if _check_q3(current_adj, n):
            break
        
        # Find a Q3
        # Use the same enumeration but stop at first found
        from itertools import combinations
        four_subsets = list(combinations(range(16), 4))
        import itertools
        all_perms = list(itertools.permutations(range(4)))
        TARGET_MASKS = [7, 11, 11, 14]
        
        found_q3_edges = None
        for E in four_subsets:
            if time.time() - start_time > max_time:
                break
            E_set = set(E)
            E_mask = 0
            for v in E:
                E_mask |= (1 << v)
            
            for O in four_subsets:
                O_mask = 0
                for v in O:
                    O_mask |= (1 << v)
                if E_mask & O_mask:
                    continue
                
                bip_adj = [0, 0, 0, 0]
                for i in range(4):
                    e = E[i]
                    adj_e = current_adj[e]
                    for j in range(4):
                        o = O[j]
                        if adj_e & (1 << o):
                            bip_adj[i] |= (1 << j)
                
                for perm in all_perms:
                    ok = True
                    for i in range(4):
                        if bip_adj[perm[i]] != TARGET_MASKS[i]:
                            ok = False
                            break
                    if ok:
                        # Found Q3 with even=E, odd=O, mapping perm
                        # The Q3 edges are:
                        # For each even i, odd j where bip_adj[i] has bit j:
                        # edge (E[i], O[perm[j]]) -- wait, need to be careful
                        # Actually, the mapping is: Q3 even vertex i maps to E[i]
                        # Q3 odd vertex perm[i]... no.
                        # Let me re-think.
                        # TARGET_MASKS[i] is the mask for Q3 even vertex i.
                        # bip_adj[i] is the mask for graph even vertex E[i].
                        # If bip_adj[perm[i]] == TARGET_MASKS[i], then
                        # graph even vertex E[perm[i]] has the same connections
                        # as Q3 even vertex i.
                        # So Q3 even vertex i -> graph E[perm[i]]
                        # Q3 odd vertex j -> graph O[j]
                        # Q3 edge (i, j) exists iff bit j in TARGET_MASKS[i]
                        # Graph edge: (E[perm[i]], O[j])
                        q3_graph_edges = set()
                        for i in range(4):
                            mask = TARGET_MASKS[i]
                            for j in range(4):
                                if mask & (1 << j):
                                    u = E[perm[i]]
                                    v = O[j]
                                    q3_graph_edges.add((min(u,v), max(u,v)))
                        found_q3_edges = list(q3_graph_edges)
                        break
                    if found_q3_edges:
                        break
                if found_q3_edges:
                    break
            if found_q3_edges:
                break
        
        if found_q3_edges is None:
            break
        
        # Remove a random edge from the Q3
        u, v = rng.choice(found_q3_edges)
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        current_count -= 1
        removed_edges.append((u, v))
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
    
    # Phase 2: Try to add edges back
    # For each removed edge, try to add it back if it doesn't create a Q3
    # Also try random edge additions
    
    # Sort removed edges in reverse order (try to re-add last removed first)
    for u, v in reversed(removed_edges):
        if time.time() - start_time > max_time:
            break
        # Try adding edge (u,v)
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        current_count += 1
        
        if not _check_q3(current_adj, n):
            # This edge creates a Q3, remove it again
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_count -= 1
        else:
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
    
    # Phase 3: Random edge addition attempts
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    attempts = 0
    while time.time() - start_time < max_time and attempts < 500:
        u, v = rng.choice(all_edges)
        if current_adj[u] & (1 << v):
            continue  # Edge already present
        attempts += 1
        
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        current_count += 1
        
        if _check_q3(current_adj, n):
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
        else:
            # Remove it
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_count -= 1
    
    # Verify best solution
    if not _check_q3(best_adj, n):
        # Fallback: return a known valid graph
        # Use a simple construction: complete bipartite K8,8 has 64 edges
        # K8,8 is Q3-free? Q3 is bipartite 4+4, so it could be a subgraph.
        # Actually, K8,8 contains Q3 as a subgraph (any 4+4 subset with all edges).
        # So K8,8 is NOT Q3-free.
        
        # Let's use a different fallback: the seed solution
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        mask = best_adj[i]
        for j in range(i + 1, n):
            if mask & (1 << j):
                edges.append([i, j])
    
    return edges
