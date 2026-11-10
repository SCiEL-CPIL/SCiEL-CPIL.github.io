import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    # Precompute all Q3 subgraphs as bitmasks
    # Vertices 0..15
    # Q3 vertices can be represented by 3-bit binary numbers 0..7
    # Edges connect i, j if i xor j is a power of 2 (1, 2, 4)
    
    # Generate all Q3 edge sets on vertices 0..7
    q3_edges_0_7 = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff in [1, 2, 4]:
                q3_edges_0_7.append((i, j))
    
    # q3_edges_0_7 has 12 edges
    
    # We need to map these to any 8 vertices in 0..15
    # There are C(16, 8) = 12870 sets of 8 vertices
    # For each set, there are 8! / 48 = 840 distinct Q3 embeddings
    # We can generate all 8! permutations of 8 vertices and filter by isomorphism?
    # No, just generate all permutations of the 8 chosen vertices into the 8 Q3 slots.
    # To avoid duplicates, we can fix the mapping of one vertex?
    # Actually, we just need the set of edges.
    # Different permutations might yield the same set of edges.
    # We can use a set of frozensets of edges, or a set of bitmasks.
    
    all_q3_bitmasks = set()
    
    # Precompute edge bitmasks for 0..15
    # Edge (u, v) with u < v -> bit index
    # Total edges 120.
    # bit_index = u * 16 + v? No, need unique index.
    # index = u * 15 - (u * (u+1) // 2) + (v - u - 1)?
    # Simpler:
    # Map (u, v) to u*16+v if u<v? No, just use a dict or formula.
    # Let's use a 120-bit integer.
    # Bit k corresponds to edge k in a list of all edges.
    
    edges_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_to_idx[(u, v)] = idx
            edges_list.append((u, v))
            idx += 1
            
    def get_edge_bit(u, v):
        if u > v: u, v = v, u
        return 1 << edge_to_idx[(u, v)]
        
    # Generate Q3 bitmasks
    # Iterate over all combinations of 8 vertices
    for verts in itertools.combinations(range(n), 8):
        # For each permutation of these 8 vertices assigned to 0..7
        # To reduce work, we can note that Q3 has automorphisms.
        # But 840 is small. 12870 * 840 = 10.8M.
        # Let's just iterate all permutations? 8! = 40320.
        # 12870 * 40320 is too big (500M).
        # We need to generate the 840 unique embeddings.
        
        # How to generate 840 unique embeddings?
        # Fix vertex 0 of Q3 to verts[0].
        # Then permute the remaining 7 vertices?
        # 7! = 5040. Still too many?
        # 12870 * 5040 = 65M. Maybe okay in C, slow in Python.
        
        # Alternative:
        # Just iterate all 8! permutations but break early if we see a duplicate bitmask?
        # Or use the fact that we just need the set of edges.
        
        # Let's try a different way:
        # Generate all 12-edge subsets of the 28 edges of K8?
        # No, that's C(28, 12) which is huge.
        
        # Let's stick to permutations but optimize.
        # We can fix the mapping of the first two vertices to reduce symmetry?
        # Q3 is vertex transitive.
        
        # Actually, 10.8M is not that big.
        # Let's just generate the 840 mappings explicitly.
        # The 840 mappings correspond to the cosets of the automorphism group.
        # This is complex to implement correctly quickly.
        
        # Let's just use all 8! permutations and store in a set.
        # The set will have at most 840 elements per vertex set.
        # The overhead of generating 40320 permutations and hashing might be high.
        
        # Optimization:
        # We can generate the edge set for a permutation.
        # perm = [v0, v1, ..., v7] mapping Q3 vertex i to verts[perm[i]]
        # Edges are (verts[perm[i]], verts[perm[j]]) for i,j in q3_edges_0_7
        
        for perm in itertools.permutations(range(8)):
            mask = 0
            for i, j in q3_edges_0_7:
                u = verts[perm[i]]
                v = verts[perm[j]]
                mask |= get_edge_bit(u, v)
            all_q3_bitmasks.add(mask)
            
            # Check time
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break
            
    # Convert to list for faster iteration
    q3_list = list(all_q3_bitmasks)
    
    # Start with K_4,12
    # Vertices 0,1,2,3 in A, 4..15 in B
    G = 0
    for u in range(4):
        for v in range(4, 16):
            G |= get_edge_bit(u, v)
            
    edges_removed = 0
    
    # Greedy removal
    # To speed up, we can shuffle q3_list
    random.shuffle(q3_list)
    
    while time.time() < deadline - 5:
        found = False
        for Q in q3_list:
            if (G & Q) == Q:
                # Found a Q3
                found = True
                # Pick a random edge from Q to remove
                # Get edges from Q
                q_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        bit = get_edge_bit(u, v)
                        if Q & bit:
                            q_edges.append((u, v))
                
                # Remove one randomly
                e_to_remove = random.choice(q_edges)
                G &= ~get_edge_bit(e_to_remove[0], e_to_remove[1])
                edges_removed += 1
                break
                
        if not found:
            break
            
    # Count edges in G
    edge_count = bin(G).count('1')
    
    # Convert to list of edges
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if G & get_edge_bit(u, v):
                result.append([u, v])
                
    return result
