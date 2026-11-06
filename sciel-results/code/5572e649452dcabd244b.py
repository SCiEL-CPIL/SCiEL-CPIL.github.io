import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # --- Fast Q3 Detector using Bitmasks ---
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    # We need to check if there is a subgraph isomorphic to Q3.
    # Q3 vertices can be labeled 0..7 (binary 000..111).
    # Edges in Q3 connect vertices that differ in exactly 1 bit.
    # A graph G contains Q3 if there is an injective mapping f: {0..7} -> V(G)
    # such that (u,v) in E(Q3) implies (f(u), f(v)) in E(G).
    
    # Instead of full isomorphism, we can use the property that Q3 is a 3-regular bipartite graph 
    # with girth 6. 
    # A faster check: Iterate over all 8-subsets of vertices? No, too slow (12870 * 8! is large).
    # Better: Iterate over all possible "centers" v and triples of neighbors (a,b,c) that are independent.
    # This is what the seed did, but we can optimize the inner loop with bitmasks.
    
    def is_q3_free(adj):
        # adj is list of 16 ints
        # For each vertex v, look at its neighbors N(v).
        # We need to find a,b,c in N(v) such that a,b,c are pairwise non-adjacent.
        # Then we need x in N(a)∩N(b)\{v}, y in N(a)∩N(c)\{v}, z in N(b)∩N(c)\{v}.
        # x,y,z must be pairwise non-adjacent.
        # And w in N(x)∩N(y)∩N(z) \ {v,a,b,c,x,y,z} such that w is not adjacent to a,b,c.
        
        for v in range(n):
            # Get neighbors of v
            nv_mask = adj[v]
            # Iterate over triples of neighbors
            # To speed up, we can precompute independent sets in N(v)?
            # N(v) size is at most 15. Triples is ~455.
            
            # Let's just iterate triples of neighbors
            neighbors = [u for u in range(n) if (nv_mask >> u) & 1]
            if len(neighbors) < 3:
                continue
                
            for a, b, c in combinations(neighbors, 3):
                # a,b,c must be pairwise non-adjacent
                if (adj[a] >> b) & 1: continue
                if (adj[a] >> c) & 1: continue
                if (adj[b] >> c) & 1: continue
                
                # Candidates for x: neighbors of a and b, not v, not a,b,c
                # x must be in N(a) ∩ N(b)
                mask_x = adj[a] & adj[b]
                # Remove v, a, b, c
                mask_x &= ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                
                # Candidates for y: neighbors of a and c
                mask_y = adj[a] & adj[c]
                mask_y &= ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                
                # Candidates for z: neighbors of b and c
                mask_z = adj[b] & adj[c]
                mask_z &= ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                
                # Iterate x from mask_x
                while mask_x:
                    x_bit = mask_x & -mask_x
                    x = x_bit.bit_length() - 1
                    mask_x ^= x_bit
                    
                    # x must not be adjacent to y or z (we check this later or filter y,z)
                    # Filter y: y in mask_y, y != x, y not adjacent to x
                    my = mask_y & ~(1 << x)
                    # y not adj to x: my &= ~adj[x]
                    my &= ~adj[x]
                    
                    # Filter z: z in mask_z, z != x, z not adjacent to x
                    mz = mask_z & ~(1 << x)
                    mz &= ~adj[x]
                    
                    # Now iterate y from my
                    while my:
                        y_bit = my & -my
                        y = y_bit.bit_length() - 1
                        my ^= y_bit
                        
                        # y must not be adjacent to z
                        mz_filter = mz & ~(1 << y) & ~adj[y]
                        
                        while mz_filter:
                            z_bit = mz_filter & -mz_filter
                            z = z_bit.bit_length() - 1
                            mz_filter ^= z_bit
                            
                            # We have v,a,b,c,x,y,z. Check pairwise non-adjacency of x,y,z
                            # We already ensured x-y, x-z, y-z non-adjacent in filtering?
                            # my filtered by ~adj[x] -> x-y non-adj.
                            # mz filtered by ~adj[x] -> x-z non-adj.
                            # mz_filter filtered by ~adj[y] -> y-z non-adj.
                            # So x,y,z are pairwise non-adjacent.
                            
                            # Now find w
                            # w in N(x) ∩ N(y) ∩ N(z)
                            mask_w = adj[x] & adj[y] & adj[z]
                            # w must not be v,a,b,c,x,y,z
                            mask_w &= ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                            mask_w &= ~(1 << x) & ~(1 << y) & ~(1 << z)
                            
                            # w must NOT be adjacent to a, b, c
                            # So w is not in N(a) ∪ N(b) ∪ N(c)
                            mask_w &= ~(adj[a] | adj[b] | adj[c])
                            
                            if mask_w:
                                return False # Found Q3
        return True

    # --- Construct Shrikhande Graph ---
    # The Shrikhande graph is a strongly regular graph (16, 6, 2, 2).
    # It can be constructed from the 4x4 grid where (i,j) is connected to (i, j±1) and (i±1, j) mod 4?
    # No, that's the 4x4 grid graph (degree 4).
    # Shrikhande graph is isomorphic to the 4x4 rook graph with diagonals?
    # Actually, the Shrikhande graph is the complement of the 4x4 grid graph?
    # Let's check: 4x4 grid has 16 vertices, degree 4 (corners 2, edges 3, inner 4). Not regular.
    # The Shrikhande graph is regular of degree 6.
    # One construction: Vertices are (i,j) in Z4 x Z4.
    # (i,j) is adjacent to (i+1, j), (i-1, j), (i, j+1), (i, j-1) AND (i+1, j+1), (i-1, j-1)?
    # Let's verify degrees.
    # Standard Shrikhande graph edges:
    # It is the Cayley graph of Z4 x Z4 with generators {(1,0), (0,1), (1,1), (1,3)}?
    # Let's use a known explicit edge list for Shrikhande graph (16, 6, 2, 2).
    
    # A common representation:
    # Vertices 0..15.
    # Edge list for Shrikhande (degree 6):
    shrikhande_edges = [
        (0, 1), (0, 2), (0, 5), (0, 6), (0, 9), (0, 10),
        (1, 3), (1, 4), (1, 5), (1, 8), (1, 9), (1, 11),
        (2, 4), (2, 7), (2, 8), (2, 9), (2, 10), (2, 12),
        (3, 5), (3, 6), (3, 7), (3, 8), (3, 11), (3, 12),
        (4, 6), (4, 7), (4, 8), (4, 11), (4, 12), (4, 13),
        (5, 7), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14),
        (6, 8), (6, 9), (6, 10), (6, 13), (6, 14), (6, 15),
        (7, 9), (7, 10), (7, 11), (7, 14), (7, 15), (7, 12), # Wait, let's just generate it programmatically to be safe
    ]
    
    # Let's generate Shrikhande properly.
    # It is the graph where vertices are (i,j) in Z4xZ4.
    # Adjacency: (i,j) ~ (i',j') if (i-i', j-j') is in {(1,0), (0,1), (1,1), (1,3)} mod 4.
    # Note: (1,3) is same as (-1, 1).
    # Generators: (1,0), (0,1), (1,1), (3,1) which is (-1,1).
    # Let's check if this is symmetric.
    # Inverses: (3,0), (0,3), (3,3), (1,3).
    # Set of generators S = {(1,0), (0,1), (1,1), (3,1)}.
    # Is S symmetric? 
    # (1,0) inv (3,0). (3,0) not in S.
    # So we need to include inverses or use undirected generation.
    
    shrikhande_adj = [0] * n
    gens = [(1,0), (0,1), (1,1), (3,1)]
    for i in range(4):
        for j in range(4):
            v = i * 4 + j
            for di, dj in gens:
                ni = (i + di) % 4
                nj = (j + dj) % 4
                u = ni * 4 + nj
                shrikhande_adj[v] |= (1 << u)
                shrikhande_adj[u] |= (1 << v)
            # Remove self loops if any (none here as no (0,0) in gens)
            shrikhande_adj[v] &= ~(1 << v)
            
    # Verify degree
    degs = [bin(shrikhande_adj[i]).count("1") for i in range(n)]
    if not all(d == 6 for d in degs):
        # Fallback to random dense graph if construction failed
        pass # Should be 6
    
    # Complement of Shrikhande
    full_mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        # Complement: all bits set except self and shrikhande neighbors
        comp_adj[i] = full_mask & ~(1 << i) & ~shrikhande_adj[i]
        
    # Ensure symmetry (should be symmetric)
    for i in range(n):
        for j in range(i+1, n):
            if (comp_adj[i] >> j) & 1:
                comp_adj[j] |= (1 << i)
            else:
                comp_adj[j] &= ~(1 << i)
                
    # Initial graph: Complement of Shrikhande
    # Number of edges in Shrikhande = 16*6/2 = 48? 
    # Wait, (16,6,2,2) means degree 6. Edges = 16*6/2 = 48.
    # Total edges = 120.
    # Complement edges = 120 - 48 = 72.
    # Let's check if this is Q3-free.
    
    current_adj = comp_adj[:]
    
    # If it's not Q3-free, repair it
    if not is_q3_free(current_adj):
        # Remove edges until Q3-free
        # To maximize edges, we should remove edges that are in many Q3s?
        # Simple heuristic: find a Q3, remove a random edge from it.
        # We need a function to FIND a Q3, not just check.
        
        def find_q3(adj):
            for v in range(n):
                nv_mask = adj[v]
                neighbors = [u for u in range(n) if (nv_mask >> u) & 1]
                if len(neighbors) < 3: continue
                for a, b, c in combinations(neighbors, 3):
                    if (adj[a] >> b) & 1: continue
                    if (adj[a] >> c) & 1: continue
                    if (adj[b] >> c) & 1: continue
                    mask_x = adj[a] & adj[b] & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    mask_y = adj[a] & adj[c] & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    mask_z = adj[b] & adj[c] & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                    while mask_x:
                        x_bit = mask_x & -mask_x
                        x = x_bit.bit_length() - 1
                        mask_x ^= x_bit
                        my = mask_y & ~(1 << x) & ~adj[x]
                        mz = mask_z & ~(1 << x) & ~adj[x]
                        while my:
                            y_bit = my & -my
                            y = y_bit.bit_length() - 1
                            my ^= y_bit
                            mz_filter = mz & ~(1 << y) & ~adj[y]
                            while mz_filter:
                                z_bit = mz_filter & -mz_filter
                                z = z_bit.bit_length() - 1
                                mz_filter ^= z_bit
                                mask_w = adj[x] & adj[y] & adj[z]
                                mask_w &= ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                                mask_w &= ~(1 << x) & ~(1 << y) & ~(1 << z)
                                mask_w &= ~(adj[a] | adj[b] | adj[c])
                                if mask_w:
                                    # Found Q3. Return one edge to remove.
                                    # Return edge (v,a) for simplicity? Or random from the 12 edges.
                                    # Let's return (x, w) as it's deep in the structure
                                    return (x, w)
            return None
        
        while time.time() < deadline - 10:
            q3_edge = find_q3(current_adj)
            if q3_edge is None:
                break
            u, v = q3_edge
            if u > v: u, v = v, u
            # Remove edge
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)

    # Now current_adj is Q3-free.
    # Local Search: Try to add edges.
    # Generate list of missing edges
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not ((current_adj[i] >> j) & 1):
                missing_edges.append((i, j))
                
    random.shuffle(missing_edges)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        random.shuffle(missing_edges)
        for u, v in missing_edges:
            if time.time() > deadline - 5: break
            # Try adding (u,v)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if is_q3_free(current_adj):
                improved = True
                # Keep the edge
            else:
                # Remove the edge
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
    # Count edges
    edge_count = 0
    for i in range(n):
        edge_count += bin(current_adj[i]).count("1")
    edge_count //= 2
    
    # Convert to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (current_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
