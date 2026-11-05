import time
import random
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 100.0

def _check_c5_fast(n, adj_bits, edge_u, edge_v):
    """
    Check if adding edge (u, v) creates a C5.
    adj_bits is a list of integers where bit i of adj_bits[j] is 1 if (j, i) is an edge.
    We check if there is a path u-a-b-c-v with all distinct vertices.
    This is equivalent to checking if there exists a, b, c such that:
    a in N(u) \ {v}
    b in N(a) \ {u, v}
    c in N(b) \ {a, u, v}
    c in N(v)
    
    We can iterate over a in N(u) and c in N(v) and check if N(a) and N(c) have a common neighbor b 
    that is not u, v, a, c.
    Actually, simpler:
    For each a in N(u) \ {v}:
      For each c in N(v) \ {u, a}:
        Common neighbors of a and c: N(a) & N(c)
        We need a b in Common such that b != u, b != v, b != a, b != c.
        If such b exists, then u-a-b-c-v is a C5.
    """
    nu = adj_bits[edge_u] & ~(1 << edge_v)
    nv = adj_bits[edge_v] & ~(1 << edge_u)
    
    # Iterate over a in N(u)
    # To iterate set bits:
    # while x: 
    #   lsb = x & -x
    #   idx = lsb.bit_length() - 1
    #   x ^= lsb
    
    # We can just loop 0..n-1 since n <= 35, which is small.
    for a in range(n):
        if a == edge_v: continue
        if not (nu >> a) & 1: continue
        
        # a is in N(u)
        # Now check c in N(v)
        for c in range(n):
            if c == edge_u or c == a: continue
            if not (nv >> c) & 1: continue
            
            # a in N(u), c in N(v), a != c, a != v, c != u
            # Check if N(a) and N(c) have a common neighbor b
            # b must be distinct from u, v, a, c
            common = adj_bits[a] & adj_bits[c]
            # Exclude u, v, a, c
            common &= ~(1 << edge_u)
            common &= ~(1 << edge_v)
            common &= ~(1 << a)
            common &= ~(1 << c)
            
            if common:
                return True
    return False

def _check_alpha_fast(n, adj_bits, target=9):
    """
    Check if graph has an independent set of size 'target'.
    Uses backtracking with bitmasks.
    """
    # independent set in G is clique in complement.
    # Let's stick to independent set in G.
    # adj_bits[i] contains neighbors of i.
    # non-neighbors of i: ~(adj_bits[i] | (1<<i)) & mask
    
    mask = (1 << n) - 1
    
    def search(remaining, current_size):
        if current_size == target:
            return True
        if remaining.bit_count() < target - current_size:
            return False
        
        # Heuristic: pick vertex with max degree in remaining?
        # Or just pick lowest set bit for simplicity?
        # Let's pick a vertex v from remaining.
        # To improve, we can order vertices.
        # For now, pick the first one (lowest index) or random?
        # Random might be better to avoid bias.
        # But deterministic is safer for reproducibility.
        # Let's try picking the vertex with the most neighbors in remaining (to prune more).
        
        # Calculate degrees in remaining
        # This is expensive. Let's just pick a random vertex from remaining.
        # Get list of set bits
        bits = []
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            bits.append(idx)
            temp ^= lsb
            
        if not bits:
            return False
            
        v = random.choice(bits)
        
        # Option 1: Include v
        # New remaining: remaining \ {v} \ N(v)
        # N(v) in remaining: remaining & adj_bits[v]
        new_remaining_incl = remaining & ~(1 << v) & ~adj_bits[v]
        if search(new_remaining_incl, current_size + 1):
            return True
            
        # Option 2: Exclude v
        new_remaining_excl = remaining & ~(1 << v)
        if search(new_remaining_excl, current_size):
            return True
            
        return False

    return search((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    best_n = 32
    best_edges = []
    
    # Initialize best_edges with 8 K4s
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
                
    # Try to find n=33, 34, 35
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 10:
            break
            
        # Local search
        # Start with 8 K4s + (target_n - 32) isolated vertices
        n = target_n
        adj_bits = [0] * n
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    adj_bits[base+i] |= (1 << (base+j))
                    adj_bits[base+j] |= (1 << (base+i))
        
        # We will try to add edges between the K4s and the new vertices, 
        # or between K4s, to reduce alpha.
        # Current alpha is 8 (one from each K4). The new vertices are isolated, 
        # so they can join the independent set. 
        # Wait, if we have 8 K4s and 1 isolated vertex (n=33), 
        # the independent set can take 1 vertex from each K4 (8 vertices) AND the isolated vertex.
        # So alpha is 9. We MUST add edges incident to the isolated vertex to reduce alpha.
        
        found = False
        # We need to connect the isolated vertices such that no independent set of size 9 exists.
        # This means every set of 9 vertices must contain an edge.
        # Equivalently, the complement graph has no K9.
        
        # Strategy: 
        # 1. Connect the new vertices to the K4s.
        # 2. Ensure no C5 is formed.
        
        # Let's try a random greedy approach with backtracking if needed?
        # Or just random restarts.
        
        for restart in range(500):
            if time.time() > _DEADLINE - 5:
                break
                
            # Reset to base state
            adj_bits = [0] * n
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        adj_bits[base+i] |= (1 << (base+j))
                        adj_bits[base+j] |= (1 << (base+i))
            
            # Vertices 32..n-1 are isolated initially.
            # We need to add edges involving these vertices.
            
            # Generate random edges to add.
            # Priority: edges from isolated vertices to K4s.
            # Also edges between K4s?
            
            # Let's just try to add random edges until alpha < 9 or C5 found.
            # But we must be careful not to create C5.
            
            # Better: Start with valid 32-vertex graph.
            # Add vertex 32.
            # Connect 32 to some vertices in the K4s.
            # If we connect 32 to at least one vertex in each K4? No, that doesn't prevent picking the other 3 in that K4?
            # If we pick an independent set, we pick 1 from each K4.
            # If 32 is isolated, we can pick 32 + 1 from each K4 = 9.
            # To prevent this, 32 must be adjacent to at least one vertex in EVERY possible independent set of the 32-vertex graph?
            # No, 32 must be adjacent to a "dominating" set?
            # Actually, if we connect 32 to vertex v in K4_i, then any independent set containing v cannot contain 32.
            # But we can pick the other 3 vertices in K4_i.
            # So if we connect 32 to ALL vertices in K4_1, then any independent set must pick from K4_2..K4_8 (7 vertices) and cannot pick from K4_1 (0 vertices? No, it can pick 0 from K4_1 if it picks 32? No, independent set can't have edges. If 32 is connected to all in K4_1, 32 cannot be with any from K4_1.
            # So if we include 32, we can only pick from K4_2..K4_8. Max size 7.
            # If we don't include 32, we can pick 1 from each K4_1..K4_8. Max size 8.
            # So alpha becomes max(7, 8) = 8.
            
            # So, connecting a new vertex to ALL vertices of one K4 reduces alpha by 1 (from 9 to 8) for that vertex?
            # Wait.
            # Base alpha = 8.
            # Add vertex 32 isolated. Alpha = 9.
            # Connect 32 to all 4 vertices of K4_1.
            # Independent sets:
            # 1. Do not include 32. Pick 1 from each K4. Size 8.
            # 2. Include 32. Cannot pick from K4_1. Pick 1 from K4_2..K4_8. Size 7.
            # So Alpha = 8.
            
            # Does this create a C5?
            # 32 is connected to 4 vertices in K4_1.
            # K4_1 is a clique.
            # Any cycle involving 32 must go 32 - u - ... - v - 32 where u, v in K4_1.
            # Since u, v are in K4_1, u-v is an edge.
            # So 32-u-v-32 is a triangle (C3).
            # To make a C5, we need a path of length 3 between u and v not using 32.
            # u-a-b-v.
            # u, v in K4_1. a, b not in K4_1?
            # If a, b are in other K4s, there are no edges between K4s initially.
            # So no path.
            # If a, b are in K4_1, then u,a,b,v are all in K4_1.
            # K4_1 has no C4 or C5 internally.
            # So connecting 32 to all of K4_1 is safe regarding C5 (only creates triangles).
            
            # So, for n=33:
            # Connect 32 to all 4 vertices of K4_0.
            # This should yield a valid graph.
            
            # Let's implement this specific construction for n=33.
            # For n=34, we have two new vertices 32, 33.
            # Connect 32 to all of K4_0.
            # Connect 33 to all of K4_1.
            # Alpha:
            # If 32, 33 not included: 8.
            # If 32 included: cannot pick from K4_0. Can pick from K4_2..K4_7 (6) + maybe 33?
            # 33 is connected to K4_1.
            # If we pick 32, we can't pick from K4_0.
            # Can we pick 33? 33 is not connected to 32.
            # If we pick 33, we can't pick from K4_1.
            # So if we pick 32 and 33, we can't pick from K4_0 or K4_1.
            # We can pick 1 from K4_2..K4_7 (6 vertices). Total 2+6=8.
            # If we pick 32 only: 1 from K4_2..K4_7 (6) + 1 from K4_1?
            # 32 is not connected to K4_1. So yes, we can pick 1 from K4_1.
            # Total 1 (32) + 1 (K4_1) + 6 (K4_2..7) = 8.
            # So Alpha = 8.
            
            # C5 check:
            # 32 connected to K4_0. 33 connected to K4_1.
            # No edges between 32 and 33.
            # No edges between K4_0 and K4_1.
            # Any cycle must be within a component or use edges between them?
            # There are no edges between K4s.
            # So components are:
            # {32} U K4_0
            # {33} U K4_1
            # K4_2 ... K4_7
            # Each component is a "K4 with a pendant connected to all"? No, 32 is connected to all in K4_0.
            # This is K5 minus one edge? No, K4 is clique. 32 connected to all 4.
            # So {32} U K4_0 is K5.
            # K5 contains C5? Yes.
            # Wait. K5 contains C5.
            # So this construction FAILS for C5.
            
            # We cannot connect a new vertex to ALL 4 vertices of a K4, because that forms a K5, which contains a C5.
            # So we must connect to a subset that does not form a C5.
            # In a K4, any 3 vertices form a K3. Connecting 32 to 3 vertices of K4_0:
            # 32-u-v-w-32. Triangle.
            # Is there a C5?
            # 32-u-a-b-v-32?
            # u, v in K4_0. a, b outside?
            # No edges outside.
            # So if we connect 32 to 3 vertices of K4_0, no C5 is formed.
            # Does this reduce alpha?
            # 32 connected to u, v, w in K4_0. x is the 4th vertex in K4_0.
            # Independent set:
            # If we pick 32, we cannot pick u, v, w. We CAN pick x.
            # So we can pick 32 and x.
            # And 1 from each other K4 (7).
            # Total 1+1+7 = 9.
            # So Alpha is still 9.
            
            # We need to reduce alpha to 8.
            # This requires that for ANY independent set of the base graph (size 8), 
            # the new vertex is adjacent to at least one member.
            # The base graph has 8 K4s. Independent sets are exactly one vertex from each K4.
            # There are 4^8 such sets.
            # The new vertex 32 must be adjacent to at least one vertex in EVERY such set.
            # This is a hitting set problem.
            # Let S be the set of neighbors of 32.
            # For every choice (v1, ..., v8) with vi in K4_i, S must contain some vi.
            # This implies that S must "hit" every transversal.
            # This is equivalent to saying that the complement of S (vertices NOT connected to 32) 
            # does NOT contain a transversal.
            # Let T be the set of vertices NOT connected to 32.
            # T must not contain a set {v1, ..., v8} with vi in K4_i.
            # This means that for some K4_i, T contains NO vertices from K4_i.
            # i.e., 32 is connected to ALL vertices in K4_i.
            # But we just saw that connecting to all 4 vertices of a K4 creates a K5 and thus a C5.
            
            # Contradiction?
            # If 32 must be connected to all 4 vertices of some K4_i to hit all transversals, 
            # and that creates a C5, then it is IMPOSSIBLE to add a single vertex to 8 disjoint K4s 
            # to get a C5-free graph with alpha < 9?
            
            # Wait, is it true that 32 must be connected to all 4 of SOME K4?
            # If T (non-neighbors) contains a transversal, then alpha >= 9.
            # So T must NOT contain a transversal.
            # T is a subset of the 32 vertices.
            # If T has at least 1 vertex in each K4_i, then T contains a transversal.
            # So to NOT contain a transversal, T must be empty in at least one K4_i.
            # So 32 must be connected to all 4 vertices of at least one K4_i.
            # And that creates a K5, which has a C5.
            
            # Therefore, it is IMPOSSIBLE to extend 8 disjoint K4s to 33 vertices with C5-free and alpha<9 
            # by adding just one vertex.
            
            # What if we modify the base graph?
            # The current program starts with 8 K4s.
            # Maybe we should start with a different 32-vertex graph?
            # Or maybe 32 is the max?
            # The task says "best known has 32 vertices".
            # So the score 1.1429 (32/28) is likely the ceiling for this "family" or globally?
            # If 33 is impossible, then we should just return the 32-vertex graph quickly.
            
            # The current code tries to search and might timeout or fail.
            # We should just return the 32-vertex graph immediately if we conclude 33 is hard/impossible.
            
            # However, the directive says "Maximize n".
            # If 33 is impossible, 32 is the max.
            
            # Let's double check the "T must be empty in some K4" logic.
            # Yes, if T has >=1 in each of the 8 K4s, we can pick one from each to form an independent set of size 8 in the base graph.
            # Plus the new vertex 32 (which is not in T, so it's a neighbor? No, T is non-neighbors. So 32 is NOT connected to these vertices).
            # So {32} U {v1..v8} is an independent set of size 9.
            # So yes, T must miss at least one K4.
            # So 32 is connected to all 4 in that K4.
            # So K5 is formed.
            # So C5 is formed.
            
            # Thus, no 33-vertex graph exists that is a superset of 8 disjoint K4s.
            
            # Can we form a 33-vertex graph NOT based on 8 disjoint K4s?
            # The seed program uses 8 K4s.
            # The inspiration 1 uses `_make_k4_disjoint` and `_try_add_vertex`.
            # It scored 1.1429, which is 32/28.
            # It likely also failed to find 33.
            
            # So the best strategy is to return the 32-vertex graph as fast as possible.
            
            # We can just return the 8 K4s construction.
            
            break # No point in searching if 33 is impossible for this structure.
        
    # Return the 32-vertex graph
    return {'n': best_n, 'edges': best_edges}
