import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s buffer
    
    # 1. Generate Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    
    n = q
    
    # Precompute triangle and independent triangle status for all triples
    # tri_mask[a][b][c] is 1 if a,b,c form a triangle in G
    # ind_tri_mask[a][b][c] is 1 if a,b,c form an independent triangle in G
    # To save memory and time, we can just check on the fly but optimized.
    # Or precompute a list of all triangles and all independent triangles.
    
    all_tris = []
    all_ind_tris = []
    
    # Precompute adjacency matrix for faster lookup? 
    # adj[i] is bitset. (adj[i] >> j) & 1 is fast.
    
    for a in range(q):
        for b in range(a+1, q):
            if not ((adj[a] >> b) & 1): continue
            for c in range(b+1, q):
                if ((adj[a] >> c) & 1) and ((adj[b] >> c) & 1):
                    all_tris.append((a, b, c))
                    
    for a in range(q):
        for b in range(a+1, q):
            if ((adj[a] >> b) & 1): continue
            for c in range(b+1, q):
                if not ((adj[a] >> c) & 1) and not ((adj[b] >> c) & 1):
                    all_ind_tris.append((a, b, c))

    # Helper to check if a specific vertex v is connected to all vertices in a triple (a,b,c)
    def is_connected_to_triple(v, tri):
        a, b, c = tri
        return ((adj[v] >> a) & 1) and ((adj[v] >> b) & 1) and ((adj[v] >> c) & 1)

    def is_not_connected_to_triple(v, tri):
        a, b, c = tri
        return not ((adj[v] >> a) & 1) and not ((adj[v] >> b) & 1) and not ((adj[v] >> c) & 1)

    def calculate_initial_cost(x_mask):
        k4_count = 0
        ind4_count = 0
        verts = [i for i in range(n) if (x_mask >> i) & 1]
        non_verts = [i for i in range(n) if not ((x_mask >> i) & 1)]
        
        # Count K4s in verts
        # A K4 is a triangle + a vertex connected to all 3.
        # Or just iterate combinations of 4.
        # Given we have all_tris, we can count K4s by:
        # For each triangle (a,b,c) in verts, count how many other vertices in verts are connected to a,b,c.
        # This is O(|verts| * #tris_in_verts).
        
        # Simpler: Just iterate all 4-combinations of verts.
        # n=37, max verts 37. 66k combos.
        # It's a one-time cost per restart.
        
        for combo in combinations(verts, 4):
            if ((adj[combo[0]] >> combo[1]) & 1) and \
               ((adj[combo[0]] >> combo[2]) & 1) and \
               ((adj[combo[0]] >> combo[3]) & 1) and \
               ((adj[combo[1]] >> combo[2]) & 1) and \
               ((adj[combo[1]] >> combo[3]) & 1) and \
               ((adj[combo[2]] >> combo[3]) & 1):
                k4_count += 1
                 
        for combo in combinations(non_verts, 4):
            if not ((adj[combo[0]] >> combo[1]) & 1) and \
               not ((adj[combo[0]] >> combo[2]) & 1) and \
               not ((adj[combo[0]] >> combo[3]) & 1) and \
               not ((adj[combo[1]] >> combo[2]) & 1) and \
               not ((adj[combo[1]] >> combo[3]) & 1) and \
               not ((adj[combo[2]] >> combo[3]) & 1):
                ind4_count += 1
                 
        return k4_count + ind4_count

    def try_add_vertex(adj, n):
        # Local search with incremental updates
        # We maintain x_mask, k4_count, ind4_count
        # And we need to be able to update them when flipping a bit.
        
        # Pre-compute for each vertex v, the list of triangles it participates in?
        # No, we need triangles in the *current* set.
        
        # Let's use the precomputed all_tris and all_ind_tris.
        # When flipping v:
        # If v enters x:
        #   New K4s = # of triangles (a,b,c) in x_old such that v is connected to a,b,c.
        #   Lost Ind4s = # of ind-triangles (a,b,c) in ~x_old such that v is NOT connected to a,b,c.
        # If v leaves x:
        #   Lost K4s = # of triangles (a,b,c) in x_old \ {v} such that v is connected to a,b,c.
        #   New Ind4s = # of ind-triangles (a,b,c) in ~x_old \ {v} such that v is NOT connected to a,b,c.
        
        # To make this fast, we can precompute for each v:
        # tri_conn[v] = list of triangles (a,b,c) such that v is connected to a,b,c.
        # ind_tri_conn[v] = list of ind-triangles (a,b,c) such that v is NOT connected to a,b,c.
        
        tri_conn = [[] for _ in range(n)]
        ind_tri_conn = [[] for _ in range(n)]
        
        for tri in all_tris:
            a, b, c = tri
            for v in range(n):
                if v not in tri and is_connected_to_triple(v, tri):
                    tri_conn[v].append(tri)
                    
        for tri in all_ind_tris:
            a, b, c = tri
            for v in range(n):
                if v not in tri and is_not_connected_to_triple(v, tri):
                    ind_tri_conn[v].append(tri)

        # Now, for a given x_mask, we can quickly calculate the cost?
        # k4_count = sum over all triangles in x of (number of other vertices in x connected to it)?
        # No, k4_count is just the number of K4s.
        # A K4 is a triangle plus a vertex.
        # k4_count = 1/4 * sum_{v in x} (number of triangles in x\{v} connected to v)
        # This is O(n * avg_tri_conn_size).
        
        # Let's just use the incremental update in the loop.
        
        best_x = None
        best_cost = float('inf')
        
        # Try a few random starts
        for _ in range(10):
            if time.time() > deadline:
                break
                
            x = random.getrandbits(n)
            
            # Calculate initial cost
            # We can use the formula:
            # k4_count = 0
            # for v in range(n):
            #   if x has v:
            #     for tri in tri_conn[v]:
            #       if x has all a,b,c of tri:
            #         k4_count += 1
            # k4_count /= 4 (since each K4 is counted 4 times, once for each vertex)
            
            k4_count = 0
            ind4_count = 0
            
            for v in range(n):
                if (x >> v) & 1:
                    for tri in tri_conn[v]:
                        a, b, c = tri
                        if ((x >> a) & 1) and ((x >> b) & 1) and ((x >> c) & 1):
                            k4_count += 1
                else:
                    for tri in ind_tri_conn[v]:
                        a, b, c = tri
                        if not ((x >> a) & 1) and not ((x >> b) & 1) and not ((x >> c) & 1):
                            ind4_count += 1
                            
            k4_count //= 4
            ind4_count //= 4
            
            current_cost = k4_count + ind4_count
            
            if current_cost == 0:
                return x
                
            # Local Search
            # We need to update k4_count and ind4_count incrementally.
            # Flip v:
            # If v enters x:
            #   Added K4s = # of tris in tri_conn[v] that are fully in x_old
            #   Lost Ind4s = # of tris in ind_tri_conn[v] that are fully in ~x_old
            # If v leaves x:
            #   Lost K4s = # of tris in tri_conn[v] that are fully in x_old (excluding v, but v not in tri)
            #   New Ind4s = # of tris in ind_tri_conn[v] that are fully in ~x_old
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                    
                v = random.randint(0, n-1)
                old_bit = (x >> v) & 1
                
                # Calculate delta
                if old_bit == 0:
                    # v enters x
                    added_k4 = 0
                    for tri in tri_conn[v]:
                        a, b, c = tri
                        if ((x >> a) & 1) and ((x >> b) & 1) and ((x >> c) & 1):
                            added_k4 += 1
                    added_k4 //= 4 # Wait, each K4 involving v is counted once here?
                    # Yes, for a K4 {v,a,b,c}, the triangle is {a,b,c}. 
                    # v is the 4th vertex. 
                    # We are iterating tri_conn[v] which contains triangles connected to v.
                    # If a,b,c are in x, then {v,a,b,c} is a K4.
                    # Each K4 is identified by its 4th vertex v and the triangle.
                    # So added_k4 is the exact number of new K4s.
                    
                    lost_ind4 = 0
                    for tri in ind_tri_conn[v]:
                        a, b, c = tri
                        if not ((x >> a) & 1) and not ((x >> b) & 1) and not ((x >> c) & 1):
                            lost_ind4 += 1
                    # Each Ind4 {v,a,b,c} is identified by v and triangle {a,b,c}.
                    # So lost_ind4 is the exact number of lost Ind4s.
                    
                    new_cost = current_cost + added_k4 - lost_ind4
                else:
                    # v leaves x
                    lost_k4 = 0
                    for tri in tri_conn[v]:
                        a, b, c = tri
                        if ((x >> a) & 1) and ((x >> b) & 1) and ((x >> c) & 1):
                            lost_k4 += 1
                            
                    added_ind4 = 0
                    for tri in ind_tri_conn[v]:
                        a, b, c = tri
                        if not ((x >> a) & 1) and not ((x >> b) & 1) and not ((x >> c) & 1):
                            added_ind4 += 1
                            
                    new_cost = current_cost - lost_k4 + added_ind4

                if new_cost < current_cost:
                    x ^= (1 << v)
                    current_cost = new_cost
                    if current_cost == 0:
                        return x
                else:
                    # Accept with some probability? Or just reject?
                    # Simple hill climbing is safer for validity.
                    # Let's add a small random walk to escape local minima.
                    if random.random() < 0.1: # 10% chance to accept worse
                         x ^= (1 << v)
                         current_cost = new_cost
                         if current_cost == 0:
                             return x

            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x
                
        return None

    # Iterate adding vertices
    for target_n in range(n + 1, 43):
        if time.time() > deadline:
            break
        added = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            x = try_add_vertex(adj, n)
            if x is not None:
                adj.append(0)
                for i in range(n):
                    if (x >> i) & 1:
                        adj[i] |= (1 << n)
                        adj[n] |= (1 << i)
                n += 1
                added = True
                break
        if not added:
            break
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
                
    return n, edges
