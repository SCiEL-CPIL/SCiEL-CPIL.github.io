import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 280.0  # 300s limit, 20s buffer

    def time_left():
        return deadline - time.time()

    # 1. Construct the seed Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_init = q
    current_n = n_init
    
    # Adjacency as bitmasks
    # graph_masks[i] is an integer where bit j is 1 if (i,j) is an edge
    graph_masks = [0] * n_init
    edges_init = []
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if (j - i) % q in residues:
                graph_masks[i] |= (1 << j)
                graph_masks[j] |= (1 << i)
                edges_init.append((i, j))
    
    all_vertices_mask = (1 << current_n) - 1
    
    # Helper to get list of neighbors from mask
    def mask_to_list(mask):
        return [i for i in range(current_n) if (mask >> i) & 1]

    # Precompute triangles in G and complement
    # A triangle is a set of 3 vertices {a,b,c} such that all 3 edges exist.
    def compute_triangles(graph_masks, n, complement=False):
        triangles = []
        # Iterate over pairs (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                if complement:
                    # Edge in complement means NO edge in G
                    if not ((graph_masks[i] >> j) & 1):
                        # Common neighbors in complement are common non-neighbors in G
                        # i.e., vertices k such that (i,k) is NOT in G and (j,k) is NOT in G
                        # mask_i = graph_masks[i]
                        # mask_j = graph_masks[j]
                        # common_non_neighbors = ~(mask_i | mask_j) & all_vertices_mask
                        # But we only care about k > j to avoid duplicates? 
                        # Actually, we just need to find k.
                        common_non_neighbors = ~(graph_masks[i] | graph_masks[j]) & all_vertices_mask
                        # Clear i and j bits
                        common_non_neighbors &= ~(1 << i)
                        common_non_neighbors &= ~(1 << j)
                        # Iterate k in common_non_neighbors
                        k = common_non_neighbors
                        while k:
                            lsb = k & -k
                            k_idx = lsb.bit_length() - 1
                            if k_idx > j: # Ensure i < j < k
                                triangles.append((1 << i) | (1 << j) | (1 << k_idx))
                            k ^= lsb
                else:
                    if (graph_masks[i] >> j) & 1:
                        common_neighbors = graph_masks[i] & graph_masks[j]
                        common_neighbors &= ~(1 << i)
                        common_neighbors &= ~(1 << j)
                        k = common_neighbors
                        while k:
                            lsb = k & -k
                            k_idx = lsb.bit_length() - 1
                            if k_idx > j:
                                triangles.append((1 << i) | (1 << j) | (1 << k_idx))
                            k ^= lsb
        return triangles

    def compute_k4s(graph_masks, n, triangles, complement=False):
        k4s = []
        # A K4 is a triangle {a,b,c} plus a vertex d connected to all 3.
        for tri in triangles:
            # Extract vertices from tri mask
            a, b, c = 0, 0, 0
            t = tri
            while t:
                lsb = t & -t
                idx = lsb.bit_length() - 1
                if a == 0: a = idx
                elif b == 0: b = idx
                else: c = idx
                t ^= lsb
            
            if complement:
                # d must be connected to a,b,c in complement (i.e., NOT in G)
                # d must NOT be in {a,b,c}
                # d must be such that (d,a) not in G, (d,b) not in G, (d,c) not in G
                # This is equivalent to d being a common non-neighbor of a,b,c in G
                # mask_a, mask_b, mask_c
                common_non_neighbors = ~(graph_masks[a] | graph_masks[b] | graph_masks[c]) & all_vertices_mask
                common_non_neighbors &= ~(1 << a)
                common_non_neighbors &= ~(1 << b)
                common_non_neighbors &= ~(1 << c)
                d = common_non_neighbors
                while d:
                    lsb = d & -d
                    d_idx = lsb.bit_length() - 1
                    # Ensure d > c to avoid duplicates? 
                    # The triangle was generated with i<j<k. 
                    # If d > k, then {a,b,c,d} is a new K4 with max vertex d.
                    # If d < k, it would have been counted when the triangle was formed differently?
                    # No, we iterate triangles. A K4 has 4 triangles.
                    # To avoid duplicates, we can require d > max(a,b,c).
                    if d_idx > c:
                        k4s.append(tri | (1 << d_idx))
                    d ^= lsb
            else:
                common_neighbors = graph_masks[a] & graph_masks[b] & graph_masks[c]
                common_neighbors &= ~(1 << a)
                common_neighbors &= ~(1 << b)
                common_neighbors &= ~(1 << c)
                d = common_neighbors
                while d:
                    lsb = d & -d
                    d_idx = lsb.bit_length() - 1
                    if d_idx > c:
                        k4s.append(tri | (1 << d_idx))
                    d ^= lsb
        return k4s

    triangles_G = compute_triangles(graph_masks, current_n, False)
    triangles_comp = compute_triangles(graph_masks, current_n, True)
    k4s_G = compute_k4s(graph_masks, current_n, triangles_G, False)
    k4s_comp = compute_k4s(graph_masks, current_n, triangles_comp, True)
    
    current_edges = edges_init

    # Local search for extension
    def find_extension():
        if time.time() > deadline:
            return None
            
        # We want to find a mask A (neighbors of new vertex)
        # such that no K4 in G is subset of A
        # and no K4 in Comp is subset of B = ~A
        
        for restart in range(2000):
            if time.time() > deadline:
                return None
                
            # Random initialization
            # Bias towards half-and-half
            A = 0
            for i in range(current_n):
                if random.random() < 0.5:
                    A |= (1 << i)
            
            # Local search / Hill climbing
            # We measure "validity" as 0 if valid, else count of violations?
            # Or just random flips until valid.
            
            for step in range(500):
                if time.time() > deadline:
                    return None
                
                # Check validity
                valid = True
                # Check K4s in G
                for mask in k4s_G:
                    if (mask & A) == mask:
                        valid = False
                        break
                if not valid:
                    # Flip a bit in one of the violating K4s
                    # We need to find which K4 violated.
                    # To speed up, we can just pick a random K4 that violates?
                    # Or just flip a random bit in A?
                    # Better: identify violation.
                    for mask in k4s_G:
                        if (mask & A) == mask:
                            # Pick a random vertex in this K4 and remove from A
                            # This reduces the chance of this K4 being a subset
                            v_list = mask_to_list(mask)
                            v = random.choice(v_list)
                            A &= ~(1 << v)
                            break
                    continue # Re-check
                    
                # Check K4s in Comp
                B = all_vertices_mask ^ A
                for mask in k4s_comp:
                    if (mask & B) == mask:
                        valid = False
                        break
                if not valid:
                    for mask in k4s_comp:
                        if (mask & B) == mask:
                            # Pick a random vertex in this K4 and ADD to A (remove from B)
                            v_list = mask_to_list(mask)
                            v = random.choice(v_list)
                            A |= (1 << v)
                            break
                    continue # Re-check
                
                if valid:
                    return A

                # Random flip to escape
                v = random.randint(0, current_n - 1)
                A ^= (1 << v)
                
        return None

    max_target = 42
    while current_n < max_target:
        if time.time() > deadline:
            break
            
        extension = find_extension()
        if extension is not None:
            new_v = current_n
            # Update graph_masks
            graph_masks.append(extension)
            for u in range(current_n):
                if (extension >> u) & 1:
                    graph_masks[u] |= (1 << new_v)
            
            # Update edges
            for u in mask_to_list(extension):
                current_edges.append((min(u, new_v), max(u, new_v)))
                
            # Update all_vertices_mask
            all_vertices_mask = (1 << (current_n + 1)) - 1
            
            # Update Triangles
            # New triangles in G involving new_v:
            # Pairs (a,b) in extension such that (a,b) is an edge in G
            ext_list = mask_to_list(extension)
            for i in range(len(ext_list)):
                a = ext_list[i]
                for j in range(i + 1, len(ext_list)):
                    b = ext_list[j]
                    if (graph_masks[a] >> b) & 1:
                        triangles_G.append((1 << new_v) | (1 << a) | (1 << b))
            
            # New triangles in Comp involving new_v:
            # Pairs (a,b) NOT in extension such that (a,b) is NOT an edge in G
            non_ext_mask = all_vertices_mask ^ extension
            non_ext_list = mask_to_list(non_ext_mask)
            for i in range(len(non_ext_list)):
                a = non_ext_list[i]
                for j in range(i + 1, len(non_ext_list)):
                    b = non_ext_list[j]
                    if not ((graph_masks[a] >> b) & 1):
                        triangles_comp.append((1 << new_v) | (1 << a) | (1 << b))
                        
            # Update K4s
            # New K4s in G involving new_v:
            # Old triangles T in G such that T is subset of extension
            for tri in triangles_G:
                # Note: triangles_G now includes new triangles with new_v.
                # We only want OLD triangles.
                # The new triangles have bit new_v set.
                if (tri >> new_v) & 1:
                    continue
                if (tri & extension) == tri:
                    k4s_G.append(tri | (1 << new_v))
                    
            # New K4s in Comp involving new_v:
            # Old triangles T in Comp such that T is subset of non_ext_mask
            for tri in triangles_comp:
                if (tri >> new_v) & 1:
                    continue
                if (tri & non_ext_mask) == tri:
                    k4s_comp.append(tri | (1 << new_v))
                    
            current_n += 1
        else:
            break

    return current_n, current_edges
