import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    N = 512
    # 0-indexed: 0 represents 1, ..., 511 represents 512
    # Vertices are 0 to N-1 corresponding to values 1 to N
    
    # 1. Build Graph
    # neighbors[i] is a bitmask of neighbors of vertex i
    neighbors = [0] * N
    
    for i in range(N):
        val_i = i + 1
        for j in range(i + 1, N):
            val_j = j + 1
            # lcm(val_i, val_j) <= N
            l = (val_i * val_j) // gcd(val_i, val_j)
            if l <= N:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
    
    # 2. Tomita Algorithm
    best_clique = []
    best_size = 0
    
    # Initial good solution: Divisors of 360 or 480
    # Let's just use a greedy heuristic to get a starting point
    # Greedy: pick vertex with max degree, add to clique, remove non-neighbors
    P = (1 << N) - 1
    current_clique = []
    while P:
        # Find vertex with max degree in P
        # This is O(N^2) if done naively, but N=512 is small
        max_deg = -1
        v = -1
        temp_P = P
        while temp_P:
            # Get lowest set bit
            lb = temp_P & -temp_P
            idx = lb.bit_length() - 1
            deg = (neighbors[idx] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_P ^= lb
        if v == -1:
            break
        current_clique.append(v)
        P &= neighbors[v]
    
    best_size = len(current_clique)
    best_clique = current_clique[:]
    
    # Tomita Search
    def color_sort(P):
        # Returns list of (vertex, color) sorted by color, then vertex
        # Greedy coloring
        colors = {}
        color_classes = []
        available = P
        max_color = 0
        # We need to assign colors to vertices in P
        # Standard greedy: iterate vertices, assign smallest possible color
        
        # To make it fast, we can just assign colors 1..k
        # But we need the bound.
        # Simpler: just count max color assigned.
        
        # Actually, Tomita's original algorithm sorts by color.
        # Let's implement a simple greedy coloring that returns the list of vertices
        # sorted by color class (highest color first for branching).
        
        # For efficiency, we can just return the max color and the list of vertices
        # ordered by color.
        
        # Let's use a list of lists for color classes
        # c[k] contains vertices with color k
        
        # Initialize all vertices in P as uncolored
        uncolored = P
        color_map = {}
        next_color = 1
        
        # To ensure determinism and speed, we can just iterate
        # But we need to sort by color for the branching.
        
        # Alternative: Just return max_color and the list of vertices
        # The branching order in Tomita is reverse color order.
        
        # Let's do a simple greedy coloring:
        # v1, v2, ... vk
        # color(v1) = 1
        # color(v2) = 1 if not adj to v1, else 2
        # etc.
        
        # This requires checking adjacency.
        
        # Since N is small, let's just do it.
        
        vertices_in_P = []
        temp_P = P
        while temp_P:
            lb = temp_P & -temp_P
            idx = lb.bit_length() - 1
            vertices_in_P.append(idx)
            temp_P ^= lb
            
        # Greedy coloring
        # We want to maximize the color bound, so we should order vertices
        # such that high-degree vertices come first? 
        # Standard Tomita orders by degree descending before coloring.
        
        # Sort by degree in P descending
        vertices_in_P.sort(key=lambda x: (neighbors[x] & P).bit_count(), reverse=True)
        
        color = 0
        last_color = 0
        colored_vertices = []
        
        # We need to assign colors such that no two adjacent vertices have the same color.
        # Greedy: for each v in sorted order, assign smallest color not used by neighbors.
        
        # This is O(N^2) per coloring step.
        # With N=512, this might be slow if called many times.
        # But Tomita reduces P quickly.
        
        # Let's try.
        
        color_of = {}
        max_c = 0
        for v in vertices_in_P:
            # Find smallest color not in {color[u] for u in neighbors[v] & P}
            # Since we process in order, only previously colored neighbors matter?
            # No, greedy coloring usually checks all colored neighbors.
            # But if we just want a valid coloring for the bound, any valid coloring works.
            # The bound is the max color used.
            
            used_colors = set()
            nb_mask = neighbors[v] & P
            # Check colors of already processed neighbors
            # To speed up, we can just check all vertices in P that are neighbors?
            # No, only those already assigned a color.
            
            # Actually, a simpler heuristic:
            # Just assign color 1 to all non-adjacent to previous in same class?
            # Let's stick to standard greedy:
            
            for c in range(1, max_c + 2):
                if c not in used_colors:
                    # Check if c is used by any neighbor
                    conflict = False
                    temp_nb = nb_mask
                    while temp_nb:
                        lb = temp_nb & -temp_nb
                        u = lb.bit_length() - 1
                        if u in color_of and color_of[u] == c:
                            conflict = True
                            break
                        temp_nb ^= lb
                    if not conflict:
                        color_of[v] = c
                        if c > max_c:
                            max_c = c
                        break
                    # If conflict, try next color
            # If loop finishes without assigning, assign max_c + 1
            if v not in color_of:
                color_of[v] = max_c + 1
                max_c += 1
            
        # Now sort vertices by color descending, then by vertex index
        # For branching, we want to try vertices with high color first
        sorted_vertices = sorted(vertices_in_P, key=lambda x: (-color_of[x], x))
        
        return sorted_vertices, max_c

    def tomita(P, current_clique):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
            
        sorted_vertices, max_color = color_sort(P)
        
        if len(current_clique) + max_color <= best_size:
            return
            
        # Branch on vertices in sorted order
        # We need to maintain the set of candidates P
        # For each v in sorted_vertices:
        #   if color[v] == max_color_remaining (dynamic bound)
        #   ...
        
        # Standard Tomita:
        # P is the set of candidates.
        # We iterate through vertices in reverse color order.
        # When we pick v, the new candidate set is P & N(v).
        # We must remove v from P for subsequent iterations.
        
        # To do this efficiently with bitmasks:
        # We can just iterate through the list.
        
        for v in sorted_vertices:
            # Check if v is still in P
            # Since we modify P, we need to be careful.
            # The standard approach is to remove v from P after processing.
            
            if not (P & (1 << v)):
                continue
                
            # Dynamic bound:
            # The color of v in the current coloring of P is an upper bound
            # for the size of the clique we can form from P \ {v, ...} + {v}
            # Actually, the bound is len(current_clique) + color[v].
            # But color[v] was calculated for the initial P.
            # As we remove vertices, the bound decreases.
            # A tighter bound is to re-color? No, too slow.
            # Tomita uses the fact that if we process in order, the color of v
            # is an upper bound for the number of vertices in P that can be added
            # after v.
            
            if len(current_clique) + color_sort(P)[1] <= best_size: # This is wrong, color_sort is expensive
                pass # We should use the pre-computed color from the list
            
            # Let's use the color assigned in the initial coloring of P for this call.
            # We need access to the color map.
            # Let's modify color_sort to return the map or just use the sorted list
            # where the index implies the color? No, multiple vertices can have same color.
            
            # Let's re-structure:
            # color_sort returns (sorted_vertices, max_color, color_map)
            
            # For now, let's just use the max_color of the current P as a rough bound?
            # No, that's not tight.
            
            # Let's assume the color of v in the initial coloring is a valid bound.
            # This is correct for Tomita.
            
            # We need the color of v.
            # Let's re-do color_sort to return color_map.
            
            # To avoid re-coding, let's just use the fact that if we process
            # in reverse color order, the color of the current vertex is an upper
            # bound on the number of vertices that can be added from the remaining
            # candidates in P (which is a subset of the original P).
            
            # So, if len(current_clique) + color_of[v] <= best_size, we can prune?
            # Yes.
            
            # But we don't have color_of here. Let's get it.
            
            # This implementation is getting complicated. Let's simplify.
            # For N=512, a simpler branch and bound might suffice.
            
            # Let's try a different approach:
            # Use the "Divisors of K" property.
            # We know the answer is likely around 24-30.
            # We can do a BFS/DFS on the set of divisors of highly composite numbers.
            
            # But let's stick to Tomita.
            
            # I will implement a clean Tomita.
            
            pass

    # Clean Tomita Implementation
    def tomita_clean(P, clique):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Color P
        # Get vertices in P
        verts = []
        temp_P = P
        while temp_P:
            lb = temp_P & -temp_P
            idx = lb.bit_length() - 1
            verts.append(idx)
            temp_P ^= lb
            
        # Sort by degree in P descending
        verts.sort(key=lambda x: (neighbors[x] & P).bit_count(), reverse=True)
        
        # Greedy Coloring
        color_of = {}
        max_c = 0
        for v in verts:
            used = set()
            nb_mask = neighbors[v] & P
            temp_nb = nb_mask
            while temp_nb:
                lb = temp_nb & -temp_nb
                u = lb.bit_length() - 1
                if u in color_of:
                    used.add(color_of[u])
                temp_nb ^= lb
            
            for c in range(1, max_c + 2):
                if c not in used:
                    color_of[v] = c
                    if c > max_c:
                        max_c = c
                    break
            if v not in color_of:
                color_of[v] = max_c + 1
                max_c += 1
        
        if len(clique) + max_c <= best_size:
            return
            
        # Sort by color descending
        verts.sort(key=lambda x: -color_of[x])
        
        # Branch
        for v in verts:
            if not (P & (1 << v)):
                continue
                
            if len(clique) + color_of[v] <= best_size:
                continue
                
            # Add v to clique
            clique.append(v)
            # New P is P & N(v)
            new_P = P & neighbors[v]
            # Remove v from P for next iterations
            P ^= (1 << v)
            
            tomita_clean(new_P, clique)
            
            clique.pop()
            
            if time.time() > deadline:
                return

    tomita_clean((1 << N) - 1, [])
    
    # Convert best_clique (0-indexed) to values (1-indexed)
    result = [v + 1 for v in best_clique]
    
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            l = (result[i] * result[j]) // gcd(result[i], result[j])
            if l > 512:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        return sorted([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
        
    return result
