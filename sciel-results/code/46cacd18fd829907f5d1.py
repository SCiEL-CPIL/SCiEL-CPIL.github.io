import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # We want to find the largest n such that a (5,5)-free graph exists.
    # We know 37 works (Paley).
    # We try 42, 41, 40, 39, 38.
    
    # Helper to generate Paley graph bitmask
    def get_paley_bitmasks(q):
        residues = {(x * x) % q for x in range(1, q)}
        masks = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks

    # Helper to check if graph is valid (5,5)-free
    def is_valid(masks, n):
        # Count K5 and Ind5
        # This is O(n^5) but with bit operations it's faster.
        # Actually, just checking for existence is enough.
        
        # Check K5
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] & (1 << j)): continue
                # Common neighbors of i and j
                common = masks[i] & masks[j]
                # We need a triangle in common to form K5 with i,j
                # Iterate over common neighbors
                # To speed up, we can just check if there are 3 vertices in common that form a triangle.
                # Let's extract list
                verts = []
                c = common
                while c:
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    c ^= lsb
                # Check for triangle in verts
                found_tri = False
                for a_idx in range(len(verts)):
                    a = verts[a_idx]
                    for b_idx in range(a_idx+1, len(verts)):
                        b = verts[b_idx]
                        if not (masks[a] & (1 << b)): continue
                        # Check if there is c in verts connected to a and b
                        common_ab = masks[a] & masks[b] & common
                        if common_ab:
                            found_tri = True
                            break
                    if found_tri: break
                if found_tri:
                    return False # Found K5
        
        # Check Ind5
        for i in range(n):
            for j in range(i+1, n):
                if (masks[i] & (1 << j)): continue # Must be non-adjacent
                # Common non-neighbors of i and j
                # Full set of vertices excluding i, j
                full_mask = (1 << n) - 1
                # Non-neighbors of i: ~(masks[i] | (1<<i))
                # Non-neighbors of j: ~(masks[j] | (1<<j))
                non_neigh_i = full_mask & ~(masks[i] | (1 << i))
                non_neigh_j = full_mask & ~(masks[j] | (1 << j))
                common_non = non_neigh_i & non_neigh_j
                
                verts = []
                c = common_non
                while c:
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    c ^= lsb
                # Check for independent triangle in verts
                found_ind_tri = False
                for a_idx in range(len(verts)):
                    a = verts[a_idx]
                    for b_idx in range(a_idx+1, len(verts)):
                        b = verts[b_idx]
                        if (masks[a] & (1 << b)): continue # Must be non-adjacent
                        # Check if there is c in verts non-adjacent to a and b
                        # Non-neighbors of a in common_non
                        non_a = common_non & ~(masks[a] | (1 << a))
                        non_b = common_non & ~(masks[b] | (1 << b))
                        common_non_ab = non_a & non_b
                        if common_non_ab:
                            found_ind_tri = True
                            break
                    if found_ind_tri: break
                if found_ind_tri:
                    return False # Found Ind5
        return True

    # Simulated Annealing
    def run_sa(n, time_limit):
        # Initialize with Paley 37 if n >= 37
        masks = [0] * n
        if n >= 37:
            paley_masks = get_paley_bitmasks(37)
            for i in range(37):
                masks[i] = paley_masks[i]
            # For vertices 37..n-1, random initialization
            for i in range(37, n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
                # Connect to existing
                for j in range(i):
                    if random.random() < 0.5:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        else:
            # Random init
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        masks[i] |= (1 << j)
                        masks[j] |= (1 << i)
        
        # Calculate initial cost (number of K5 + Ind5)
        # We need a fast cost function for SA steps.
        # Let's use the incremental delta approach.
        
        def get_common_neigh(u, v):
            return masks[u] & masks[v]
            
        def get_common_non_neigh(u, v):
            full_mask = (1 << n) - 1
            non_u = full_mask & ~(masks[u] | (1 << u))
            non_v = full_mask & ~(masks[v] | (1 << v))
            return non_u & non_v

        def count_triangles_in_mask(mask):
            # Count triangles in induced subgraph defined by mask
            verts = []
            c = mask
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c ^= lsb
            
            count = 0
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i+1, len(verts)):
                    v = verts[j]
                    if (masks[u] & (1 << v)):
                        # Check for w
                        common_uv = masks[u] & masks[v] & mask
                        if common_uv:
                            count += 1
            return count

        def count_ind_triangles_in_mask(mask):
            # Count independent triangles in induced subgraph defined by mask
            verts = []
            c = mask
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c ^= lsb
            
            count = 0
            for i in range(len(verts)):
                u = verts[i]
                for j in range(i+1, len(verts)):
                    v = verts[j]
                    if not (masks[u] & (1 << v)):
                        # Check for w non-adjacent to u and v
                        # w must be in mask, non-adj to u, non-adj to v
                        non_u = mask & ~(masks[u] | (1 << u))
                        non_v = mask & ~(masks[v] | (1 << v))
                        common_non = non_u & non_v
                        if common_non:
                            count += 1
            return count

        # Initial Cost
        current_cost = 0
        # This is O(n^5) effectively if we do it naively, but we can sum deltas?
        # No, just calculate once.
        # To calculate total cost:
        # Sum over all pairs (u,v) of edge: triangles in common neighbors.
        # Each K5 is counted 5 times (once for each edge).
        # So Total K5 = (Sum of triangles in common neighbors for all edges) / 5
        # Same for Ind5.
        
        sum_k5_contrib = 0
        sum_ind5_contrib = 0
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] & (1 << v)):
                    sum_k5_contrib += count_triangles_in_mask(get_common_neigh(u, v))
                else:
                    sum_ind5_contrib += count_ind_triangles_in_mask(get_common_non_neigh(u, v))
        
        current_cost = sum_k5_contrib // 5 + sum_ind5_contrib // 5
        
        if current_cost == 0:
            return masks

        T = 1.0
        cooling = 0.999
        best_cost = current_cost
        best_masks = masks[:]
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            edge_exists = (masks[u] & (1 << v)) != 0
            
            # Calculate Delta
            if edge_exists:
                # Remove edge
                # Delta K5 = - (triangles in common neighbors of u,v)
                # Delta Ind5 = + (ind triangles in common non-neighbors of u,v)
                # Note: When we remove edge, u and v become non-adjacent.
                # The common neighbors set for K5 calculation uses the CURRENT graph state.
                # The triangles in common neighbors are those that *currently* form K5s with (u,v).
                tri_count = count_triangles_in_mask(get_common_neigh(u, v))
                
                # For Ind5, we look at common non-neighbors.
                # But wait, if we remove edge, the set of common non-neighbors for the *new* state?
                # Actually, the formula is:
                # Delta Cost = (New Cost) - (Old Cost)
                # If we remove edge (u,v):
                # We lose K5s containing (u,v). Count = triangles in N(u) ∩ N(v).
                # We gain Ind5s containing (u,v). Count = ind triangles in (V \ N[u]) ∩ (V \ N[v])?
                # No, Ind5 containing (u,v) requires u,v non-adjacent.
                # The 3 other vertices must be common non-neighbors of u and v in the NEW graph?
                # No, they must be non-adjacent to u and v.
                # If we remove (u,v), u and v are non-adjacent.
                # The other 3 vertices must be non-adjacent to u, non-adjacent to v, and form an independent triangle.
                # The adjacency of the other 3 vertices to u and v does NOT change when we flip (u,v).
                # So the set of candidates is the same: common non-neighbors in the OLD graph (excluding u,v themselves).
                # Wait, in the old graph, u and v were adjacent.
                # Common non-neighbors of u and v: w such that !adj(u,w) and !adj(v,w).
                # This set is the same before and after flip.
                # So Ind Triangles count is on this set.
                
                ind_tri_count = count_ind_triangles_in_mask(get_common_non_neigh(u, v))
                
                delta_k5 = -tri_count
                delta_ind = ind_tri_count
            else:
                # Add edge
                tri_count = count_triangles_in_mask(get_common_neigh(u, v))
                ind_tri_count = count_ind_triangles_in_mask(get_common_non_neigh(u, v))
                
                delta_k5 = tri_count
                delta_ind = -ind_tri_count
                
            delta_cost = delta_k5 / 5.0 + delta_ind / 5.0
            
            # Acceptance
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
                # Apply
                if edge_exists:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                else:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_masks = masks[:]
                    if best_cost == 0:
                        return best_masks
                        
            T *= cooling
            
        return best_masks

    # Try n=42, 41, 40, 39, 38
    for n in [42, 41, 40, 39, 38]:
        # Allocate time
        time_left = deadline - time.time()
        if time_left < 10: break
        
        # Give more time to higher n
        # 42: 100s, 41: 50s, 40: 20s, 39: 10s, 38: 5s
        # Adjust based on time left
        if n == 42:
            t_alloc = min(120, time_left)
        elif n == 41:
            t_alloc = min(60, time_left)
        elif n == 40:
            t_alloc = min(30, time_left)
        else:
            t_alloc = min(10, time_left)
            
        if t_alloc <= 0: break
        
        masks = run_sa(n, t_alloc)
        
        if masks is not None and is_valid(masks, n):
            # Convert masks to edge list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (masks[i] & (1 << j)):
                        edges.append((i, j))
            return n, edges

    # Fallback: Return Paley 37
    n = 37
    masks = get_paley_bitmasks(n)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (masks[i] & (1 << j)):
                edges.append((i, j))
    return n, edges
