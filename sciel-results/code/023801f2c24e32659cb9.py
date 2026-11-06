import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline for safety
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def count_bits(x):
        return bin(x).count('1')
        
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if adj_mask[u] & (1 << v):
            return False
        # Adding (u,v) creates K4 if there exists an edge (w,x) where w,x are common neighbors of u and v
        common = adj_mask[u] & adj_mask[v] & ~((1 << u) | (1 << v))
        # Check if any pair in common has an edge
        temp = common
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # For each w in common, check neighbors of w in common
            if adj_mask[w] & common:
                return False
            temp -= lsb
        return True

    def largest_triangle_free(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        
        def is_triangle_free_mask(mask):
            # Check if induced subgraph on mask is triangle-free
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                neigh = adj_mask[i] & mask
                # Check if neigh is independent
                temp = neigh
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if adj_mask[v] & neigh:
                        return False
                    temp -= lsb
            return True
            
        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best
            # Upper bound: current_size + count_bits(remaining)
            if current_size + count_bits(remaining) <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: pick vertex with maximum degree in remaining to fail fast
            max_deg = -1
            v = None
            for i in range(n):
                if remaining & (1 << i):
                    deg = count_bits(adj_mask[i] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        v = i
                        
            if v is None:
                return
            
            bit_v = 1 << v
            
            # Option 1: Exclude v
            branch_and_bound(remaining & ~bit_v, current_mask, current_size)
            
            # Option 2: Include v
            # Check if adding v to current_mask creates a triangle
            # v creates a triangle if there is an edge (a,b) in current_mask 
            # such that a and b are both neighbors of v.
            neigh_in_current = adj_mask[v] & current_mask
            creates_triangle = False
            temp = neigh_in_current
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if adj_mask[a] & neigh_in_current:
                    creates_triangle = True
                    break
                temp -= lsb
            
            if not creates_triangle:
                branch_and_bound(remaining & ~bit_v, current_mask | bit_v, current_size + 1)
        
        branch_and_bound(all_vertices_mask, 0, 0)
        return best

    def find_max_triangle_free_set(adj_mask, target_size):
        """Find a triangle-free set of size target_size."""
        # This is a helper to find a specific set for local search
        # We can use a simple backtracking or just pick one found during B&B
        # For simplicity, let's just find one by checking combinations if target is small
        # Or modify B&B to return the set.
        
        best_set = 0
        found = False
        
        def is_triangle_free_mask(mask):
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                neigh = adj_mask[i] & mask
                temp = neigh
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if adj_mask[v] & neigh:
                        return False
                    temp -= lsb
            return True

        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best_set, found
            if found:
                return
            if current_size + count_bits(remaining) < target_size:
                return
            if current_size == target_size:
                if is_triangle_free_mask(current_mask):
                    best_set = current_mask
                    found = True
                    return
            
            # Pick a vertex
            v = None
            for i in range(n):
                if remaining & (1 << i):
                    v = i
                    break
            if v is None:
                return
            
            bit_v = 1 << v
            
            # Include v
            neigh_in_current = adj_mask[v] & current_mask
            creates_triangle = False
            temp = neigh_in_current
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if adj_mask[a] & neigh_in_current:
                    creates_triangle = True
                    break
                temp -= lsb
            
            if not creates_triangle:
                branch_and_bound(remaining & ~bit_v, current_mask | bit_v, current_size + 1)
                
            # Exclude v
            if not found:
                branch_and_bound(remaining & ~bit_v, current_mask, current_size)

        branch_and_bound(all_vertices_mask, 0, 0)
        return best_set if found else 0

    best_score = n
    best_adj_mask = [0] * n
    
    # Initialize with a dense random K4-free graph
    # Start with a random graph and remove edges to make it K4-free?
    # Or start empty and add edges.
    # Let's try starting with a random graph of density 0.5 and checking K4-free.
    # If not K4-free, remove edges from K4s.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        adj_mask = [0] * n
        
        # Random initialization
        for u, v in all_pairs:
            if random.random() < 0.4:
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        current_score = largest_triangle_free(adj_mask)
        
        if current_score < best_score:
            best_score = current_score
            best_adj_mask = adj_mask[:]
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            if current_score <= 9: # Good enough
                break
                
            # Find a max triangle-free set
            target_set_mask = find_max_triangle_free_set(adj_mask, current_score)
            if target_set_mask == 0:
                break
                
            # Get list of vertices in target_set_mask
            target_vertices = []
            for i in range(n):
                if target_set_mask & (1 << i):
                    target_vertices.append(i)
            
            # Try adding edges within target_set
            found_improvement = False
            for i in range(len(target_vertices)):
                for j in range(i + 1, len(target_vertices)):
                    u = target_vertices[i]
                    v = target_vertices[j]
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v):
                            # Add edge
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            
                            # Check new score
                            new_score = largest_triangle_free(adj_mask)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                                found_improvement = True
                                break
                            else:
                                # Revert
                                adj_mask[u] &= ~(1 << v)
                                adj_mask[v] &= ~(1 << u)
                if found_improvement:
                    break
                    
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask[:]
                
            if not improved:
                # Perturbation: Remove 1-3 random edges
                adj_mask_perturbed = adj_mask[:]
                edges_present = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mask_perturbed[u] & (1 << v):
                            edges_present.append((u, v))
                
                if edges_present:
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if not edges_present:
                            break
                        u, v = random.choice(edges_present)
                        edges_present.remove((u, v))
                        adj_mask_perturbed[u] &= ~(1 << v)
                        adj_mask_perturbed[v] &= ~(1 << u)
                    
                    # Run local search on perturbed
                    improved2 = True
                    while improved2 and time.time() < deadline:
                        improved2 = False
                        current_score2 = largest_triangle_free(adj_mask_perturbed)
                        
                        target_set_mask2 = find_max_triangle_free_set(adj_mask_perturbed, current_score2)
                        if target_set_mask2 == 0:
                            break
                            
                        target_vertices2 = []
                        for i in range(n):
                            if target_set_mask2 & (1 << i):
                                target_vertices2.append(i)
                                
                        for i in range(len(target_vertices2)):
                            for j in range(i + 1, len(target_vertices2)):
                                u = target_vertices2[i]
                                v = target_vertices2[j]
                                if not (adj_mask_perturbed[u] & (1 << v)):
                                    if can_add_edge(adj_mask_perturbed, u, v):
                                        adj_mask_perturbed[u] |= (1 << v)
                                        adj_mask_perturbed[v] |= (1 << u)
                                        
                                        new_score2 = largest_triangle_free(adj_mask_perturbed)
                                        if new_score2 < current_score2:
                                            current_score2 = new_score2
                                            improved2 = True
                                            break
                                        else:
                                            adj_mask_perturbed[u] &= ~(1 << v)
                                            adj_mask_perturbed[v] &= ~(1 << u)
                            if improved2:
                                break
                            
                        if current_score2 < best_score:
                            best_score = current_score2
                            best_adj_mask = adj_mask_perturbed[:]
                            
                        if not improved2:
                            adj_mask = adj_mask_perturbed[:]
                            break
                            
    # Final polish
    if time.time() < deadline:
        adj_mask = best_adj_mask[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            for u, v in all_pairs:
                if not (adj_mask[u] & (1 << v)):
                    if can_add_edge(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        
                        new_score = largest_triangle_free(adj_mask)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_adj_mask = adj_mask[:]
                        
            if not improved:
                # Try removing edges
                edges_present = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mask[u] & (1 << v):
                            edges_present.append((u, v))
                
                if edges_present:
                    u, v = random.choice(edges_present)
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
                    improved2 = True
                    while improved2 and time.time() < deadline:
                        improved2 = False
                        current_score2 = largest_triangle_free(adj_mask)
                        
                        for u2, v2 in all_pairs:
                            if not (adj_mask[u2] & (1 << v2)):
                                if can_add_edge(adj_mask, u2, v2):
                                    adj_mask[u2] |= (1 << v2)
                                    adj_mask[v2] |= (1 << u2)
                                    
                                    new_score2 = largest_triangle_free(adj_mask)
                                    if new_score2 < current_score2:
                                        current_score2 = new_score2
                                        improved2 = True
                                        break
                                    else:
                                        adj_mask[u2] &= ~(1 << v2)
                                        adj_mask[v2] &= ~(1 << u2)
                                
                                if current_score2 < best_score:
                                    best_score = current_score2
                                    best_adj_mask = adj_mask[:]
                                    
                        if not improved2:
                            break

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
