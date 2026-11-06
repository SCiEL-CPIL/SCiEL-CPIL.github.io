import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Bitmask helpers
    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            mask = adj[u]
            # Only consider v > u to avoid duplicates
            # We can iterate through bits
            v = 0
            while v < n:
                if (mask >> v) & 1:
                    if v > u:
                        edges.append([u, v])
                v += 1
        return edges

    def is_valid_graph(n, adj):
        """Exact check for no K4 and no I6."""
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if common has any edge (triangle in common means K4 with u,v)
                    # Iterate through bits in common
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if (adj[x] & common) & (1 << x):
                            # There is an edge between x and some y in common
                            # Wait, this checks if x is connected to any other in common.
                            # If so, (u, v, x, y) is a K4.
                            return False
        # Check I6 (Independent Set of size 6)
        # This is equivalent to K6 in the complement.
        # We can use a recursive branch and bound or simple backtracking for small n.
        # For n <= 35, a simple recursive check for independent set of size 6 is feasible if optimized.
        
        def find_independent_set(vertices, current_size, target_size):
            if current_size == target_size:
                return True
            # Pruning: if remaining vertices + current_size < target_size, fail
            if len(vertices) + current_size < target_size:
                return False
            
            # Pick a vertex, try to include it
            v = vertices[0]
            rest = vertices[1:]
            
            # Option 1: Include v. We must exclude all neighbors of v.
            non_neighbors = []
            for u in rest:
                if not ((adj[v] >> u) & 1):
                    non_neighbors.append(u)
            if find_independent_set(non_neighbors, current_size + 1, target_size):
                return True
            
            # Option 2: Exclude v
            if find_independent_set(rest, current_size, target_size):
                return True
            
            return False

        if find_independent_set(list(range(n)), 0, 6):
            return False
        return True

    # Fast local cost calculation for Simulated Annealing
    def count_triangles_in_mask(mask, adj):
        """Count triangles in the subgraph induced by mask."""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            neigh = adj[u] & m  # Neighbors of u within the remaining mask (which is mask minus u and previous)
            # Actually, to count triangles in induced subgraph, we sum edges in N(u) induced by mask
            # But we need to be careful not to double count.
            # Standard way: For each u, count edges between neighbors of u that are also in mask.
            # Let's just iterate pairs in mask and check if they are connected and connected to a 3rd?
            # Simpler: Count edges in the induced subgraph of mask.
            # Sum of degrees / 2.
            deg_sum = 0
            mm = mask
            while mm:
                lsb2 = mm & -mm
                v = lsb2.bit_length() - 1
                mm ^= lsb2
                deg_sum += (adj[v] & mask).bit_count()
            return deg_sum // 2
            
    def get_k4_delta(u, v, adj, n):
        """
        Returns the change in K4 count if edge (u,v) is ADDED.
        If edge is removed, the delta is negative of this (assuming no other changes).
        K4s containing (u,v) correspond to triangles in N(u) ∩ N(v).
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        return count_triangles_in_mask(common, adj)

    def get_i6_delta(u, v, adj, n):
        """
        Returns the change in I6 count if edge (u,v) is ADDED.
        Adding an edge (u,v) destroys any independent set containing both u and v.
        Such sets correspond to independent sets of size 4 in the non-neighborhood of {u,v}.
        Non-neighborhood of {u,v} = V \ ({u,v} ∪ N(u) ∪ N(v)).
        Let S = V \ ({u,v} ∪ N(u) ∪ N(v)).
        We need to count independent sets of size 4 in S.
        """
        all_mask = (1 << n) - 1
        # Neighbors of u or v
        neigh_union = adj[u] | adj[v]
        # Remove u and v themselves
        non_neigh = all_mask & ~(neigh_union | (1 << u) | (1 << v))
        
        # Count independent sets of size 4 in non_neigh
        # This is equivalent to K4 in the complement graph induced by non_neigh.
        # Let's implement a fast counter for small sets.
        vertices = []
        m = non_neigh
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
            
        if len(vertices) < 4:
            return 0
            
        count = 0
        n_v = len(vertices)
        # Check all 4-subsets
        for i in range(n_v):
            vi = vertices[i]
            for j in range(i+1, n_v):
                vj = vertices[j]
                if (adj[vi] >> vj) & 1: continue # Edge exists, not independent
                for k in range(j+1, n_v):
                    vk = vertices[k]
                    if (adj[vi] >> vk) & 1 or (adj[vj] >> vk) & 1: continue
                    for l in range(k+1, n_v):
                        vl = vertices[l]
                        if (adj[vi] >> vl) & 1 or (adj[vj] >> vl) & 1 or (adj[vk] >> vl) & 1: continue
                        count += 1
        return count

    def calculate_initial_cost(n, adj):
        """
        Calculate total cost: #K4 + #I6.
        """
        k4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    k4_count += get_k4_delta(u, v, adj, n)
        
        i6_count = 0
        # Sum over all pairs? No, that's not right.
        # An I6 is a set of 6 vertices with no edges.
        # We can count them by iterating? C(35,6) is ~1.6M. Feasible once.
        # Or use the delta logic: 
        # Total I6s = Sum over all pairs (u,v) of (I4s in non-neigh(u,v)) / 15?
        # Each I6 contains C(6,2)=15 pairs.
        # For a specific I6 S, and a pair {u,v} in S, the other 4 vertices are in non-neigh(u,v).
        # So sum_{u<v} I4(non-neigh(u,v)) = 15 * (#I6).
        
        total_i4_in_nonneigh = 0
        for u in range(n):
            for v in range(u+1, n):
                # Only count if u,v are NOT connected? 
                # If u,v are connected, they can't be in an independent set together.
                # So we only need to sum for non-edges.
                if not ((adj[u] >> v) & 1):
                    total_i4_in_nonneigh += get_i6_delta(u, v, adj, n)
        
        i6_count = total_i4_in_nonneigh // 15
        return k4_count + i6_count

    def get_turan_fallback():
        n = 15
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if i // 5 != j // 5:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    best_result = get_turan_fallback()
    best_n = 15
    best_cost = float('inf')

    # Search from high n down
    for target_n in range(35, 19, -1):
        if time.time() > deadline - 5:
            break
            
        found_valid = False
        # Try a few restarts for this n
        max_restarts = 10 if target_n > 30 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = calculate_initial_cost(target_n, adj)
            
            if current_cost == 0:
                if is_valid_graph(target_n, adj):
                    best_n = target_n
                    best_result = {'n': target_n, 'edges': get_edges_from_adj(adj, target_n)}
                    found_valid = True
                    break
            
            # Simulated Annealing
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            steps = 5000 if target_n <= 30 else 2000
            
            for s in range(steps):
                if time.time() > deadline - 5:
                    break
                
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                is_edge = bool((adj[u] >> v) & 1)
                
                # Calculate deltas
                # Note: get_k4_delta and get_i6_delta assume we are ADDING the edge.
                # If we are REMOVING the edge, the delta is -delta_add.
                
                d_k4 = get_k4_delta(u, v, adj, target_n)
                d_i6 = get_i6_delta(u, v, adj, target_n)
                
                if is_edge:
                    # Removing edge
                    # K4s decrease by d_k4
                    # I6s increase by d_i6
                    delta_cost = -d_k4 + d_i6
                else:
                    # Adding edge
                    # K4s increase by d_k4
                    # I6s decrease by d_i6
                    delta_cost = d_k4 - d_i6
                
                # Acceptance
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Apply change
                    if is_edge:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    current_cost += delta_cost
                    
                    if current_cost == 0:
                        # Potential valid graph
                        if is_valid_graph(target_n, adj):
                            best_n = target_n
                            best_result = {'n': target_n, 'edges': get_edges_from_adj(adj, target_n)}
                            found_valid = True
                            break
                
                T = max(T_min, T * cooling)
                
            if found_valid:
                break
        
        if found_valid:
            # Found a valid graph for this n. 
            # Since we iterate n downwards, and we want MAX n, 
            # we can stop here? 
            # Wait, the loop is `for target_n in range(35, 19, -1)`.
            # If we find n=35, we return it.
            # If we don't find 35, we try 34.
            # So we can break the outer loop if we find a valid graph?
            # Yes, because we are going from high to low.
            break

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")
