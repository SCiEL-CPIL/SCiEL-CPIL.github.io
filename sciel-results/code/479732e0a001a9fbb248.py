import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def get_violation(masks):
        """Find one K3,3 violation or return None."""
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                if popcount(common_ij) < 3:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & masks[k]
                    if popcount(common_ijk) >= 3:
                        return (i, j, k, common_ijk)
        return None

    def is_valid(masks):
        return get_violation(masks) is None

    def conflict_graph_init(masks):
        """
        Hypergraph-inspired repair: build a conflict graph where each edge (u,v)
        is a node, and two edges conflict if they participate in a common K3,3 violation.
        Then greedily remove edges to break all violations, prioritizing high-degree conflicts.
        """
        masks = masks[:]
        
        # Build list of all violations
        def find_all_violations(masks):
            violations = []
            for i in range(n):
                for j in range(i+1, n):
                    common_ij = masks[i] & masks[j]
                    if popcount(common_ij) < 3:
                        continue
                    for k in range(j+1, n):
                        common_ijk = common_ij & masks[k]
                        if popcount(common_ijk) >= 3:
                            # Get the 3+ common neighbors
                            neighbors = []
                            m = common_ijk
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                neighbors.append(idx)
                                m ^= lsb
                            violations.append((i, j, k, neighbors[:3]))
            return violations
        
        # Iteratively find and break violations
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            viols = find_all_violations(masks)
            if not viols:
                break
            
            # For each violation (i,j,k, common_neighbors), we need to remove at least one edge
            # from {i,j,k} to one of the common neighbors.
            # Strategy: for each violation, pick the edge that appears in the most other violations.
            
            # Count how many violations each edge is involved in
            edge_violation_count = {}
            for (i, j, k, neighbors) in viols:
                for w in neighbors:
                    for v in [i, j, k]:
                        if v != w:
                            key = (min(v, w), max(v, w))
                            edge_violation_count[key] = edge_violation_count.get(key, 0) + 1
            
            # Remove the edge with highest violation count
            if edge_violation_count:
                best_edge = max(edge_violation_count, key=edge_violation_count.get)
                u, v = best_edge
                if masks[u] & (1 << v):
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                else:
                    # Edge not present? Shouldn't happen, but break
                    break
            else:
                break
        
        return masks

    def try_add_edge(masks, u, v):
        """
        Try to add edge (u,v).
        Returns True if valid, False otherwise.
        Uses incremental check: only triples involving u or v need checking.
        """
        # Tentatively add
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
        valid = True
        
        # Check triples containing u: (u, a, b) where a < b, a != u, b != u
        V_u = [x for x in range(n) if x != u]
        for idx_a in range(len(V_u)):
            a = V_u[idx_a]
            for idx_b in range(idx_a + 1, len(V_u)):
                b = V_u[idx_b]
                common = masks[u] & masks[a] & masks[b]
                if popcount(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check triples containing v but not u: (v, a, b) where a < b, a != v, b != v, and u not in {a,b}
            V_v = [x for x in range(n) if x != v and x != u]
            for idx_a in range(len(V_v)):
                a = V_v[idx_a]
                for idx_b in range(idx_a + 1, len(V_v)):
                    b = V_v[idx_b]
                    common = masks[v] & masks[a] & masks[b]
                    if popcount(common) >= 3:
                        valid = False
                        break
                if not valid:
                    break

        if not valid:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
        
        return True

    def try_remove_edge(masks, u, v):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)

    def final_polish(masks):
        """
        Final high-precision polish: iteratively try to add every missing edge,
        and for each edge that can't be added, try to swap it with a different edge
        to enable it.
        """
        masks = masks[:]
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline - 3:
                        break
                    if not (masks[i] & (1 << j)):
                        if try_add_edge(masks, i, j):
                            improved = True
                        else:
                            # Try swap: remove some existing edge (a,b) and add (i,j)
                            # Only try a few random swaps to save time
                            existing = [(a, b) for a in range(n) for b in range(a+1, n) 
                                       if (masks[a] & (1 << b)) and (a,b) != (i,j)]
                            if existing:
                                # Try up to 5 random swaps
                                for _ in range(5):
                                    if time.time() > deadline - 3:
                                        break
                                    a, b = random.choice(existing)
                                    # Remove (a,b)
                                    masks[a] &= ~(1 << b)
                                    masks[b] &= ~(1 << a)
                                    # Try add (i,j)
                                    if try_add_edge(masks, i, j):
                                        improved = True
                                        break
                                    else:
                                        # Revert: re-add (a,b)
                                        masks[a] |= (1 << b)
                                        masks[b] |= (1 << a)
        return masks

    best_edges = 0
    best_masks = [0] * n
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize with conflict-graph-inspired approach
        masks = [0] * n
        # Random dense start
        p = 0.8
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
                    
        # Use conflict graph repair instead of random repair
        masks = conflict_graph_init(masks)
        
        if not is_valid(masks):
            # Fallback to random repair
            max_repair = 100
            for _ in range(max_repair):
                if time.time() > deadline - 5:
                    break
                viol = get_violation(masks)
                if viol is None:
                    break
                i, j, k, common_mask = viol
                neighbors = []
                m = common_mask
                while len(neighbors) < 3 and m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    neighbors.append(idx)
                    m ^= lsb
                u = random.choice([i, j, k])
                v = random.choice(neighbors)
                try_remove_edge(masks, u, v)
            
            if not is_valid(masks):
                continue
        
        current_edges = sum(bin(m).count('1') for m in masks) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
            
        # SA Phase
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        max_sa_steps = 20000
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                if try_add_edge(masks, u, v):
                    new_edges = sum(bin(m).count('1') for m in masks) // 2
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        # Revert (try_add_edge already reverted if invalid, 
                        # but if valid and we reject due to T, we need to revert manually)
                        # Wait, try_add_edge returns True if valid and leaves edge added.
                        # So if we reject, we must remove it.
                        try_remove_edge(masks, u, v)
                # If try_add_edge returned False, it already reverted.
                
            elif move_type < 0.7:
                # Remove edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                try_remove_edge(masks, u, v)
                new_edges = sum(bin(m).count('1') for m in masks) // 2
                delta = new_edges - current_edges
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                else:
                    # Re-add
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    
            else:
                # Swap
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                # Remove u1,v1
                try_remove_edge(masks, u1, v1)
                # Try add u2,v2
                if try_add_edge(masks, u2, v2):
                    new_edges = sum(bin(m).count('1') for m in masks) // 2
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        # Revert: Remove u2,v2 and Re-add u1,v1
                        try_remove_edge(masks, u2, v2)
                        masks[u1] |= (1 << v1)
                        masks[v1] |= (1 << u1)
                else:
                    # try_add_edge reverted u2,v2. Re-add u1,v1.
                    masks[u1] |= (1 << v1)
                    masks[v1] |= (1 << u1)
                    
            T *= cooling_rate
            
        # Final high-precision polish
        polished = final_polish(masks)
        if is_valid(polished):
            polished_edges = sum(bin(m).count('1') for m in polished) // 2
            if polished_edges > best_edges:
                best_edges = polished_edges
                best_masks = polished[:]

    # Final polish on best solution
    if time.time() < deadline - 2:
        polished = final_polish(best_masks)
        if is_valid(polished):
            polished_edges = sum(bin(m).count('1') for m in polished) // 2
            if polished_edges > best_edges:
                best_edges = polished_edges
                best_masks = polished[:]

    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
