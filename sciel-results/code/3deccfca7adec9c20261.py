import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4_fast(n, masks):
        """Count K4s using bit operations."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1
        return k4_count

    def count_ind6_fast(n, masks):
        """Count independent 6-sets using bit operations."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1
        return ind6_count

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return False
        if count_ind6_fast(n, masks) > 0:
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    def circulant_edges(n, S):
        """Generate edges for circulant graph C(n, S) where S is set of offsets."""
        edges = []
        for i in range(n):
            for d in S:
                j = (i + d) % n
                if i < j:
                    edges.append([i, j])
        return edges

    def check_circulant(n, S):
        """Check if circulant graph C(n, S) is K4-free and has no independent 6-set."""
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for d in S:
                j = (i + d) % n
                adj[i, j] = True
                adj[j, i] = True
        
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return False
        if count_ind6_fast(n, masks) > 0:
            return False
        return True

    # Clebsch graph: 16 vertices, K4-free, alpha=5. Valid baseline.
    clebsch_edges = [
        [0, 6], [0, 7], [0, 10], [0, 11],
        [1, 5], [1, 8], [1, 10], [1, 13],
        [2, 4], [2, 9], [2, 11], [2, 13],
        [3, 4], [3, 7], [3, 9], [3, 12],
        [5, 6], [5, 8], [5, 12], [5, 15],
        [6, 7], [6, 9], [6, 14], [6, 15],
        [7, 8], [7, 11], [7, 14], [7, 15],
        [8, 9], [8, 12], [8, 15],
        [9, 10], [9, 14],
        [10, 11], [10, 13], [10, 14],
        [11, 12], [11, 14], [11, 15],
        [12, 13], [12, 15],
        [13, 14], [13, 15]
    ]
    
    best_n = 16
    best_edges = clebsch_edges

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    
    if not is_valid(16, adj_16):
        best_n = 5
        best_edges = []

    # STRATEGY: Circulant graph search
    # A circulant graph C(n, S) has vertex i connected to i±d for d in S.
    # This reduces search from 2^(n(n-1)/2) to 2^(n/2) per n.
    
    def search_circulants(max_n=36):
        """Systematically search circulant graphs for valid K4-free, ind6-free graphs."""
        found = {}
        
        for n in range(17, max_n + 1):
            if time.time() > deadline - 2.0:
                break
            
            # For circulant C(n, S), S ⊆ {1, 2, ..., n//2}
            # (offsets > n/2 are equivalent to their complement)
            half = n // 2
            possible_offsets = list(range(1, half + 1))
            
            # Use randomized search with pruning
            # Start with dense configurations and remove problematic offsets
            best_S_for_n = None
            min_violations = float('inf')
            
            # Try random subsets of offsets
            num_tries = min(500, max(50, int((deadline - time.time()) * 20)))
            
            for trial in range(num_tries):
                if time.time() > deadline - 2.0:
                    break
                
                # Random density of offsets
                density = random.uniform(0.2, 0.5)
                S = set()
                for d in possible_offsets:
                    if random.random() < density:
                        S.add(d)
                
                if not S:
                    continue
                
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for d in S:
                        j = (i + d) % n
                        adj[i, j] = True
                        adj[j, i] = True
                
                masks = build_masks(n, adj)
                k4 = count_k4_fast(n, masks)
                ind6 = count_ind6_fast(n, masks)
                violations = k4 + ind6
                
                if violations == 0:
                    best_S_for_n = S
                    found[n] = S
                    break
                
                if violations < min_violations:
                    min_violations = violations
                    best_S_for_n = S
            
            # If we found a valid one, also try to extend
            if n in found:
                continue
            
            # Try local search on the best non-valid configuration
            if best_S_for_n is not None:
                S = set(best_S_for_n)
                for step in range(100):
                    if time.time() > deadline - 2.0:
                        break
                    
                    adj = np.zeros((n, n), dtype=bool)
                    for i in range(n):
                        for d in S:
                            j = (i + d) % n
                            adj[i, j] = True
                            adj[j, i] = True
                    
                    masks = build_masks(n, adj)
                    k4 = count_k4_fast(n, masks)
                    ind6 = count_ind6_fast(n, masks)
                    
                    if k4 == 0 and ind6 == 0:
                        found[n] = S
                        break
                    
                    # Try adding or removing a random offset
                    if random.random() < 0.5 and len(S) < len(possible_offsets):
                        candidates = [d for d in possible_offsets if d not in S]
                        if candidates:
                            d = random.choice(candidates)
                            S.add(d)
                    elif S:
                        d = random.choice(list(S))
                        S.remove(d)
        
        return found

    # Run circulant search
    circ_results = search_circulants(36)
    
    for n in sorted(circ_results.keys(), reverse=True):
        if n > best_n:
            S = circ_results[n]
            edges = circulant_edges(n, S)
            # Verify
            adj = np.zeros((n, n), dtype=bool)
            for i, j in edges:
                adj[i, j] = True
                adj[j, i] = True
            if is_valid(n, adj):
                best_n = n
                best_edges = edges
    
    # If circulant search didn't find much, try a hybrid: 
    # start from best circulant and do local search on full graph
    if best_n < 30 and time.time() < deadline - 15:
        # Use the best circulant as seed for local search
        seed_n = best_n
        adj = np.zeros((seed_n, seed_n), dtype=bool)
        for u, v in best_edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Try to grow: add vertices one at a time
        for target_n in range(seed_n + 1, min(seed_n + 10, 37)):
            if time.time() > deadline - 5:
                break
            
            new_adj = np.zeros((target_n, target_n), dtype=bool)
            new_adj[:seed_n, :seed_n] = adj
            
            # Choose edges from new vertex to old vertices
            # Heuristic: connect to vertices with similar "type" in circulant
            new_v = seed_n
            for old_v in range(seed_n):
                # Connect with some probability based on circulant structure
                # In circulant, vertex i connects to i±d; use modular distance
                dist = min(new_v % target_n - old_v, target_n - (new_v % target_n - old_v)) if new_v % target_n >= old_v else min(old_v - new_v % target_n, target_n - (old_v - new_v % target_n))
                if random.random() < 0.4:
                    new_adj[new_v, old_v] = True
                    new_adj[old_v, new_v] = True
            
            # Local search to fix violations
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                masks = build_masks(target_n, new_adj)
                k4 = count_k4_fast(target_n, masks)
                ind6 = count_ind6_fast(target_n, masks)
                
                if k4 == 0 and ind6 == 0:
                    best_n = target_n
                    best_edges = edges_from_adj(target_n, new_adj)
                    adj = new_adj
                    seed_n = target_n
                    break
                
                # Toggle a random edge
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                new_adj[u, v] = not new_adj[u, v]
                new_adj[v, u] = not new_adj[v, u]
    
    return {'n': best_n, 'edges': best_edges}
