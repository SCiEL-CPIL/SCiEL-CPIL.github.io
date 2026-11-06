"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n = 16
    # 1. Generate all possible triples (indices 0..559)
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # We need to choose 6 distinct points, then partition them into 3 pairs.
    # Number of ways to choose 6 points: C(16, 6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total Ks: 8008 * 15 = 120120
    
    # Precompute bitmask for each K and incidence list
    triple_to_ks = [[] for _ in range(num_triples)]
    all_ks_masks = []
    all_ks_indices = []  # store the 8 indices for each K
    
    # Hardcoded 15 matchings of indices 0-5
    matchings = [
        [(0,1),(2,3),(4,5)],
        [(0,1),(2,4),(3,5)],
        [(0,1),(2,5),(3,4)],
        [(0,2),(1,3),(4,5)],
        [(0,2),(1,4),(3,5)],
        [(0,2),(1,5),(3,4)],
        [(0,3),(1,2),(4,5)],
        [(0,3),(1,4),(2,5)],
        [(0,3),(1,5),(2,4)],
        [(0,4),(1,2),(3,5)],
        [(0,4),(1,3),(2,5)],
        [(0,4),(1,5),(2,3)],
        [(0,5),(1,2),(3,4)],
        [(0,5),(1,3),(2,4)],
        [(0,5),(1,4),(2,3)]
    ]
    
    # Precompute bits for each triple index
    triple_bits = [1 << i for i in range(num_triples)]
    
    for six_points in combinations(range(n), 6):
        for matching in matchings:
            p1 = (six_points[matching[0][0]], six_points[matching[0][1]])
            p2 = (six_points[matching[1][0]], six_points[matching[1][1]])
            p3 = (six_points[matching[2][0]], six_points[matching[2][1]])
            
            # Generate 8 triple indices
            t_indices = []
            mask = 0
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        t = tuple(sorted((x1, x2, x3)))
                        idx = triple_to_idx[t]
                        t_indices.append(idx)
                        mask |= triple_bits[idx]
            
            k_idx = len(all_ks_masks)
            all_ks_masks.append(mask)
            all_ks_indices.append(t_indices)
            
            for ti in t_indices:
                triple_to_ks[ti].append(k_idx)
                
    # Build neighbors list for SDF heuristic
    neighbors = [[] for _ in range(num_triples)]
    for k_idx, mask in enumerate(all_ks_masks):
        indices_in_k = all_ks_indices[k_idx]
        # Add edges between all pairs
        for i in range(len(indices_in_k)):
            for j in range(i+1, len(indices_in_k)):
                u = indices_in_k[i]
                v = indices_in_k[j]
                neighbors[u].append(v)
                neighbors[v].append(u)
    
    # Convert neighbors to sets for faster membership testing
    neighbor_sets = [set(neighbors[i]) for i in range(num_triples)]

    # Heuristic 1: Fast Random Greedy
    def greedy_random():
        current_mask = 0
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            bit = triple_bits[idx]
            is_conflict = False
            for k_idx in triple_to_ks[idx]:
                k_mask = all_ks_masks[k_idx]
                if (k_mask & current_mask) == (k_mask ^ bit):
                    is_conflict = True
                    break
            
            if not is_conflict:
                current_mask |= bit
        return current_mask

    # Heuristic 2: Smallest Degree First (SDF) Greedy
    def greedy_sdf():
        available = [1] * num_triples
        avail_list = list(range(num_triples))
        solution_mask = 0
        
        while avail_list:
            min_deg = float('inf')
            candidates = []
            
            for u in avail_list:
                if not available[u]:
                    continue
                deg = 0
                for v in neighbor_sets[u]:
                    if available[v]:
                        deg += 1
                
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            if not candidates:
                break
                
            u = random.choice(candidates)
            solution_mask |= triple_bits[u]
            
            # Remove u and neighbors
            available[u] = 0
            for v in neighbor_sets[u]:
                if available[v]:
                    available[v] = 0
            
            # Update avail_list (lazy deletion)
            avail_list = [x for x in avail_list if available[x]]
            
        return solution_mask

    def greedy_sdf_deterministic():
        available = [1] * num_triples
        avail_list = list(range(num_triples))
        solution_mask = 0
        
        while avail_list:
            min_deg = float('inf')
            candidates = []
            
            for u in avail_list:
                if not available[u]:
                    continue
                deg = 0
                for v in neighbor_sets[u]:
                    if available[v]:
                        deg += 1
                
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            if not candidates:
                break
                
            u = min(candidates)
            solution_mask |= triple_bits[u]
            
            # Remove u and neighbors
            available[u] = 0
            for v in neighbor_sets[u]:
                if available[v]:
                    available[v] = 0
            
            # Update avail_list (lazy deletion)
            avail_list = [x for x in avail_list if available[x]]
            
        return solution_mask

    def local_search(start_mask, max_iters=5000):
        """Local search: try to add triples, and if stuck, try swaps (remove 1, add 2)"""
        current_mask = start_mask
        current_count = current_mask.bit_count()
        
        # Build set of available triples (not in current solution)
        available = [1] * num_triples
        for i in range(num_triples):
            if current_mask & triple_bits[i]:
                available[i] = 0
        
        # Build conflict counts for available triples
        # For each K, count how many of its 8 triples are in current_mask
        k_counts = [0] * len(all_ks_masks)
        for k_idx, mask in enumerate(all_ks_masks):
            k_counts[k_idx] = (mask & current_mask).bit_count()
        
        # For each available triple, count how many Ks it would complete
        triple_conflicts = [0] * num_triples
        for i in range(num_triples):
            if not available[i]:
                continue
            for k_idx in triple_to_ks[i]:
                if k_counts[k_idx] == 7:
                    triple_conflicts[i] += 1
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try to add any triple with 0 conflicts
            added_any = False
            for i in range(num_triples):
                if available[i] and triple_conflicts[i] == 0:
                    # Add triple i
                    current_mask |= triple_bits[i]
                    current_count += 1
                    available[i] = 0
                    
                    # Update conflict counts for all triples in Ks containing i
                    for k_idx in triple_to_ks[i]:
                        k_counts[k_idx] += 1
                        for j in all_ks_indices[k_idx]:
                            if available[j]:
                                if k_counts[k_idx] == 8:
                                    triple_conflicts[j] += 1
                    improved = True
                    added_any = True
                    break
            
            if added_any:
                continue
            
            # Try swaps: remove 1, add 2
            # Find a triple to remove that allows adding 2 new triples
            # This is expensive, so we limit it
            if max_iters > 100:  # Only do swaps if we have plenty of iterations left
                swap_found = False
                # Pick a random triple from the current solution
                in_solution = [i for i in range(num_triples) if current_mask & triple_bits[i]]
                if in_solution:
                    random.shuffle(in_solution)
                    for rem in in_solution[:20]:  # Try at most 20 removals
                        # Tentatively remove rem
                        current_mask ^= triple_bits[rem]
                        current_count -= 1
                        available[rem] = 1
                        
                        # Update k_counts
                        for k_idx in triple_to_ks[rem]:
                            k_counts[k_idx] -= 1
                            for j in all_ks_indices[k_idx]:
                                if available[j]:
                                    if k_counts[k_idx] < 8:
                                        triple_conflicts[j] -= 1
                        
                        # Try to add 2 triples
                        added = 0
                        added_indices = []
                        for i in range(num_triples):
                            if available[i] and triple_conflicts[i] == 0:
                                current_mask |= triple_bits[i]
                                current_count += 1
                                available[i] = 0
                                added_indices.append(i)
                                
                                for k_idx in triple_to_ks[i]:
                                    k_counts[k_idx] += 1
                                    for j in all_ks_indices[k_idx]:
                                        if available[j]:
                                            if k_counts[k_idx] == 8:
                                                triple_conflicts[j] += 1
                                
                                added += 1
                                if added == 2:
                                    break
                        
                        if added == 2:
                            # Keep the swap
                            improved = True
                            swap_found = True
                            break
                        else:
                            # Revert the removal
                            current_mask |= triple_bits[rem]
                            current_count += 1
                            available[rem] = 0
                            
                            for k_idx in triple_to_ks[rem]:
                                k_counts[k_idx] += 1
                                for j in all_ks_indices[k_idx]:
                                    if available[j]:
                                        if k_counts[k_idx] == 8:
                                            triple_conflicts[j] += 1
                            
                            # Revert the additions
                            for i in added_indices:
                                current_mask ^= triple_bits[i]
                                current_count -= 1
                                available[i] = 1
                                
                                for k_idx in triple_to_ks[i]:
                                    k_counts[k_idx] -= 1
                                    for j in all_ks_indices[k_idx]:
                                        if available[j]:
                                            if k_counts[k_idx] < 8:
                                                triple_conflicts[j] -= 1
                
                if swap_found:
                    continue
            
            if not improved:
                break
        
        return current_mask, current_count

    best_mask = 0
    best_count = 0
    
    # Run several initializations
    # First, deterministic SDF
    m = greedy_sdf_deterministic()
    c = m.bit_count()
    if c > best_count:
        best_count = c
        best_mask = m
    
    # Run Random Greedy for a while
    while time.time() < deadline - 60.0:  # Leave 60s for local search
        m = greedy_random()
        c = m.bit_count()
        if c > best_count:
            best_count = c
            best_mask = m
        
        # Also try SDF
        m = greedy_sdf()
        c = m.bit_count()
        if c > best_count:
            best_count = c
            best_mask = m

    # Run local search on the best solution found
    if time.time() < deadline - 10.0:
        m, c = local_search(best_mask, max_iters=10000)
        if c > best_count:
            best_count = c
            best_mask = m
    
    # Run more local search if time permits
    while time.time() < deadline - 5.0:
        # Try local search on a fresh random greedy solution
        m0 = greedy_random()
        c0 = m0.bit_count()
        if c0 > best_count:
            best_count = c0
            best_mask = m0
        
        m, c = local_search(m0, max_iters=5000)
        if c > best_count:
            best_count = c
            best_mask = m
    
    # Convert best_mask back to list of triples
    result = []
    for i in range(num_triples):
        if best_mask & triple_bits[i]:
            result.append(list(all_triples[i]))
            
    return result
# EVOLVE-BLOCK-END
