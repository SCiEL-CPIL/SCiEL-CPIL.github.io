"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Build conflict graph using bitmasks
    # adj[i] is a bitmask of vertices j such that (i+1) + (j+1) is a perfect square
    adj = [0] * N
    
    for i in range(N):
        val = i + 1
        for j in range(N):
            other = j + 1
            if val + other in squares:
                adj[i] |= (1 << j)
    
    # Identify vertices that cannot be in any valid set (self-conflict: 2*a is a square)
    self_conflict = [False] * N
    for i in range(N):
        if adj[i] & (1 << i):
            self_conflict[i] = True
    
    # Candidate vertices (no self-conflict)
    candidates = [i for i in range(N) if not self_conflict[i]]
    
    # Precompute neighbor masks restricted to candidates
    candidate_mask = 0
    for i in candidates:
        candidate_mask |= (1 << i)
    
    # neighbors[i] contains only valid candidates that conflict with i
    neighbors = [adj[i] & candidate_mask for i in range(N)]
    
    def greedy_independent_set(order):
        """Greedy: process vertices in given order, add if no conflict with already added."""
        selected_mask = 0
        for v in order:
            if selected_mask & neighbors[v]:
                continue
            selected_mask |= (1 << v)
        return selected_mask

    def degree_based_order(cand_list):
        """Order by degree (number of conflicts within candidates) ascending."""
        degrees = []
        for v in cand_list:
            deg = bin(neighbors[v]).count('1')
            degrees.append((deg, v))
        degrees.sort()
        return [v for _, v in degrees]
    
    def modular_seeded_order():
        """
        Use modular arithmetic (mod 4) to create a strong initial seed.
        Numbers congruent to 1 or 3 mod 4 tend to have fewer conflicts.
        """
        # Group candidates by mod 4
        mod_groups = {0: [], 1: [], 2: [], 3: []}
        for v in candidates:
            val = v + 1
            mod_groups[val % 4].append(v)
        
        # Prioritize groups with fewer internal conflicts
        # Groups 1 and 3 mod 4 typically have fewer square-sum conflicts
        # Order: 1 mod 4, 3 mod 4, then 0 and 2 by degree
        
        # Calculate average degree for each group
        group_degrees = {}
        for mod in [1, 3, 0, 2]:
            if mod_groups[mod]:
                total_deg = sum(bin(neighbors[v]).count('1') for v in mod_groups[mod])
                group_degrees[mod] = total_deg / len(mod_groups[mod])
            else:
                group_degrees[mod] = float('inf')
        
        # Sort groups by average degree (ascending)
        sorted_mods = sorted([1, 3, 0, 2], key=lambda m: group_degrees.get(m, float('inf')))
        
        # Build order: within each group, sort by individual degree
        order = []
        for mod in sorted_mods:
            group = mod_groups[mod]
            if group:
                group_sorted = sorted(group, key=lambda v: bin(neighbors[v]).count('1'))
                order.extend(group_sorted)
        
        return order

    def local_search_1_2_exchange(selected_mask, max_iters=1000):
        """
        Local search using 1-2 exchanges.
        Tries to replace one vertex u in S with two vertices v, w not in S,
        such that v, w are compatible with S\{u} and with each other.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            selected_list = [i for i in range(N) if best_mask & (1 << i)]
            random.shuffle(selected_list)
            
            for u in selected_list:
                # Remove u from the set
                mask_without_u = best_mask & ~(1 << u)
                
                # Calculate neighbors of mask_without_u
                union_neighbors = 0
                temp_mask = mask_without_u
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = (lsb.bit_length() - 1)
                    union_neighbors |= neighbors[idx]
                    temp_mask ^= lsb
                    
                # Available vertices are those in candidate_mask, not in mask_without_u, 
                # and not in union_neighbors.
                available = candidate_mask & ~mask_without_u & ~union_neighbors
                
                if bin(available).count('1') < 2:
                    continue
                
                avail_list = [i for i in range(N) if available & (1 << i)]
                random.shuffle(avail_list)
                
                found_pair = False
                for i in range(len(avail_list)):
                    v = avail_list[i]
                    for j in range(i + 1, len(avail_list)):
                        w = avail_list[j]
                        if not (neighbors[v] & (1 << w)):
                            new_mask = mask_without_u | (1 << v) | (1 << w)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
                                found_pair = True
                                break
                    if found_pair:
                        break
                
            if not improved:
                break
                
        return best_mask, best_size
    
    def final_polish(selected_mask, max_iters=50):
        """
        Final high-precision polish pass with more aggressive 1-2 exchanges.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            selected_list = [i for i in range(N) if best_mask & (1 << i)]
            random.shuffle(selected_list)
            
            for u in selected_list:
                mask_without_u = best_mask & ~(1 << u)
                
                union_neighbors = 0
                temp_mask = mask_without_u
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    idx = (lsb.bit_length() - 1)
                    union_neighbors |= neighbors[idx]
                    temp_mask ^= lsb
                    
                available = candidate_mask & ~mask_without_u & ~union_neighbors
                
                if bin(available).count('1') < 2:
                    continue
                
                avail_list = [i for i in range(N) if available & (1 << i)]
                
                # Try all pairs (not just random) for final polish
                for i in range(len(avail_list)):
                    v = avail_list[i]
                    for j in range(i + 1, len(avail_list)):
                        w = avail_list[j]
                        if not (neighbors[v] & (1 << w)):
                            new_mask = mask_without_u | (1 << v) | (1 << w)
                            new_size = bin(new_mask).count('1')
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                improved = True
                                break
                    if improved:
                        break
                
            if not improved:
                break
                
        return best_mask, best_size

    # Strategy 1: Modular-seeded greedy (strong initial seed)
    best_mask = 0
    best_size = 0
    
    # Try modular-seeded ordering
    order = modular_seeded_order()
    selected_mask = greedy_independent_set(order)
    best_size = bin(selected_mask).count('1')
    best_mask = selected_mask
    
    # Local search on the modular-seeded result
    improved_mask, improved_size = local_search_1_2_exchange(best_mask, max_iters=150)
    if improved_size > best_size:
        best_size = improved_size
        best_mask = improved_mask
    
    # Also try degree-based ordering as backup
    order_deg = degree_based_order(candidates)
    selected_mask_deg = greedy_independent_set(order_deg)
    size_deg = bin(selected_mask_deg).count('1')
    if size_deg > best_size:
        improved_mask_deg, improved_size_deg = local_search_1_2_exchange(selected_mask_deg, max_iters=100)
        if improved_size_deg > best_size:
            best_size = improved_size_deg
            best_mask = improved_mask_deg

    # Strategy 2: Multiple random restarts
    while time.time() - start_time < DEADLINE - 5.0:  # Leave 5s for final polish
        # Random ordering
        random_order = list(candidates)
        random.shuffle(random_order)
        
        selected_mask = greedy_independent_set(random_order)
        current_size = bin(selected_mask).count('1')
        
        if current_size >= best_size - 5:
             improved_mask, improved_size = local_search_1_2_exchange(selected_mask, max_iters=30)
             if improved_size > best_size:
                 best_size = improved_size
                 best_mask = improved_mask
        
        if time.time() - start_time > DEADLINE - 5.0:
            break

    # Final polish pass with more aggressive search
    if time.time() - start_time < DEADLINE - 2.0:
        polished_mask, polished_size = final_polish(best_mask, max_iters=100)
        if polished_size > best_size:
            best_size = polished_size
            best_mask = polished_mask

    # Convert back to values
    result = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    if not result:
        for i in candidates:
            return [i+1]
        return [1]

    return result
# EVOLVE-BLOCK-END
