"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm, gcd
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)
    
    # Weights are 1/x (use float for speed, Fraction for final polish)
    weights = [1.0 / x for x in nodes]
    
    # Build conflict masks
    # masks[i] is an integer where bit j is set if nodes[i] conflicts with nodes[j]
    masks = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(nodes[i], nodes[j]) <= 256:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)

    def get_value(mask):
        val = 0.0
        for i in range(N):
            if (mask >> i) & 1:
                val += weights[i]
        return val

    def get_value_fraction(mask):
        """High-precision value using Fraction"""
        val = Fraction(0, 1)
        for i in range(N):
            if (mask >> i) & 1:
                val += Fraction(1, nodes[i])
        return val

    def greedy_fill(seed_mask):
        cur_mask = seed_mask
        # Iterate through all nodes sorted by weight descending (node ascending)
        for i in range(N):
            if not ((cur_mask >> i) & 1):
                if (masks[i] & cur_mask) == 0:
                    cur_mask |= (1 << i)
        return cur_mask

    def local_search(init_mask):
        cur_mask = init_mask
        cur_val = get_value(cur_mask)
        best_mask = cur_mask
        best_val = cur_val
        
        # List of indices in current mask for faster iteration
        cur_indices = [i for i in range(N) if (cur_mask >> i) & 1]
        
        improved = True
        while improved:
            improved = False
            best_swap_gain = 0.0
            best_swap_u = -1
            best_swap_v = -1
            
            # Try single swaps
            for u in cur_indices:
                mask_without_u = cur_mask ^ (1 << u)
                # Iterate through all v not in cur_mask
                for v in range(N):
                    if not ((cur_mask >> v) & 1):
                        if (masks[v] & mask_without_u) == 0:
                            gain = weights[v] - weights[u]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_u = u
                                best_swap_v = v
            
            if best_swap_u != -1:
                # Apply swap
                cur_mask ^= (1 << best_swap_u)
                cur_mask |= (1 << best_swap_v)
                cur_val += best_swap_gain
                # Update indices
                cur_indices.remove(best_swap_u)
                cur_indices.append(best_swap_v)
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_mask = cur_mask
                    
                improved = True
                
                # Time check inside loop
                if time.time() > deadline - 5.0:
                    break
            else:
                # No single swap improves. Try 2-swaps (remove 2, add 2)
                best_2swap_gain = 0.0
                best_2swap_u1 = -1
                best_2swap_u2 = -1
                best_2swap_v1 = -1
                best_2swap_v2 = -1
                
                n_cur = len(cur_indices)
                for i in range(n_cur):
                    u1 = cur_indices[i]
                    for j in range(i + 1, n_cur):
                        u2 = cur_indices[j]
                        mask_without_2 = cur_mask ^ (1 << u1) ^ (1 << u2)
                        # Find best pair of v's to add
                        # Pre-filter: v must be compatible with mask_without_2
                        compatible_v = []
                        for v in range(N):
                            if not ((cur_mask >> v) & 1) and (masks[v] & mask_without_2) == 0:
                                compatible_v.append(v)
                        
                        if len(compatible_v) < 2:
                            continue
                        
                        # Find best pair of compatible v's that don't conflict with each other
                        best_pair_gain = 0.0
                        best_v1 = -1
                        best_v2 = -1
                        for a in range(len(compatible_v)):
                            v1 = compatible_v[a]
                            for b in range(a + 1, len(compatible_v)):
                                v2 = compatible_v[b]
                                if (masks[v1] & (1 << v2)) == 0:
                                    gain = weights[v1] + weights[v2] - weights[u1] - weights[u2]
                                    if gain > best_pair_gain:
                                        best_pair_gain = gain
                                        best_v1 = v1
                                        best_v2 = v2
                        
                        if best_v1 != -1 and best_pair_gain > best_2swap_gain:
                            best_2swap_gain = best_pair_gain
                            best_2swap_u1 = u1
                            best_2swap_u2 = u2
                            best_2swap_v1 = best_v1
                            best_2swap_v2 = best_v2
                
                if best_2swap_u1 != -1:
                    # Apply 2-swap
                    cur_mask ^= (1 << best_2swap_u1)
                    cur_mask ^= (1 << best_2swap_u2)
                    cur_mask |= (1 << best_2swap_v1)
                    cur_mask |= (1 << best_2swap_v2)
                    cur_val += best_2swap_gain
                    # Update indices
                    cur_indices.remove(best_2swap_u1)
                    cur_indices.remove(best_2swap_u2)
                    cur_indices.append(best_2swap_v1)
                    cur_indices.append(best_2swap_v2)
                    
                    if cur_val > best_val:
                        best_val = cur_val
                        best_mask = cur_mask
                        
                    improved = True
                    
                    if time.time() > deadline - 5.0:
                        break
                else:
                    # No 2-swap improves either
                    break
                    
        return best_mask, best_val

    def polish(mask):
        """Final high-precision polish using Fraction arithmetic"""
        cur_mask = mask
        cur_indices = [i for i in range(N) if (cur_mask >> i) & 1]
        
        # Use Fraction for exact comparisons
        cur_val = get_value_fraction(cur_mask)
        best_mask = cur_mask
        best_val = cur_val
        
        improved = True
        while improved:
            improved = False
            best_swap_gain = Fraction(0, 1)
            best_swap_u = -1
            best_swap_v = -1
            
            for u in cur_indices:
                mask_without_u = cur_mask ^ (1 << u)
                for v in range(N):
                    if not ((cur_mask >> v) & 1):
                        if (masks[v] & mask_without_u) == 0:
                            gain = Fraction(1, nodes[v]) - Fraction(1, nodes[u])
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_u = u
                                best_swap_v = v
            
            if best_swap_u != -1:
                cur_mask ^= (1 << best_swap_u)
                cur_mask |= (1 << best_swap_v)
                cur_val += best_swap_gain
                cur_indices.remove(best_swap_u)
                cur_indices.append(best_swap_v)
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_mask = cur_mask
                    
                improved = True
                if time.time() > deadline - 3.0:
                    break
            else:
                break
        
        return best_mask, best_val

    best_result_mask = 0
    best_result_val = 0.0
    
    # Try different seeds - more diverse set
    # Small numbers, primes, powers of 2, and some strategic choices
    seeds = list(range(2, 31))  # 2..30
    # Add some primes
    primes_to_add = [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes_to_add:
        if p not in seeds:
            seeds.append(p)
    # Add powers of 2
    for p in [128, 64, 32]:
        if p not in seeds:
            seeds.append(p)
    
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break
            
        seed_idx = seed - 2 # Index in nodes
        seed_mask = 1 << seed_idx
        
        # Greedy fill
        init_mask = greedy_fill(seed_mask)
        
        # Local search
        final_mask, val = local_search(init_mask)
        
        if val > best_result_val:
            best_result_val = val
            best_result_mask = final_mask

    # If no seed worked (shouldn't happen), fallback to greedy
    if best_result_mask == 0:
        init_mask = greedy_fill(0)
        final_mask, val = local_search(init_mask)
        best_result_mask = final_mask
        best_result_val = val

    # Final polish with Fraction precision
    if time.time() < deadline - 5.0:
        best_result_mask, best_result_val_frac = polish(best_result_mask)
        best_result_val = float(best_result_val_frac)

    # Extract result
    result = [nodes[i] for i in range(N) if (best_result_mask >> i) & 1]
    
    return result
# EVOLVE-BLOCK-END
