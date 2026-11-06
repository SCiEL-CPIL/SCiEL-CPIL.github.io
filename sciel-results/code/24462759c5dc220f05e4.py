import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # 1. Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    num_primes = len(primes)
    
    # 2. Precompute masks
    # L_max for bitset. We want to go as high as possible.
    # Let's set L_max = 20000 to be safe for "jumps". 
    # If we cover more, we can extend, but 20000 is a good start.
    L_MAX = 20000
    full_mask = (1 << L_MAX) - 1
    
    # mask_cache[p_idx][r]
    # p_idx is index in primes list
    mask_cache = {}
    for p_idx, p in enumerate(primes):
        masks_for_p = [0] * p
        for r in range(p):
            mask = 0
            val = r + 1 # 1-based index for bit position? 
            # Bit i represents integer i+1.
            # So integer x corresponds to bit x-1.
            # Residue r means x % p == r.
            # Smallest x >= 1 with x % p == r:
            # if r == 0, x = p, 2p, ...
            # if r > 0, x = r, r+p, ...
            # Wait, r is in 0..p-1.
            # If r == 0, numbers are p, 2p, ...
            # If r > 0, numbers are r, r+p, ...
            
            start_val = r if r > 0 else p
            mask = 0
            for x in range(start_val, L_MAX + 1, p):
                mask |= (1 << (x - 1))
            masks_for_p[r] = mask
        mask_cache[p_idx] = masks_for_p

    # 3. Beam Search
    # State: (covered_mask, chosen_classes, cost, prefix_len)
    # chosen_classes: list of (p_idx, r)
    
    # Initial state
    initial_state = (0, [], 0.0, 0)
    beam = [initial_state]
    
    best_L_found = 0
    best_state = initial_state
    
    # Helper to get smallest uncovered k
    def get_smallest_uncovered(covered_mask):
        # Invert bits within L_MAX
        # If all bits up to L_MAX are set, return L_MAX + 1
        # We check if covered_mask has 0 at position k-1
        
        # Find first 0 bit from LSB
        # ~covered_mask will have 1s where covered_mask has 0s.
        # But ~covered_mask is infinite 1s in Python.
        # So mask it with full_mask.
        inv = (~covered_mask) & full_mask
        if inv == 0:
            return L_MAX + 1
        # Get index of lowest set bit
        # bit_length gives 1 + index of highest set bit.
        # We want lowest set bit index.
        # inv & -inv isolates lowest set bit.
        lowest_bit = inv & -inv
        # bit_length of lowest_bit is index + 1
        return lowest_bit.bit_length()
        
    # Helper to calculate prefix length
    def calc_prefix_len(covered_mask):
        # Same as get_smallest_uncovered, but we want the count
        # If k is smallest uncovered, prefix length is k-1
        k = get_smallest_uncovered(covered_mask)
        return k - 1

    beam_width = 50
    max_depth = 100 # Limit depth to avoid infinite loops if L doesn't grow
    
    for depth in range(max_depth):
        if time.time() - start_time > time_limit:
            break
            
        next_beam = []
        
        for state in beam:
            covered_mask, chosen_classes, cost, prefix_len = state
            
            # Update best
            if prefix_len > best_L_found:
                best_L_found = prefix_len
                best_state = state
                
            # If we reached L_MAX, we can't improve much more in this branch
            # But we might still need to check if we can go higher?
            # The bitset is limited to L_MAX. If we cover all L_MAX, 
            # we should probably expand L_MAX or stop.
            # For now, if prefix_len == L_MAX, we stop expanding this state 
            # because we can't represent more coverage.
            if prefix_len >= L_MAX:
                continue
                
            k = prefix_len + 1 # Smallest uncovered integer
            
            # If k > L_MAX, we are out of bitset range. 
            # This shouldn't happen if we stop at L_MAX.
            
            # Try all unused primes
            used_primes_set = set(c[0] for c in chosen_classes)
            
            for p_idx in range(num_primes):
                if p_idx in used_primes_set:
                    continue
                    
                p = primes[p_idx]
                r = k % p
                
                new_cost = cost + 1.0 / p
                if new_cost > 1.0:
                    continue
                
                new_mask = covered_mask | mask_cache[p_idx][r]
                
                # Calculate new prefix length
                # Optimization: We know k is now covered.
                # The new prefix is at least k.
                # We can just call calc_prefix_len
                new_prefix_len = calc_prefix_len(new_mask)
                
                # Pruning: If this move doesn't improve the prefix length 
                # compared to the best possible for this state? 
                # Actually, we MUST cover k, so prefix_len will be at least k.
                # If new_prefix_len == prefix_len, it means k was already covered?
                # No, k was uncovered. So new_prefix_len >= k = prefix_len + 1.
                
                # Add to next_beam
                new_classes = chosen_classes + [(p_idx, r)]
                next_beam.append((new_mask, new_classes, new_cost, new_prefix_len))
        
        if not next_beam:
            break
            
        # Sort next_beam by prefix_len (desc), then cost (asc)
        next_beam.sort(key=lambda x: (-x[3], x[2]))
        
        # Keep top W
        beam = next_beam[:beam_width]
        
        # Early termination if best possible is reached?
        # If the best state in beam has cost close to 1 and prefix is high, 
        # we might be stuck.
        
    # Convert best_state to output format
    # best_state: (covered_mask, chosen_classes, cost, prefix_len)
    _, best_classes, _, _ = best_state
    
    # Format: [[p, r], ...]
    result = []
    for p_idx, r in best_classes:
        result.append([primes[p_idx], r])
        
    return result
# EVOLVE-BLOCK-END
