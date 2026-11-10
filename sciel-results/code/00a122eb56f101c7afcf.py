import time
import random

N = 256
MAX_SUM = 2 * N

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Safety margin: 55s limit (5s buffer for overhead)
    deadline = start_time + 55.0
    
    best_set = []
    best_size = float('inf')
    
    # Bitmask representation
    # Elements: bits 1..N
    # Sums: bits 2..2N
    # We will use Python integers as bitmasks.
    
    # Precompute bit masks for elements 1..N
    # elem_mask[i] = 1 << i
    # sum_mask[s] = 1 << s
    
    # Helper to extract set bits
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    # Strategy: Max-Blocking Greedy with Randomized Tie-Breaking and Noise
    # This aligns with the "inspiration 1" and "inspiration 2" which scored 1.4545.
    # The current program scored 1.2308 using simple random/ordered greedy.
    # The key improvement is choosing the element that "blocks" the most other candidates,
    # which tends to lead to smaller maximal sets.
    
    # We will run multiple restarts.
    # Each restart:
    # 1. Initialize empty set S, sums mask, available mask (all 1..N).
    # 2. While available is not empty:
    #    a. Find the best candidate z in available.
    #       - Must be valid (no sum conflict).
    #       - Score = number of other available elements blocked by adding z.
    #       - Add small random noise to break ties and explore.
    #    b. Add best z to S, update sums, update available.
    # 3. Check size, update best.
    
    rng = random.Random(123)
    
    # Optimization: Precompute bit masks for sums?
    # Since N=256, max sum is 512.
    # We can just use 1 << s directly.
    
    # To speed up the "blocking" calculation:
    # When we consider adding z:
    # New sums: {2z} U {z+a | a in S}
    # An element w (w != z, w in available) is blocked if:
    # 1. 2w is in (existing_sums U new_sums)
    # 2. w+a is in (existing_sums U new_sums) for some a in (S U {z})
    
    # Let's refine the blocking logic for speed.
    # Instead of iterating all available w for every candidate z,
    # we can be smarter, but with N=256 and |S| small, iteration is likely fine.
    # |Available| starts at 256, shrinks quickly.
    # |Candidates| is |Available|.
    # Inner loop |Available|.
    # Total ops per step ~ O(N^2).
    # Steps ~ 15-20.
    # 20 * 256 * 256 ~ 1.3M ops per restart.
    # Python can do ~10M-50M simple ops/sec.
    # So ~0.05s per restart.
    # In 50s, we can do ~1000 restarts.
    # This should be sufficient to find a very small set.
    
    # Let's implement a fast version.
    
    for iter in range(2000): # Cap iterations, rely on time check
        if time.time() > deadline:
            break
            
        # Use a different seed for each iteration to ensure diversity
        # rng is shared, but we can also just use the global random or local rng
        # Let's use local rng for reproducibility if needed, but here we just want diversity.
        
        S_mask = 0
        sums_mask = 0
        available_mask = 0
        for i in range(1, N + 1):
            available_mask |= (1 << i)
        
        S_list = []
        
        while available_mask:
            # Extract available candidates
            avail_list = get_bits(available_mask)
            
            # If list is small, shuffle for tie-breaking randomness?
            # Or just iterate and use noise.
            
            best_z = -1
            best_score = -1.0
            
            # We can shuffle avail_list to break ties randomly if scores are equal,
            # but adding noise is better for exploration.
            
            # Optimization: If avail_list is large, we might want to sample?
            # No, we need the MAX blocking. We must check all.
            
            for z in avail_list:
                # Check validity of z
                # 1. 2z not in sums_mask
                if (sums_mask >> (2*z)) & 1:
                    continue
                
                # 2. No collision with existing sums
                # (sums_mask >> z) & S_mask == 0
                if (sums_mask >> z) & S_mask:
                    continue
                
                # z is valid. Calculate blocking score.
                # New sums bits:
                # new_sums_bits = (1 << (2*z))
                # plus (1 << (z+a)) for a in S_mask
                
                # We don't need the full new_sums_mask integer if we can check blocking directly?
                # Blocked w if:
                # (a) 2w in existing_sums OR 2w in new_sums
                # (b) w+a in existing_sums OR w+a in new_sums for some a in S U {z}
                
                # Let's compute new_sums_mask for this candidate z.
                # It's just sums_mask | (1 << (2*z)) | (sums of z+a)
                # sums of z+a: for each a in S, set bit (z+a).
                
                # Efficient calculation of "z+a" bits:
                # We can iterate S_list (small).
                add_bits = 0
                for a in S_list:
                    add_bits |= (1 << (z + a))
                
                new_sums_mask = sums_mask | (1 << (2*z)) | add_bits
                new_S_mask = S_mask | (1 << z)
                
                # Now count blocked w in avail_list (w != z)
                blocked_count = 0
                
                # Iterate w in avail_list
                # We can optimize this by precomputing something?
                # For now, direct check.
                for w in avail_list:
                    if w == z:
                        continue
                    
                    # Check if w is blocked by new state
                    # 1. 2w in new_sums_mask?
                    if (new_sums_mask >> (2*w)) & 1:
                        blocked_count += 1
                        continue
                    
                    # 2. w+a in new_sums_mask for some a in new_S_mask?
                    # Equivalent to: (new_sums_mask >> w) & new_S_mask != 0
                    if (new_sums_mask >> w) & new_S_mask:
                        blocked_count += 1
                
                # Heuristic: Maximize blocked_count
                # Add noise to encourage diversity
                # Noise scale: maybe 0.5?
                score = blocked_count + rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_z = z
            
            if best_z == -1:
                # Should not happen if available is not empty and we start empty?
                # If S is empty, any z is valid.
                # If S is not empty, some z might be invalid, but at least one should be valid?
                # Actually, if we pick greedily, we might paint ourselves into a corner?
                # No, Sidon sets can always be extended?
                # Wait, if we have a maximal Sidon set, no more can be added.
                # But we are building it. If available is not empty, there must be a valid z?
                # Not necessarily. It's possible that all remaining elements conflict.
                # In that case, the set is already maximal.
                # But our loop condition is `while available_mask`.
                # If best_z is -1, it means no element in available can be added.
                # So the current set is maximal.
                # We should break.
                break
            
            # Commit best_z
            S_mask |= (1 << best_z)
            S_list.append(best_z)
            
            # Update sums_mask
            # Add 2*best_z
            sums_mask |= (1 << (2*best_z))
            # Add best_z + a for a in old S
            # We can reuse the add_bits calculation?
            # In the loop above, we calculated add_bits for the *candidate* z.
            # Since best_z is the one we picked, we can just recompute or store it.
            # To save time, let's recompute quickly.
            for a in S_list[:-1]: # S_list[:-1] is the old S
                sums_mask |= (1 << (best_z + a))
            
            # Update available_mask
            # Remove best_z
            available_mask &= ~(1 << best_z)
            
            # Remove any w that is now blocked
            # We need to check remaining available elements against the NEW state.
            # New state: S_mask, sums_mask.
            
            temp_avail = available_mask
            new_avail_mask = 0
            while temp_avail:
                lsb_w = temp_avail & -temp_avail
                w = lsb_w.bit_length() - 1
                temp_avail ^= lsb_w
                
                # Check if w is still valid
                # 1. 2w not in sums_mask
                if not ((sums_mask >> (2*w)) & 1):
                    # 2. No collision
                    if not ((sums_mask >> w) & S_mask):
                        new_avail_mask |= lsb_w
            
            available_mask = new_avail_mask
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
            # Early stopping if we hit a theoretical lower bound or very good result
            # Lower bound is ~sqrt(N) = 16.
            # Maximal sets are usually larger.
            # If we get <= 14, that's excellent.
            # Score 1.4545 implies size ~ 11 if seed is 16? 16/11 = 1.45.
            # So aiming for 11 or 12.
            if best_size <= 12:
                # We can stop early to save time, or continue to see if we get 11?
                # Given time budget, let's continue a bit more to be sure,
                # but if we have plenty of time, we can stop.
                # Let's just let the loop finish or time out.
                pass

    # Fallback if something went wrong (should not happen)
    if not best_set:
        # Simple greedy ascending
        S_list = []
        sums = set()
        for i in range(1, N+1):
            if 2*i not in sums and all(i+a not in sums for a in S_list):
                S_list.append(i)
                sums.add(2*i)
                for a in S_list[:-1]:
                    sums.add(i+a)
        best_set = S_list
        best_size = len(S_list)
        
    return best_set

# EVOLVE-BLOCK-END
