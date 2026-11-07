import time
import random
import math

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Precompute divisibility masks: div_mask[s] has bit m set if s divides m and m != s
    # Bit s in sums_mask means sum s is present
    MAX_SUM = 256 * 30  # max possible sum
    # Actually max sum is 256+255+... but for n elements each <=256, max sum is n*256
    # For n up to 30, that's 7680. But we only care about divisibility among sums.
    # Let's use a reasonable max. For 30 elements of max 256, max sum = 7680.
    # But we can cap it. Let's use 8000 to be safe.
    MAX_S = 8000
    
    # div_mask[s]: integer where bit m is set if s | m and m != s, for m in 1..MAX_S
    div_mask = [0] * (MAX_S + 1)
    for s in range(1, MAX_S + 1):
        m = 0
        for multiple in range(2 * s, MAX_S + 1, s):
            m |= (1 << multiple)
        div_mask[s] = m
    
    def _is_valid_addition(sums_mask, x):
        """Check if x can be added given current sums_mask (bitmask of present sums)."""
        # New sums: x and s+x for each s where bit s is set
        # First check: x itself must not be in sums_mask
        if (sums_mask >> x) & 1:
            return False
        
        # Build new_sums_mask
        new_mask = sums_mask | (1 << x)
        
        # For each existing sum s, s+x must not collide with existing or other new sums
        # and must not create divisibility issues
        # Efficient: iterate over set bits in sums_mask
        temp = sums_mask
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            ns = s + x
            if ns > MAX_S:
                temp ^= lsb
                continue
            if (new_mask >> ns) & 1:
                return False  # collision
            new_mask |= (1 << ns)
            temp ^= lsb
        
        # Now check divisibility: for each sum a in new_mask, no other sum b in new_mask should be a multiple of a
        # Using div_mask: for each a, check if new_mask & div_mask[a] != 0
        temp = new_mask
        while temp:
            lsb = temp & (-temp)
            a = lsb.bit_length() - 1
            if a > MAX_S:
                temp ^= lsb
                continue
            if new_mask & div_mask[a]:
                return False
            temp ^= lsb
        
        return True
    
    def _update_sums_mask(sums_mask, x):
        """Return new sums_mask after adding x."""
        new_mask = sums_mask | (1 << x)
        temp = sums_mask
        while temp:
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            ns = s + x
            if ns <= MAX_S:
                new_mask |= (1 << ns)
            temp ^= lsb
        return new_mask
    
    def _validate_full(arr):
        """Full validation of a set of integers."""
        n = len(arr)
        all_sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += arr[i]
            if s in all_sums:
                return False
            all_sums.add(s)
        
        sums_list = sorted(all_sums)
        for i in range(len(sums_list)):
            a = sums_list[i]
            for j in range(i + 1, len(sums_list)):
                b = sums_list[j]
                if b % a == 0:
                    return False
        return True
    
    def _greedy_construct(start_pool, deadline):
        """Greedy construction from a given pool of candidates."""
        current = []
        sums_mask = 0
        
        # Sort candidates: prefer larger numbers (less likely to be divisors)
        candidates = sorted(start_pool, reverse=True)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 30:
                break
            if _is_valid_addition(sums_mask, x):
                current.append(x)
                sums_mask = _update_sums_mask(sums_mask, x)
        
        return current, sums_mask
    
    # Phase 1: Multiple greedy starts with different random pools
    best_set = []
    best_size = 0
    
    # Strategy A: Greedy from large numbers [129..256]
    pool_a = list(range(129, 257))
    random.shuffle(pool_a)
    result_a, _ = _greedy_construct(pool_a, deadline)
    if len(result_a) > best_size:
        best_size = len(result_a)
        best_set = result_a[:]
    
    # Strategy B: Greedy from full range [1..256]
    pool_b = list(range(1, 257))
    random.shuffle(pool_b)
    result_b, _ = _greedy_construct(pool_b, deadline)
    if len(result_b) > best_size:
        best_size = len(result_b)
        best_set = result_b[:]
    
    # Strategy C: Greedy from primes and large composites
    pool_c = []
    for x in range(256, 0, -1):
        is_prime = True
        for d in range(2, int(math.sqrt(x)) + 1):
            if x % d == 0:
                is_prime = False
                break
        if is_prime or x > 128:
            pool_c.append(x)
    random.shuffle(pool_c)
    result_c, _ = _greedy_construct(pool_c, deadline)
    if len(result_c) > best_size:
        best_size = len(result_c)
        best_set = result_c[:]
    
    # Phase 2: Local search with swaps and insertions
    if best_set and time.time() < deadline - 15:
        current = best_set[:]
        current_sums_mask = 0
        # Rebuild sums_mask
        n = len(current)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current[i]
            if s <= MAX_S:
                current_sums_mask |= (1 << s)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try insertions: add a new element
            for x in range(256, 0, -1):
                if x in current:
                    continue
                if _is_valid_addition(current_sums_mask, x):
                    current.append(x)
                    current_sums_mask = _update_sums_mask(current_sums_mask, x)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: replace one element with another
                for i in range(len(current)):
                    if time.time() > deadline - 10:
                        break
                    old_val = current[i]
                    # Remove old_val from sums_mask
                    # This is complex; let's just rebuild for now if we find a swap
                    for new_val in range(256, 0, -1):
                        if new_val == old_val or new_val in current:
                            continue
                        candidate = current[:]
                        candidate[i] = new_val
                        # Try to add more after swap
                        cand_sums = 0
                        cn = len(candidate)
                        for mask in range(1, 1 << cn):
                            s = 0
                            for j in range(cn):
                                if mask & (1 << j):
                                    s += candidate[j]
                            if s <= MAX_S:
                                cand_sums |= (1 << s)
                        
                        # Check if swap is valid
                        valid = True
                        temp = cand_sums
                        while temp:
                            lsb = temp & (-temp)
                            a = lsb.bit_length() - 1
                            if a > MAX_S:
                                temp ^= lsb
                                continue
                            if cand_sums & div_mask[a]:
                                valid = False
                                break
                            temp ^= lsb
                        
                        if not valid:
                            continue
                        
                        # Try to add more
                        for add_val in range(256, 0, -1):
                            if add_val in candidate:
                                continue
                            if _is_valid_addition(cand_sums, add_val):
                                candidate.append(add_val)
                                break
                        
                        if len(candidate) > len(current):
                            current = candidate
                            current_sums_mask = 0
                            cn = len(current)
                            for mask in range(1, 1 << cn):
                                s = 0
                                for j in range(cn):
                                    if mask & (1 << j):
                                        s += current[j]
                                if s <= MAX_S:
                                    current_sums_mask |= (1 << s)
                            improved = True
                            break
                    if improved:
                        break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Final validation
    if best_set:
        if not _validate_full(best_set):
            # Fallback: try a known small valid set
            fallback = [256, 255, 254]
            if _validate_full(fallback):
                best_set = fallback
            else:
                best_set = [256, 255]
    
    return list(best_set)
