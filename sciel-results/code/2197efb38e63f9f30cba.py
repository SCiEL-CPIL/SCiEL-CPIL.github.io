import random
import time

def _count_violations(mask):
    """
    Counts the number of triples {a,b,c} with a<b<c such that
    a,b,c are in mask and a+b, a+c, b+c are in mask.
    """
    violations = 0
    # a from 1 to 126 (since b>a, c>b, max c=128, min b=2, min a=1)
    # Actually a can go up to 126 if b=127, c=128?
    # a+b <= 128 => a <= 128-b.
    # If b is large, a is small.
    
    # Precompute bit masks for "greater than b"
    # mask_gt[b] has bits b+1..128 set
    
    for a in range(1, 127): # a must be < b < c <= 128
        if not (mask >> a) & 1:
            continue
        
        # b must be > a
        # Also a+b must be <= 128 for a+b to be in mask (mask only has bits up to 128)
        # If a+b > 128, it's not in mask, so no violation for this pair.
        max_b = 128 - a
        if max_b < a + 1:
            continue
            
        for b in range(a + 1, max_b + 1):
            if not (mask >> b) & 1:
                continue
            
            # Check if a+b is in mask
            if not (mask >> (a + b)) & 1:
                continue
            
            # Find c > b such that a+c in mask and b+c in mask
            # candidates = (mask >> a) & (mask >> b) & mask
            # We need c > b.
            
            # Optimization:
            # If we precompute mask_a = mask >> a
            # candidates = mask_a & (mask >> b) & mask
            
            # Let's just do it directly
            candidates = (mask >> a) & (mask >> b) & mask
            
            # Mask out c <= b
            # We want bits b+1 to 128.
            # ~((1 << (b+1)) - 1) creates a mask with 1s from b+1 upwards.
            # But Python ints are infinite precision.
            # We need to be careful with ~.
            # Better: candidates & (mask & ~((1 << (b+1)) - 1))
            # Or simply:
            # Check if candidates has any bit set at index > b.
            
            # A faster way to check if any bit > b is set:
            # If candidates & (~((1 << (b+1)) - 1)) != 0
            # But ~((1 << (b+1)) - 1) is negative.
            # In Python, bitwise AND with negative works as expected for two's complement infinite bits?
            # Yes, but let's use a positive mask.
            # mask_high = ((1 << 129) - 1) ^ ((1 << (b+1)) - 1)
            # This is expensive to compute inside loop.
            
            # Alternative:
            # If candidates is 0, continue.
            # Else, check if the lowest set bit is > b? No, we need ANY set bit > b.
            # We can just check:
            # if candidates >> (b+1):
            #   This shifts out bits 0..b. If result is non-zero, there is a bit > b.
            #   But we also need to count them.
            
            if candidates >> (b + 1):
                # Count bits in candidates that are > b
                # candidates_gt_b = candidates >> (b+1)
                # violations += candidates_gt_b.bit_count()
                # But we only care about bits up to 128.
                # candidates already only has bits up to 128 (from mask).
                # So candidates >> (b+1) will have bits 0..127-b.
                # bit_count() counts all set bits.
                violations += (candidates >> (b + 1)).bit_count()

    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_mask = 0
    best_size = 0
    
    # Helper to update best
    def try_update(mask):
        nonlocal best_mask, best_size
        if _count_violations(mask) == 0:
            size = mask.bit_count()
            if size > best_size:
                best_size = size
                best_mask = mask

    # 1. Seed Init
    seed_mask = 0
    for i in range(65, 129):
        seed_mask |= (1 << i)
    try_update(seed_mask)
    
    # 2. Random Inits
    random.seed(42) # For reproducibility in debugging, but evaluator runs fresh?
    # Actually, random seed should be time based or fixed?
    # Fixed is fine, but different restarts need variation.
    
    inits = [seed_mask]
    for _ in range(10):
        m = 0
        for i in range(1, 129):
            if random.random() < 0.5:
                m |= (1 << i)
        inits.append(m)
        
    # 3. Greedy Inits
    for _ in range(5):
        m = 0
        elems = list(range(1, 129))
        random.shuffle(elems)
        for e in elems:
            new_m = m | (1 << e)
            # Only add if it doesn't create violations?
            # This makes init slow if we check every time.
            # Let's just add all, then let SA fix.
            # Or, add if valid.
            # _count_violations is fast enough for 128 steps * 5 inits.
            if _count_violations(new_m) == 0:
                m = new_m
        inits.append(m)
        try_update(m)

    for init_mask in inits:
        if time.time() > deadline:
            break
            
        mask = init_mask
        current_viol = _count_violations(mask)
        
        T = 5.0
        best_local_mask = mask
        best_local_viol = current_viol
        best_local_size = mask.bit_count()
        
        if best_local_viol == 0 and best_local_size > best_size:
            best_mask = best_local_mask
            best_size = best_local_size

        # SA Loop
        # Estimate: 10000 steps per init.
        # 15 inits * 10000 = 150k steps.
        # 150k * 8k ops = 1.2B ops. Too slow.
        
        # Reduce steps.
        # Let's do 2000 steps per init.
        # 15 * 2000 = 30k steps.
        # 30k * 8k = 240M ops. ~10-20s. OK.
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            i = random.randint(1, 128)
            bit = 1 << i
            new_mask = mask ^ bit
            
            new_viol = _count_violations(new_mask)
            
            if new_viol < current_viol:
                mask = new_mask
                current_viol = new_viol
                if current_viol == 0:
                    size = mask.bit_count()
                    if size > best_local_size:
                        best_local_mask = mask
                        best_local_size = size
                        best_local_viol = 0
            else:
                # Accept with probability
                delta = new_viol - current_viol
                if delta == 0:
                    # Prefer larger size?
                    if new_mask.bit_count() > mask.bit_count():
                        mask = new_mask
                        current_viol = new_viol
                else:
                    # Boltzmann factor
                    # exp(-delta / T)
                    # Use math.exp? Or approximate?
                    # T is small, delta is small.
                    if random.random() < pow(2.718, -delta / T):
                        mask = new_mask
                        current_viol = new_viol

            T *= 0.9995 # Slow cooling

        # End SA
        if best_local_viol == 0 and best_local_size > best_size:
            best_mask = best_local_mask
            best_size = best_local_size

    # Fallback: If best_mask is 0 (should not happen as seed is valid), return seed
    if best_mask == 0:
        best_mask = seed_mask

    # Convert to list
    result = []
    for i in range(1, 129):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
