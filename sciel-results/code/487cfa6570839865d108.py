import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    N = 512
    
    # Precompute compatibility bitmasks
    # compat[i] is a bitmask where bit j is 1 if lcm(i, j) <= N
    compat = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                mask |= (1 << j)
        compat[i] = mask
    
    # Precompute blocks: elements not compatible with each i
    blocks = [0] * (N + 1)
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    for i in range(1, N + 1):
        blocks[i] = full_mask & ~compat[i]
        
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_mask(mask):
        # Check if the set represented by mask is a clique
        # For each i in mask, mask must be a subset of compat[i]
        # i.e., mask & ~compat[i] == 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            i = lsb.bit_length() - 1
            if mask & ~compat[i]:
                return False
            m ^= lsb
        return True

    def complete(mask):
        # Greedily add all compatible elements
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline - 1.0:
                    return mask
                bit = 1 << x
                if mask & bit:
                    continue
                # Check if x is compatible with all elements in mask
                if mask & ~compat[x]:
                    continue
                mask |= bit
                changed = True
        return mask

    def local_search(mask):
        mask = complete(mask)
        best_mask = mask
        best_size = popcount(mask)
        improved = True
        rng = random.Random(42)
        
        # Precompute for each element, the set of elements it's incompatible with
        # This helps in quickly finding candidates for swaps
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            m = mask
            elements = []
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # 1-for-1 swaps: remove u, add v
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                # Find all v compatible with mask_minus_u but not in mask
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if v == u or (mask & (1 << v)):
                        continue
                    if mask_minus_u & ~compat[v]:
                        continue
                    new_mask = complete(mask_minus_u | (1 << v))
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue
            
            # 1-for-2 swaps: remove u, add two elements greedily
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                new_mask = mask_minus_u
                added = 0
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if added >= 2:
                        break
                    if new_mask & (1 << v):
                        continue
                    if new_mask & ~compat[v]:
                        continue
                    new_mask |= (1 << v)
                    added += 1
                if added >= 2:
                    new_mask = complete(new_mask)
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2-for-3 swaps: remove two elements u1, u2, add three elements greedily
            for i in range(len(elements)):
                if time.time() > deadline - 2.0:
                    break
                u1 = elements[i]
                for j in range(i + 1, len(elements)):
                    if time.time() > deadline - 2.0:
                        break
                    u2 = elements[j]
                    mask_minus_2 = mask & ~(1 << u1) & ~(1 << u2)
                    new_mask = mask_minus_2
                    added = 0
                    for v in range(1, N + 1):
                        if time.time() > deadline - 2.0:
                            break
                        if added >= 3:
                            break
                        if new_mask & (1 << v):
                            continue
                        if new_mask & ~compat[v]:
                            continue
                        new_mask |= (1 << v)
                        added += 1
                    if added >= 3:
                        new_mask = complete(new_mask)
                        new_size = popcount(new_mask)
                        if new_size > best_size:
                            best_size = new_size
                            best_mask = new_mask
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break
            
            # Strategy 4: Bottleneck removal - remove element that blocks most additions
            if not improved:
                bottleneck_scores = []
                for u in elements:
                    blocked = blocks[u] & ~mask
                    score = popcount(blocked)
                    bottleneck_scores.append((score, u))
                bottleneck_scores.sort(reverse=True)
                
                for score, u in bottleneck_scores[:5]:
                    if time.time() > deadline - 2.0:
                        break
                    mask_minus_u = mask & ~(1 << u)
                    new_mask = complete(mask_minus_u)
                    new_size = popcount(new_mask)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        mask = new_mask
                        improved = True
                        break
        
        return best_mask

    # Generate multiple initial candidates
    candidates = []
    
    # 1. Divisors of highly composite numbers
    for M in [360, 420, 480, 504, 512, 240, 180, 120]:
        divs = []
        for i in range(1, M + 1):
            if M % i == 0:
                divs.append(i)
        mask = 0
        for d in divs:
            mask |= (1 << d)
        candidates.append(mask)
        
    # 2. Greedy from small to large
    mask = 0
    for i in range(1, N + 1):
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)
    
    # 3. Greedy from large to small
    mask = 0
    for i in range(N, 0, -1):
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)
    
    # 4. Random restarts with different strategies
    rng = random.Random(123)
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        strategy = rng.randint(0, 2)
        if strategy == 0:
            # Start with {1} and add randomly
            mask = 1 << 1
            order = list(range(2, N + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Start with small primes and build up
            mask = 0
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
                if p <= N and mask & ~compat[p] == 0:
                    mask |= (1 << p)
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            # Start with divisors of a random highly composite number
            M = rng.choice([120, 180, 240, 360, 420, 480, 504])
            mask = 0
            for d in range(1, M + 1):
                if M % d == 0:
                    mask |= (1 << d)
            order = list(range(1, N + 1))
            rng.shuffle(order)
        for x in order:
            if mask & (1 << x):
                continue
            if mask & ~compat[x] == 0:
                mask |= (1 << x)
        candidates.append(mask)
    
    # 5. Power-of-2 based candidates
    for base in [2, 3, 5, 7, 11, 13]:
        mask = 1 << 1
        for k in range(1, 12):
            val = base ** k
            if val > N:
                break
            if mask & ~compat[val] == 0:
                mask |= (1 << val)
        candidates.append(mask)
    
    # 6. Mixed prime power candidates
    for p1 in [2, 3, 5, 7]:
        for p2 in [2, 3, 5, 7]:
            if p1 == p2:
                continue
            mask = 1 << 1
            for k1 in range(1, 12):
                for k2 in range(1, 12):
                    val = (p1 ** k1) * (p2 ** k2)
                    if val > N:
                        break
                    if mask & ~compat[val] == 0:
                        mask |= (1 << val)
            candidates.append(mask)
    
    # 7. Greedy with different orderings based on compatibility density
    # Sort by number of compatible elements (ascending - most constrained first)
    compat_counts = [(popcount(compat[i]), i) for i in range(1, N + 1)]
    compat_counts.sort()
    mask = 0
    for _, i in compat_counts:
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)
    
    # Sort by number of compatible elements (descending - most flexible first)
    compat_counts.sort(reverse=True)
    mask = 0
    for _, i in compat_counts:
        if mask & ~compat[i] == 0:
            mask |= (1 << i)
    candidates.append(mask)

    best_overall_mask = 0
    best_overall_size = 0
    
    # Sort candidates by initial size (descending) to prioritize promising ones
    candidates.sort(key=lambda m: popcount(m), reverse=True)
    
    for mask in candidates:
        if time.time() > deadline - 3.0:
            break
        # Verify candidate is a valid clique before optimization
        if not is_valid_mask(mask):
            # Try to repair by removing conflicting elements
            m = mask
            while m:
                lsb = m & -m
                x = lsb.bit_length() - 1
                if mask & ~compat[x]:
                    mask &= ~lsb
                m ^= lsb
        
        optimized_mask = local_search(mask)
        size = popcount(optimized_mask)
        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = optimized_mask
            
    # Final polish: try to add any remaining compatible elements
    m = best_overall_mask
    for x in range(1, N + 1):
        if m & (1 << x):
            continue
        if m & ~compat[x] == 0:
            m |= (1 << x)
    
    # Convert best mask to list
    result = []
    mm = m
    while mm:
        lsb = mm & -mm
        result.append(lsb.bit_length() - 1)
        mm ^= lsb
        
    if not result:
        return list(range(1, 23))
        
    return sorted(result)

# EVOLVE-BLOCK-END
