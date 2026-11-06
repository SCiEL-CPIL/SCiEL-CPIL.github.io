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
        
        # Precompute incompatibility masks for faster swap checks
        # incompat[i] = bitmask of elements j where lcm(i,j) > N
        incompat = [0] * (N + 1)
        for i in range(1, N + 1):
            incompat[i] = ~compat[i]
        
        # Precompute for each pair (u,v) whether they can be swapped
        # swap_ok[u][v] = True if removing u and adding v maintains clique property
        # (i.e., v is compatible with all elements in mask\{u})
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            m = mask
            elements = []
            while m:
                lsb = m & -m
                elements.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # 1-for-1 swaps: remove u, add v
            # For each u in the set, precompute the union of incompatibilities of all
            # other elements in the set.
            union_incompat = [0] * (N + 1)
            for u in elements:
                ui = 0
                for w in elements:
                    if w != u:
                        ui |= incompat[w]
                union_incompat[u] = ui
            
            # Try all 1-for-1 swaps
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                ui = union_incompat[u]
                mask_without_u = mask & ~(1 << u)
                for v in range(1, N + 1):
                    if time.time() > deadline - 2.0:
                        break
                    if v == u or (mask & (1 << v)):
                        continue
                    # Check if v is compatible with all elements in mask\{u}
                    if (ui | incompat[v]) & mask_without_u:
                        continue
                    new_mask = complete(mask_without_u | (1 << v))
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
                # Try to add elements in order of increasing value (more likely to fit)
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
            
            # 1-for-3 swaps: remove u, add three elements greedily
            for u in elements:
                if time.time() > deadline - 2.0:
                    break
                mask_minus_u = mask & ~(1 << u)
                new_mask = mask_minus_u
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
                continue
            
            # Random restarts within local search: perturb the current solution
            # by removing 2-3 random elements and refilling
            if rng.random() < 0.3:
                if len(elements) >= 3:
                    remove_count = min(3, len(elements) // 4)
                    to_remove = rng.sample(elements, remove_count)
                    perturbed = mask
                    for u in to_remove:
                        perturbed &= ~(1 << u)
                    perturbed = complete(perturbed)
                    perturbed_size = popcount(perturbed)
                    if perturbed_size > best_size - 2:  # Allow slight degradation
                        mask = perturbed
                        improved = True
                        continue
        
        # Final polish: repeatedly try removing one element and greedily refilling.
        # This can escape local optima where no single swap helps but removing a
        # "blocking" element and refilling yields a larger set.
        polish_rounds = 0
        while polish_rounds < 300 and time.time() < deadline - 1.0:
            polish_rounds += 1
            m = best_mask
            elems = []
            while m:
                lsb = m & -m
                elems.append(lsb.bit_length() - 1)
                m ^= lsb
            found = False
            # Try removing each element and refilling
            for u in elems:
                if time.time() > deadline - 1.0:
                    break
                candidate = best_mask & ~(1 << u)
                candidate = complete(candidate)
                sz = popcount(candidate)
                if sz > best_size:
                    best_size = sz
                    best_mask = candidate
                    found = True
                    break
            if not found:
                # Try removing pairs of elements
                for i in range(len(elems)):
                    if time.time() > deadline - 1.0:
                        break
                    u1 = elems[i]
                    for j in range(i + 1, len(elems)):
                        if time.time() > deadline - 1.0:
                            break
                        u2 = elems[j]
                        candidate = best_mask & ~(1 << u1) & ~(1 << u2)
                        candidate = complete(candidate)
                        sz = popcount(candidate)
                        if sz > best_size:
                            best_size = sz
                            best_mask = candidate
                            found = True
                            break
                    if found:
                        break
            if not found:
                break
        
        return best_mask

    # Generate multiple initial candidates
    candidates = []
    
    # 1. Divisors of highly composite numbers
    for M in [360, 420, 480, 504, 512, 240, 180, 120, 336, 288, 384, 432]:
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
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        if trial % 3 == 0:
            # Start with {1} and add randomly
            mask = 1 << 1
            order = list(range(2, N + 1))
            rng.shuffle(order)
            for x in order:
                if mask & ~compat[x] == 0:
                    mask |= (1 << x)
        elif trial % 3 == 1:
            # Start with a random small number and add randomly
            start_val = rng.randint(1, 10)
            mask = 1 << start_val
            order = [x for x in range(1, N + 1) if x != start_val]
            rng.shuffle(order)
            for x in order:
                if mask & ~compat[x] == 0:
                    mask |= (1 << x)
        else:
            # Start with a divisor-rich number and add randomly
            mask = 1 << 1
            # Add some divisors of 512 first
            for d in [2, 4, 8, 16, 32, 64, 128, 256]:
                if mask & ~compat[d] == 0:
                    mask |= (1 << d)
            order = [x for x in range(1, N + 1) if not (mask & (1 << x))]
            rng.shuffle(order)
            for x in order:
                if mask & ~compat[x] == 0:
                    mask |= (1 << x)
        candidates.append(mask)
    
    # 5. Power-of-2 based candidates: include all numbers with same prime power structure
    # Numbers of form 2^a * 3^b * 5^c * 7^d where lcm stays <= 512
    # Try maximal sets under divisibility constraints
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
    
    # 7. Numbers with many divisors (highly composite)
    highly_composite = [1, 2, 3, 4, 6, 8, 12, 16, 24, 30, 36, 48, 60, 72, 84, 96, 120, 144, 168, 180, 240, 360, 420, 480, 504]
    mask = 0
    for d in highly_composite:
        if d <= N and mask & ~compat[d] == 0:
            mask |= (1 << d)
    candidates.append(mask)
    
    # 8. All divisors of LCM of first k integers
    for k in [5, 6, 7, 8, 9, 10]:
        # LCM of 1..k
        l = 1
        for i in range(2, k + 1):
            l = l * i // gcd(l, i)
        if l <= N:
            divs = []
            for i in range(1, N + 1):
                if l % i == 0:
                    divs.append(i)
            mask = 0
            for d in divs:
                if mask & ~compat[d] == 0:
                    mask |= (1 << d)
            candidates.append(mask)

    best_overall_mask = 0
    best_overall_size = 0
    
    # Sort candidates by initial size (descending) to prioritize promising ones
    candidates.sort(key=lambda m: popcount(m), reverse=True)
    
    # Deduplicate candidates to avoid redundant work
    seen_masks = set()
    unique_candidates = []
    for mask in candidates:
        if mask not in seen_masks:
            seen_masks.add(mask)
            unique_candidates.append(mask)
    candidates = unique_candidates
    
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
            # If we've found a very good solution, we can be more aggressive with time
            if best_overall_size >= 50:
                deadline = start_time + 58.0
    
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
