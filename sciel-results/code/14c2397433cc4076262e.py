import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability more efficiently
    # For each mask, precompute which masks it's comparable with
    # comp[a] is a set of masks b where a and b are comparable
    comp = [None] * num_masks
    for a in range(num_masks):
        comp[a] = bytearray(num_masks)
    
    # More efficient precomputation using subset relationships
    # A mask a is comparable with b iff a⊆b or b⊆a
    # We can precompute for each a, the set of all supersets and subsets
    for a in range(num_masks):
        # All subsets of a
        sub = a
        while True:
            comp[a][sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & a
        # All supersets of a
        sup = a
        while True:
            comp[a][sup] = 1
            if sup == max_mask:
                break
            # Find next superset: add a bit that's not in sup
            # This is tricky; let's use a different approach
            # Actually, let's just check all b for each a - it's O(2^n * 2^n) = 16M which is fine
            break
    
    # Actually, let's use a simpler but still efficient approach
    # Precompute a lookup table: for each pair (a,b), is a comparable with b?
    # Use a flat array for cache efficiency
    comp_flat = bytearray(num_masks * num_masks)
    
    for a in range(num_masks):
        for b in range(num_masks):
            if (a & b) == a or (a & b) == b:
                comp_flat[a * num_masks + b] = 1
    
    def is_comparable(a, b):
        return comp_flat[a * num_masks + b] == 1
    
    def calc_score(masks):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(masks[i], masks[j]):
                    score += 1
        return score
    
    # Greedy initialization
    masks = []
    used = set()
    
    # Start with a full chain
    cur = 0
    for i in range(n + 1):
        masks.append(cur)
        used.add(cur)
        cur |= (1 << i)
    
    # Greedily add remaining sets - use a faster scoring method
    while len(masks) < m and time.time() < deadline - 5.0:
        best_candidate = None
        best_score = -1
        # Sample candidates rather than checking all 4096
        num_samples = min(200, num_masks - len(used))
        sampled = set()
        for _ in range(num_samples):
            candidate = random.randint(0, max_mask)
            if candidate not in used and candidate not in sampled:
                sampled.add(candidate)
        
        # Also try some structured candidates
        for existing in masks:
            # Try adding one element to existing
            for bit in range(n):
                candidate = existing | (1 << bit)
                if candidate not in used and candidate not in sampled:
                    sampled.add(candidate)
            # Try removing one element from existing
            if existing != 0:
                for bit in range(n):
                    if (existing >> bit) & 1:
                        candidate = existing & ~(1 << bit)
                        if candidate not in used and candidate not in sampled:
                            sampled.add(candidate)
        
        for candidate in sampled:
            s = 0
            for existing in masks:
                if is_comparable(candidate, existing):
                    s += 1
            if s > best_score:
                best_score = s
                best_candidate = candidate
        
        if best_candidate is not None:
            masks.append(best_candidate)
            used.add(best_candidate)
        else:
            break
    
    # If we didn't reach m, fill randomly
    while len(masks) < m:
        r = random.randint(0, max_mask)
        if r not in used:
            masks.append(r)
            used.add(r)
    
    current_score = calc_score(masks)
    best_score = current_score
    best_masks = masks[:]
    
    # Simulated Annealing with better mutation strategy
    T = 10.0
    cooling = 0.9999
    min_T = 0.001
    
    random.seed(42)
    
    # Precompute for each mask, a list of "nearby" masks for faster mutation
    # Nearby = differ by 1-2 bits
    nearby = [[] for _ in range(num_masks)]
    for a in range(num_masks):
        # Differ by 1 bit
        for bit in range(n):
            b = a ^ (1 << bit)
            nearby[a].append(b)
        # Differ by 2 bits
        for bit1 in range(n):
            for bit2 in range(bit1 + 1, n):
                b = a ^ (1 << bit1) ^ (1 << bit2)
                nearby[a].append(b)
    
    while time.time() < deadline and T > min_T:
        # Pick random index
        i = random.randint(0, m - 1)
        old_mask = masks[i]
        
        # Generate candidate with bias toward nearby masks
        r = random.random()
        if r < 0.7:
            # Pick from nearby masks
            if nearby[old_mask]:
                new_mask = random.choice(nearby[old_mask])
            else:
                new_mask = random.randint(0, max_mask)
        elif r < 0.9:
            # Medium: flip 3-4 bits
            new_mask = old_mask
            for _ in range(random.randint(3, 4)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        else:
            # Global: random mask
            new_mask = random.randint(0, max_mask)
            
        if new_mask == old_mask or new_mask in used:
            continue
             
        # Calculate Delta efficiently
        delta = 0
        for j in range(m):
            if j == i:
                continue
            m_j = masks[j]
            old_c = comp_flat[old_mask * num_masks + m_j]
            new_c = comp_flat[new_mask * num_masks + m_j]
            delta += new_c - old_c
            
        # Accept?
        if delta > 0:
            accept = True
        else:
            # Use exp(delta/T) for acceptance probability
            # Clamp to avoid underflow
            exponent = max(-50.0, delta / T)
            accept = random.random() < math.exp(exponent)
        
        if accept:
            used.remove(old_mask)
            used.add(new_mask)
            masks[i] = new_mask
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = masks[:]
            
            T *= cooling
    
    # Final local search polish pass
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(m):
                old_mask = masks[i]
                best_local = old_mask
                best_local_delta = 0
                
                # Try all nearby masks
                for candidate in nearby[old_mask]:
                    if candidate in used or candidate == old_mask:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        m_j = masks[j]
                        old_c = comp_flat[old_mask * num_masks + m_j]
                        new_c = comp_flat[candidate * num_masks + m_j]
                        delta += new_c - old_c
                    
                    if delta > best_local_delta:
                        best_local_delta = delta
                        best_local = candidate
                
                if best_local != old_mask and best_local_delta > 0:
                    used.remove(old_mask)
                    used.add(best_local)
                    masks[i] = best_local
                    current_score += best_local_delta
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_masks = masks[:]
    
    # Final conversion
    result = []
    for mask in best_masks:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
