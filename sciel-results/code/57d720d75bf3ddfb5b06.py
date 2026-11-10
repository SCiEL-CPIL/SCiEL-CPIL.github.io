import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    max_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: comp[a][b] = 1 if a and b are comparable
    # Use a list of bytearrays for memory efficiency
    comp = [bytearray(num_masks) for _ in range(num_masks)]
    for a in range(num_masks):
        for b in range(a + 1, num_masks):
            if (a & b) == a or (a & b) == b:
                comp[a][b] = 1
                comp[b][a] = 1
    
    def calc_score(masks):
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comp[masks[i]][masks[j]]:
                    score += 1
        return score
    
    # Greedy initialization: build a chain first, then greedily add sets
    # that maximize comparability with existing sets
    masks = []
    used = set()
    
    # Start with a full chain: ∅, {0}, {0,1}, ..., {0,1,2,...,11}
    cur = 0
    for i in range(n + 1):
        masks.append(cur)
        used.add(cur)
        cur |= (1 << i)
    
    # Greedily add remaining sets
    while len(masks) < m and time.time() < deadline - 5.0:
        best_candidate = None
        best_score = -1
        for candidate in range(num_masks):
            if candidate in used:
                continue
            # Calculate how many existing sets are comparable with candidate
            s = 0
            for existing in masks:
                if comp[candidate][existing]:
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
    
    # Simulated Annealing with precomputed comparability
    T = 15.0
    cooling = 0.99995
    min_T = 0.01
    
    random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Pick random index
        i = random.randint(0, m - 1)
        old_mask = masks[i]
        
        # Generate candidate
        if random.random() < 0.6:
            # Local mutation: flip 1-2 bits
            new_mask = old_mask
            for _ in range(random.randint(1, 2)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        elif random.random() < 0.8:
            # Medium: flip 3-5 bits
            new_mask = old_mask
            for _ in range(random.randint(3, 5)):
                bit = random.randint(0, n - 1)
                new_mask ^= (1 << bit)
        else:
            # Global: random mask
            new_mask = random.randint(0, max_mask)
            
        if new_mask == old_mask:
            continue
            
        if new_mask in used:
            continue
             
        # Calculate Delta using precomputed comparability
        delta = 0
        for j in range(m):
            if j == i:
                continue
            m_j = masks[j]
            old_c = comp[old_mask][m_j]
            new_c = comp[new_mask][m_j]
            delta += new_c - old_c
            
        # Accept?
        if delta > 0 or random.random() < (2.71828 ** (delta / T)):
            # Accept
            used.remove(old_mask)
            used.add(new_mask)
            masks[i] = new_mask
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = masks[:]
            
            T *= cooling
    
    # Final conversion
    result = []
    for mask in best_masks:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
