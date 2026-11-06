def construct():
    import random
    import math
    import time

    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # bad_masks[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    # a, b, c are 1-indexed in logic, 0-indexed in bit position (c-1)
    bad_masks = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Precomputation
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = get_lcm(a, b)
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    mask |= (1 << (c - 1))
            bad_masks[a][b] = mask
            bad_masks[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def is_valid(S_mask):
        """Check if set represented by S_mask has no bad triples."""
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                if S_mask & bad_masks[a][b]:
                    return False
        return True

    def count_violations(S_mask):
        """Count the number of bad triples in S_mask."""
        count = 0
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                count += popcount(S_mask & bad_masks[a][b])
        return count // 3

    def energy(S_mask):
        size = popcount(S_mask)
        viol = count_violations(S_mask)
        return -size + 100 * viol

    def greedy_init():
        S_mask = 0
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            new_mask = S_mask | (1 << (x - 1))
            if is_valid(new_mask):
                S_mask = new_mask
        return S_mask

    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    max_time = 55.0
    
    rng = random.Random(42)
    
    S_mask = greedy_init()
    curr_energy = energy(S_mask)
    curr_size = popcount(S_mask)
    
    if curr_size > best_size:
        best_size = curr_size
        best_mask = S_mask
    
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        move_type = rng.random()
        if move_type < 0.4:
            candidates = [x for x in elements if not (S_mask & (1 << (x - 1)))]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_mask = S_mask | (1 << (x - 1))
        elif move_type < 0.8:
            candidates = [x for x in elements if (S_mask & (1 << (x - 1)))]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_mask = S_mask & ~(1 << (x - 1))
        else:
            in_candidates = [x for x in elements if (S_mask & (1 << (x - 1)))]
            out_candidates = [x for x in elements if not (S_mask & (1 << (x - 1)))]
            if not in_candidates or not out_candidates:
                continue
            y = rng.choice(in_candidates)
            x = rng.choice(out_candidates)
            new_mask = (S_mask & ~(1 << (y - 1))) | (1 << (x - 1))
            
        new_energy = energy(new_mask)
        delta_e = new_energy - curr_energy
        
        if delta_e < 0:
            S_mask = new_mask
            curr_energy = new_energy
            curr_size = popcount(S_mask)
            if curr_size > best_size:
                best_size = curr_size
                best_mask = S_mask
        else:
            if rng.random() < math.exp(-delta_e / T):
                S_mask = new_mask
                curr_energy = new_energy
                curr_size = popcount(S_mask)
                if curr_size > best_size:
                    best_size = curr_size
                    best_mask = S_mask
        
        T *= cooling
        if T < 0.01:
            S_mask = greedy_init()
            curr_energy = energy(S_mask)
            curr_size = popcount(S_mask)
            if curr_size > best_size:
                best_size = curr_size
                best_mask = S_mask
            T = 1.0

    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
