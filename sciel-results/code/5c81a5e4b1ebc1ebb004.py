import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    n_max = 256

    # Precompute conflict masks: bit j set in mask[i] means i and j conflict
    conflict_mask = [0] * (n_max + 1)
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Compute 2-adic valuation for structural understanding
    def v2(n):
        c = 0
        while n % 2 == 0:
            c += 1
            n //= 2
        return c

    v2_val = [0] * (n_max + 1)
    for i in range(1, n_max + 1):
        v2_val[i] = v2(i)

    def popcount(x):
        return x.bit_count()

    def can_add(mask, x):
        return (mask & conflict_mask[x]) == 0

    # Greedy completion: add all possible elements iteratively until fixpoint
    def greedy_complete(mask):
        changed = True
        while changed:
            changed = False
            for i in range(1, n_max + 1):
                bit = 1 << i
                if not (mask & bit):
                    if (mask & conflict_mask[i]) == 0:
                        mask |= bit
                        changed = True
        return mask

    # Multi-start greedy to find good initial solution
    all_elements = list(range(1, n_max + 1))
    evens = list(range(2, n_max + 1, 2))
    odds = list(range(1, n_max + 1, 2))
    
    # Partition by 2-adic valuation for structured search
    by_v2 = {}
    for i in range(1, n_max + 1):
        v = v2_val[i]
        if v not in by_v2:
            by_v2[v] = []
        by_v2[v].append(i)

    random.seed(42)
    best_mask = 0
    best_size = 0

    # Start from all odds (known good base)
    base_mask = 0
    for i in odds:
        base_mask |= (1 << i)
    best_mask = greedy_complete(base_mask)
    best_size = popcount(best_mask)

    # Try structured greedy: process by 2-adic valuation groups
    for _ in range(30):
        if time.time() > deadline - 15.0:
            break
        cur = 0
        # Randomly order the v2 groups
        v2_keys = sorted(by_v2.keys())
        random.shuffle(v2_keys)
        for v in v2_keys:
            group = by_v2[v][:]
            random.shuffle(group)
            for e in group:
                if can_add(cur, e):
                    cur |= (1 << e)
        cur = greedy_complete(cur)
        sz = popcount(cur)
        if sz > best_size:
            best_size = sz
            best_mask = cur

    # Try multiple random greedy orderings
    for _ in range(40):
        if time.time() > deadline - 15.0:
            break
        cur = 0
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        for e in shuffled:
            if can_add(cur, e):
                cur |= (1 << e)
        cur = greedy_complete(cur)
        sz = popcount(cur)
        if sz > best_size:
            best_size = sz
            best_mask = cur

    # Also try starting from random subsets of various sizes
    for trial_size in [64, 96, 128]:
        for _ in range(10):
            if time.time() > deadline - 15.0:
                break
            cur = 0
            sample = random.sample(all_elements, min(trial_size, n_max))
            for e in sample:
                if can_add(cur, e):
                    cur |= (1 << e)
            cur = greedy_complete(cur)
            sz = popcount(cur)
            if sz > best_size:
                best_size = sz
                best_mask = cur

    # Simulated Annealing with optimized move generation
    current_mask = best_mask
    current_size = best_size

    T = 5.0
    T_min = 0.001
    cooling_rate = 0.997  # Slower cooling for more thorough exploration
    steps = 0

    # Precompute list of elements for faster iteration
    elements_list = list(range(1, n_max + 1))
    
    while time.time() < deadline - 8.0 and T > T_min:
        steps += 1
        r = random.random()
        new_mask = current_mask
        delta = 0

        if r < 0.30:
            # Add a random element
            x = random.choice(elements_list)
            bit_x = 1 << x
            if not (current_mask & bit_x):
                if (current_mask & conflict_mask[x]) == 0:
                    new_mask = current_mask | bit_x
                    delta = 1
        elif r < 0.50:
            # Remove a random element
            x = random.choice(elements_list)
            bit_x = 1 << x
            if current_mask & bit_x:
                new_mask = current_mask & ~bit_x
                delta = -1
        elif r < 0.80:
            # Swap: remove x, add y
            x = random.choice(elements_list)
            bit_x = 1 << x
            if current_mask & bit_x:
                y = random.choice(elements_list)
                bit_y = 1 << y
                if not (current_mask & bit_y):
                    mask_without = current_mask & ~bit_x
                    if (mask_without & conflict_mask[y]) == 0:
                        new_mask = mask_without | bit_y
                        delta = 0
        else:
            # Double swap: remove x1, x2, add y1, y2
            x1 = random.choice(elements_list)
            bit_x1 = 1 << x1
            if current_mask & bit_x1:
                x2 = random.choice(elements_list)
                bit_x2 = 1 << x2
                if current_mask & bit_x2 and x2 != x1:
                    y1 = random.choice(elements_list)
                    bit_y1 = 1 << y1
                    if not (current_mask & bit_y1):
                        y2 = random.choice(elements_list)
                        bit_y2 = 1 << y2
                        if not (current_mask & bit_y2) and y2 != y1:
                            mask_without = current_mask & ~bit_x1 & ~bit_x2
                            if (mask_without & conflict_mask[y1]) == 0 and (mask_without & conflict_mask[y2]) == 0:
                                new_mask = mask_without | bit_y1 | bit_y2
                                delta = 0

        new_size = new_mask.bit_count()

        if new_size > current_size:
            current_mask = new_mask
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        elif new_size < current_size:
            # Accept worse moves with probability exp(delta/T)
            if random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_size = new_size
        else:
            # Same size: accept to explore
            current_mask = new_mask

        T *= cooling_rate

        # Periodic greedy completion
        if steps % 15 == 0:
            completed = greedy_complete(current_mask)
            csz = completed.bit_count()
            if csz > current_size:
                current_mask = completed
                current_size = csz
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask

    # Final local search: systematic 1-swap improvement
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for x in range(1, n_max + 1):
            if time.time() > deadline - 3.0:
                break
            bit_x = 1 << x
            if not (best_mask & bit_x):
                continue
            mask_without_x = best_mask & ~bit_x
            for y in range(1, n_max + 1):
                if time.time() > deadline - 3.0:
                    break
                bit_y = 1 << y
                if best_mask & bit_y:
                    continue
                if (mask_without_x & conflict_mask[y]) == 0:
                    candidate = mask_without_x | bit_y
                    cand_size = candidate.bit_count()
                    if cand_size > best_size:
                        best_mask = candidate
                        best_size = cand_size
                        improved = True
                        break
                    elif cand_size == best_size:
                        completed = greedy_complete(candidate)
                        csz = completed.bit_count()
                        if csz > best_size:
                            best_mask = completed
                            best_size = csz
                            improved = True
                            break

    # Final greedy completion
    final_mask = greedy_complete(best_mask)
    
    # Convert to sorted list
    result = []
    for i in range(1, n_max + 1):
        if final_mask & (1 << i):
            result.append(i)

    return result
