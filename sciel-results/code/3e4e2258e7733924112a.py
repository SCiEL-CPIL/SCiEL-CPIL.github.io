import itertools
import random
import time
import math

def construct(q, n, radius):
    # Only optimized for the specific task instance, fallback to simple method otherwise
    if q == 3 and n == 6 and radius == 1:
        return _solve_3_6_1_sa()
    else:
        # Fallback to a simple greedy for other params to avoid breaking interface
        return _generic_construct(q, n, radius)

def _solve_3_6_1_sa():
    q, n, radius = 3, 6, 1
    total = q ** n
    
    # Helper to decode
    def decode(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w

    # Helper to encode
    def encode(w):
        val = 0
        for d in w:
            val = val * q + d
        return val

    # Precompute masks
    masks = [0] * total
    for c in range(total):
        center = decode(c)
        mask = 0
        mask |= (1 << c)
        for pos in range(n):
            orig = center[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_center = center[:]
                    new_center[pos] = new_val
                    nb = encode(new_center)
                    mask |= (1 << nb)
        masks[c] = mask
        
    total_mask = (1 << total) - 1
    
    # 1. Greedy Initialization to get a valid starting point
    uncovered = total_mask
    init_codewords = []
    while uncovered:
        best_cw = -1
        best_count = -1
        for cw in range(total):
            count = (masks[cw] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_cw = cw
        init_codewords.append(best_cw)
        uncovered &= ~masks[best_cw]
        
    # Simple reduction on init
    changed = True
    while changed:
        changed = False
        for i in range(len(init_codewords)):
            others = [c for j, c in enumerate(init_codewords) if j != i]
            covered = 0
            for c in others:
                covered |= masks[c]
            if covered == total_mask:
                init_codewords.pop(i)
                changed = True
                break

    # 2. Simulated Annealing
    # State: set of active codewords
    # We represent state as a list of active codeword indices.
    # To make moves fast, we can use a set for O(1) lookup, but list is fine for ~70 items.
    
    current_code = init_codewords[:]
    current_active_set = set(current_code)
    
    # Calculate initial coverage
    covered_mask = 0
    for c in current_code:
        covered_mask |= masks[c]
    uncovered_mask = total_mask & ~covered_mask
    current_uncovered_count = uncovered_mask.bit_count()
    current_count = len(current_code)
    
    # Cost function: Size + Penalty * Uncovered
    # We want to minimize size, but we must end with 0 uncovered.
    # High penalty forces validity.
    penalty = 1000.0 
    
    def get_cost(count, unc_count):
        return count + penalty * unc_count

    current_cost = get_cost(current_count, current_uncovered_count)
    
    best_valid_code = current_code[:]
    best_valid_size = current_count
    
    start_time = time.time()
    max_time = 110.0
    T = 50.0
    T_min = 0.01
    cooling = 0.99
    
    random.seed(42) # Deterministic for reproducibility in testing, though task allows random
    
    while time.time() - start_time < max_time and T > T_min:
        # Generate a neighbor
        move_type = random.random()
        
        new_code = current_code[:]
        new_active_set = current_active_set.copy()
        
        if move_type < 0.33:
            # Remove a random codeword
            if len(new_code) > 0:
                idx = random.randint(0, len(new_code) - 1)
                cw_to_remove = new_code[idx]
                new_code.pop(idx)
                new_active_set.remove(cw_to_remove)
        elif move_type < 0.66:
            # Add a random codeword
            # Pick a random word not in active set
            # Efficient way: pick random 0..total-1, if in set, retry or pick another
            # Since set is small, retry is fast
            while True:
                cw_to_add = random.randint(0, total - 1)
                if cw_to_add not in new_active_set:
                    break
            new_code.append(cw_to_add)
            new_active_set.add(cw_to_add)
        else:
            # Swap
            if len(new_code) > 0:
                idx = random.randint(0, len(new_code) - 1)
                cw_to_remove = new_code[idx]
                while True:
                    cw_to_add = random.randint(0, total - 1)
                    if cw_to_add not in new_active_set:
                        break
                new_code[idx] = cw_to_add
                new_active_set.remove(cw_to_remove)
                new_active_set.add(cw_to_add)
                
        # Calculate new cost
        # Recalculate covered mask
        new_covered_mask = 0
        for c in new_code:
            new_covered_mask |= masks[c]
            
        new_uncovered_mask = total_mask & ~new_covered_mask
        new_uncovered_count = new_uncovered_mask.bit_count()
        new_count = len(new_code)
        
        new_cost = get_cost(new_count, new_uncovered_count)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (random.random() < math.exp(-delta / T)):
            # Accept move
            current_code = new_code
            current_active_set = new_active_set
            current_count = new_count
            current_uncovered_count = new_uncovered_count
            current_cost = new_cost
            
            # Check if valid and better
            if current_uncovered_count == 0 and current_count < best_valid_size:
                best_valid_size = current_count
                best_valid_code = current_code[:]
                
        T *= cooling
        
    # Final reduction on best valid code
    # The SA might leave redundant codewords in the best valid code
    final_code = best_valid_code[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(final_code)):
            others = [c for j, c in enumerate(final_code) if j != i]
            covered = 0
            for c in others:
                covered |= masks[c]
            if covered == total_mask:
                final_code.pop(i)
                changed = True
                break
                
    return [decode(c) for c in final_code]

def _generic_construct(q, n, radius):
    # Simple greedy for other cases
    total = q ** n
    
    def decode(val):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = val % q
            val //= q
        return w
        
    def encode(w):
        val = 0
        for d in w:
            val = val * q + d
        return val
        
    masks = [0] * total
    for c in range(total):
        center = decode(c)
        mask = 0
        if radius == 1:
            mask |= (1 << c)
            for pos in range(n):
                orig = center[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_center = center[:]
                        new_center[pos] = new_val
                        nb = encode(new_center)
                        mask |= (1 << nb)
        else:
            for word in itertools.product(range(q), repeat=n):
                if sum(1 for a,b in zip(center, word) if a!=b) <= radius:
                    mask |= (1 << encode(word))
        masks[c] = mask
        
    total_mask = (1 << total) - 1
    uncovered = total_mask
    codewords = []
    
    while uncovered:
        best_cw = -1
        best_count = -1
        for cw in range(total):
            count = (masks[cw] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_cw = cw
        codewords.append(best_cw)
        uncovered &= ~masks[best_cw]
        
    # Reduction
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords)):
            others = [c for j, c in enumerate(codewords) if j != i]
            covered = 0
            for c in others:
                covered |= masks[c]
            if covered == total_mask:
                codewords.pop(i)
                changed = True
                break
                
    return [decode(c) for c in codewords]
