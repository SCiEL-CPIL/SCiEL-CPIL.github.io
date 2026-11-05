import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute reciprocals
    recip = [0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i
    
    # Precompute AP pairs for each x
    ap_pairs = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the middle: y, x, z -> y + z = 2x
        for y in range(1, x):
            z = 2 * x - y
            if z <= n and z > x:
                ap_pairs[x].append((y, z))
        
        # x is an endpoint: x, y, z -> z = 2y - x
        for y in range(x + 1, n + 1):
            z = 2 * y - x
            if z <= n and z > y:
                ap_pairs[x].append((y, z))
                
        # x is the other endpoint: y, z, x -> y = 2z - x
        for z in range(1, x):
            y = 2 * z - x
            if y >= 1 and y < z:
                ap_pairs[x].append((y, z))

    # Precompute separate lists for faster access
    ap_pairs_y = [list() for _ in range(n + 1)]
    ap_pairs_z = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y, z in ap_pairs[x]:
            ap_pairs_y[x].append(y)
            ap_pairs_z[x].append(z)
    
    def can_add(x, mask):
        """Check if x can be added to the set represented by mask."""
        ys = ap_pairs_y[x]
        zs = ap_pairs_z[x]
        for i in range(len(ys)):
            if ((mask >> ys[i]) & 1) and ((mask >> zs[i]) & 1):
                return False
        return True

    def get_sum(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += recip[idx]
            m ^= lsb
        return total

    def get_exact_sum(mask):
        """Compute exact sum using Fraction for high-precision comparison."""
        total = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            total += Fraction(1, idx)
            m ^= lsb
        return total

    def greedy_fill(mask):
        for x in range(1, n + 1):
            if not ((mask >> x) & 1):
                if can_add(x, mask):
                    mask |= (1 << x)
        return mask

    def local_search(mask, use_exact=False):
        improved = True
        current_sum = get_exact_sum(mask) if use_exact else get_sum(mask)
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try remove-and-refill
            for rem in range(1, n + 1):
                if not ((mask >> rem) & 1):
                    continue
                
                trial_mask = mask & ~(1 << rem)
                trial_mask = greedy_fill(trial_mask)
                
                if use_exact:
                    new_sum = get_exact_sum(trial_mask)
                else:
                    new_sum = get_sum(trial_mask)
                
                if new_sum > current_sum + (Fraction(0,1) if use_exact else 1e-12):
                    mask = trial_mask
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try swap: remove one element, add another
            if not improved:
                elements_in_mask = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    elements_in_mask.append(idx)
                    m ^= lsb
                
                for rem in elements_in_mask:
                    if time.time() > deadline - 3.0:
                        break
                    trial_mask = mask & ~(1 << rem)
                    # Try adding each non-present element
                    for add in range(1, n + 1):
                        if (trial_mask >> add) & 1:
                            continue
                        if can_add(add, trial_mask):
                            trial2 = trial_mask | (1 << add)
                            if use_exact:
                                new_sum = get_exact_sum(trial2)
                            else:
                                new_sum = get_sum(trial2)
                            if new_sum > current_sum + (Fraction(0,1) if use_exact else 1e-12):
                                mask = trial2
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
            
            # Try multi-swap: remove two elements, add two different elements
            if not improved and time.time() < deadline - 5.0:
                elements_in_mask = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    elements_in_mask.append(idx)
                    m ^= lsb
                
                for i in range(len(elements_in_mask)):
                    if time.time() > deadline - 5.0:
                        break
                    rem1 = elements_in_mask[i]
                    for j in range(i + 1, len(elements_in_mask)):
                        rem2 = elements_in_mask[j]
                        trial_mask = mask & ~(1 << rem1) & ~(1 << rem2)
                        # Try adding two non-present elements
                        for add1 in range(1, n + 1):
                            if (trial_mask >> add1) & 1:
                                continue
                            if not can_add(add1, trial_mask):
                                continue
                            trial_mask1 = trial_mask | (1 << add1)
                            for add2 in range(1, n + 1):
                                if (trial_mask1 >> add2) & 1:
                                    continue
                                if not can_add(add2, trial_mask1):
                                    continue
                                trial2 = trial_mask1 | (1 << add2)
                                if use_exact:
                                    new_sum = get_exact_sum(trial2)
                                else:
                                    new_sum = get_sum(trial2)
                                if new_sum > current_sum + (Fraction(0,1) if use_exact else 1e-12):
                                    mask = trial2
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return mask

    # 1. Initial Greedy
    initial_mask = 0
    best_mask = greedy_fill(initial_mask)
    best_sum = get_sum(best_mask)
    
    # 2. Local Search on Initial
    improved_mask = local_search(best_mask)
    if get_sum(improved_mask) > best_sum:
        best_mask = improved_mask
        best_sum = get_sum(best_mask)
        
    # 3. Randomized Restart
    random.seed(42)
    
    while time.time() < deadline - 8.0:
        perturbed_mask = best_mask
        
        # Vary the number of removals based on time remaining
        time_left = deadline - 8.0 - time.time()
        if time_left > 30.0:
            num_removals = random.randint(1, 5)
        else:
            num_removals = random.randint(1, 3)
        
        elements_in_set = []
        m = perturbed_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elements_in_set.append(idx)
            m ^= lsb
            
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_mask &= ~(1 << rem_val)
            elements_in_set.pop(idx)
        
        perturbed_mask = greedy_fill(perturbed_mask)
        improved_mask = local_search(perturbed_mask)
        
        new_sum = get_sum(improved_mask)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_mask = improved_mask

    # 4. High-precision polish pass with exact arithmetic
    if time.time() < deadline - 2.0:
        # Do multiple polish passes if time allows
        polished_mask = best_mask
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            next_polished = local_search(polished_mask, use_exact=True)
            exact_current = get_exact_sum(polished_mask)
            exact_next = get_exact_sum(next_polished)
            if exact_next > exact_current:
                polished_mask = next_polished
            else:
                break
        exact_best = get_exact_sum(best_mask)
        exact_polished = get_exact_sum(polished_mask)
        if exact_polished > exact_best:
            best_mask = polished_mask

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    # Validation
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
                    
    return result
