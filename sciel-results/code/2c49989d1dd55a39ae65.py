import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 256
    recip_float = [0.0] * (N + 1)
    recip_frac = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip_float[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)

    # Helper functions for bitmask
    def bit_set(mask, i):
        return (mask >> (i - 1)) & 1
    
    def set_bit(mask, i):
        return mask | (1 << (i - 1))
        
    def clear_bit(mask, i):
        return mask & ~(1 << (i - 1))

    def get_elements(mask):
        elems = []
        m = mask
        while m:
            lsb = m & -m
            elems.append(lsb.bit_length())
            m &= m - 1
        return elems

    def is_valid_addition(mask, x):
        if bit_set(mask, x):
            return False
        # Case 1: x is middle. a, x, b => b = 2x - a
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length()
            b = 2 * x - a
            if 1 <= b <= N and b != x and b != a and bit_set(mask, b):
                return False
            m &= m - 1
            
        # Case 2: a is middle. x, a, b => b = 2a - x
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length()
            b = 2 * a - x
            if 1 <= b <= N and b != x and b != a and bit_set(mask, b):
                return False
            m &= m - 1
        return True

    def get_weight(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length()
            total += recip_float[x]
            m &= m - 1
        return total

    # 1. Initialization: Base-3 Cantor-like set
    # Numbers with no 1 in base 3
    base3_mask = 0
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 1:
                valid = False
                break
            n //= 3
        if valid:
            base3_mask = set_bit(base3_mask, i)
            
    # Greedy add to base3_mask
    mask = base3_mask
    # Sort by weight descending (index ascending)
    for x in range(1, N + 1):
        if not bit_set(mask, x) and is_valid_addition(mask, x):
            mask = set_bit(mask, x)
            
    best_mask = mask
    best_weight = get_weight(mask)

    # 2. Local Search
    current_mask = best_mask
    current_weight = best_weight
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < DEADLINE:
        if time.time() > DEADLINE:
            break
            
        improved = False
        
        # A. Additions
        for x in range(1, N + 1):
            if not bit_set(current_mask, x) and is_valid_addition(current_mask, x):
                current_mask = set_bit(current_mask, x)
                current_weight += recip_float[x]
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_mask = current_mask
                improved = True
                break # Restart loop to check for more additions
        
        if improved:
            no_improve = 0
            continue
            
        no_improve += 1
        
        # B. 1-for-1 Swaps
        elements = get_elements(current_mask)
        found_swap = False
        for u in elements:
            # Try replacing u with v < u (higher weight)
            for v in range(1, u):
                if bit_set(current_mask, v):
                    continue
                temp_mask = clear_bit(current_mask, u)
                if is_valid_addition(temp_mask, v):
                    new_w = current_weight - recip_float[u] + recip_float[v]
                    if new_w > current_weight + 1e-12:
                        current_mask = temp_mask | (1 << (v - 1))
                        current_weight = new_w
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_mask = current_mask
                        found_swap = True
                        break
            if found_swap:
                break
        
        if found_swap:
            no_improve = 0
            continue
            
        # C. 2-for-1 Swaps (Remove 2, Add 1)
        # Only if time permits
        if time.time() < DEADLINE - 5:
            elements = get_elements(current_mask)
            found_2_1 = False
            # Limit pairs to check
            max_pairs = 100
            pairs_tried = 0
            for i in range(len(elements)):
                if pairs_tried >= max_pairs: break
                u1 = elements[i]
                for j in range(i + 1, len(elements)):
                    if pairs_tried >= max_pairs: break
                    u2 = elements[j]
                    pairs_tried += 1
                    temp_mask = clear_bit(clear_bit(current_mask, u1), u2)
                    weight_loss = recip_float[u1] + recip_float[u2]
                    # Find best v to add
                    best_v = -1
                    best_v_w = 0.0
                    # Only check small v for speed
                    for v in range(1, min(N, 100) + 1):
                        if bit_set(temp_mask, v):
                            continue
                        if recip_float[v] <= weight_loss:
                            continue
                        if is_valid_addition(temp_mask, v):
                            if recip_float[v] > best_v_w:
                                best_v_w = recip_float[v]
                                best_v = v
                    if best_v != -1:
                        new_mask = temp_mask | (1 << (best_v - 1))
                        new_w = current_weight - weight_loss + best_v_w
                        if new_w > current_weight + 1e-12:
                            current_mask = new_mask
                            current_weight = new_w
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_mask = current_mask
                            found_2_1 = True
                            break
                if found_2_1:
                    break
            if found_2_1:
                no_improve = 0
                continue

        # D. 1-for-2 Swaps (Remove 1, Add 2)
        if time.time() < DEADLINE - 5:
            elements = get_elements(current_mask)
            found_1_2 = False
            # Limit u to check
            for u in elements[:50]: # Only try removing from first 50 elements (smallest indices)
                temp_mask = clear_bit(current_mask, u)
                weight_loss = recip_float[u]
                # Find best pair v1, v2
                # Candidates for v are small indices not in S
                candidates = [v for v in range(1, 50) if not bit_set(current_mask, v)]
                for i in range(len(candidates)):
                    v1 = candidates[i]
                    if not is_valid_addition(temp_mask, v1):
                        continue
                    temp_mask1 = set_bit(temp_mask, v1)
                    for j in range(i + 1, len(candidates)):
                        v2 = candidates[j]
                        if not is_valid_addition(temp_mask1, v2):
                            continue
                        new_w = current_weight - weight_loss + recip_float[v1] + recip_float[v2]
                        if new_w > current_weight + 1e-12:
                            new_mask = set_bit(temp_mask1, v2)
                            current_mask = new_mask
                            current_weight = new_w
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_mask = current_mask
                            found_1_2 = True
                            break
                    if found_1_2:
                        break
                if found_1_2:
                    break
            if found_1_2:
                no_improve = 0
                continue

        # E. Random Restart / Perturbation
        if no_improve > max_no_improve:
            # Perturb current solution
            if time.time() > DEADLINE - 10:
                break
                
            # Remove 2 random elements
            elems = get_elements(current_mask)
            if len(elems) > 2:
                idx1 = random.randint(0, len(elems) - 1)
                idx2 = random.randint(0, len(elems) - 1)
                while idx2 == idx1:
                    idx2 = random.randint(0, len(elems) - 1)
                current_mask = clear_bit(current_mask, elems[idx1])
                current_mask = clear_bit(current_mask, elems[idx2])
                current_weight = get_weight(current_mask)
                
            # Greedy re-fill
            for x in range(1, N + 1):
                if not bit_set(current_mask, x) and is_valid_addition(current_mask, x):
                    current_mask = set_bit(current_mask, x)
                    current_weight += recip_float[x]
                    
            if current_weight > best_weight:
                best_weight = current_weight
                best_mask = current_mask
                no_improve = 0
            else:
                # If no improvement after restart, break to save time for other things?
                # Or just keep going with a higher no_improve count?
                # Let's break if we've been stuck for a long time.
                if no_improve > max_no_improve * 2:
                    break
                no_improve += 1

    # Final Polish
    if time.time() < DEADLINE - 1:
        # One last pass of additions
        for x in range(1, N + 1):
            if not bit_set(best_mask, x) and is_valid_addition(best_mask, x):
                best_mask = set_bit(best_mask, x)
                best_weight += recip_float[x]

    result = get_elements(best_mask)
    result.sort()
    return result
