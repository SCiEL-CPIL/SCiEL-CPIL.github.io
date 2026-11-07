import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    n = P['n']
    max_val = n

    def popcount(x):
        return bin(x).count('1')

    def is_valid_addition(x, mask):
        """
        Check if adding x to set represented by mask is valid.
        mask: bitset where bit i-1 is 1 if i is in S.
        """
        if mask & (1 << (x - 1)):
            return True

        # Get list of elements in S
        # Optimization: We only need to check pairs (u, v) in S such that u + v == x.
        # Then for each such pair, check if there exists w in S (w != u, w != v)
        # such that u + w in S and v + w in S.
        
        # Find pairs (u, v) in S with u < v and u + v == x
        valid_pairs = []
        # u must be at least 1, v must be at most x-1
        # Also u, v must be in S
        for u in range(1, x // 2 + 1):
            if mask & (1 << (u - 1)):
                v = x - u
                if v > u and (mask & (1 << (v - 1))):
                    valid_pairs.append((u, v))
        
        if not valid_pairs:
            return True

        # For each pair, check for w
        for u, v in valid_pairs:
            # We need w in S, w != u, w != v
            # such that (u + w) in S and (v + w) in S
            # Note: u+w and v+w must be <= max_val
            # w can range from 1 to max_val - max(u, v)
            
            # Iterate through potential w
            # To speed up, we can check bits directly
            # Let's just iterate w from 1 to max_val
            # But we can restrict w such that u+w <= max_val and v+w <= max_val
            max_w = max_val - max(u, v)
            
            # Precompute bit masks for u+w and v+w? No, just check.
            found_w = False
            for w in range(1, max_w + 1):
                if w == u or w == v:
                    continue
                if mask & (1 << (w - 1)):
                    # Check if u+w in S
                    sum_1 = u + w
                    sum_2 = v + w
                    if (mask & (1 << (sum_1 - 1))) and (mask & (1 << (sum_2 - 1))):
                        found_w = True
                        break
            
            if found_w:
                return False

        return True

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    # ---- Initial Greedy Construction ----
    # Start with high numbers
    best_mask = 0
    for v in range(max_val, 0, -1):
        if is_valid_addition(v, best_mask):
            best_mask |= (1 << (v - 1))
    
    best_size = popcount(best_mask)
    current_mask = best_mask
    current_size = best_size

    no_progress_count = 0
    max_no_progress = 100

    # Precompute list of all elements for faster iteration
    all_elements = list(range(1, max_val + 1))

    while time.time() < deadline - 2.0:
        # 1. Greedy Add - try all unused elements, pick first valid
        unused_elements = unused_of(current_mask)
        if unused_elements:
            added = False
            # Try in random order for diversity
            candidates = list(unused_elements)
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    no_progress_count = 0
                    break
            
            if added:
                continue

        # 2. Multi-swap strategies
        found_swap = False
        current_elements = elements_of(current_mask)
        
        # 2a. 1-out, 2-in Swap (Growth)
        if current_elements and len(unused_elements) >= 2:
            x_out = random.choice(current_elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            valid_adds = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_adds.append(x)
            
            if len(valid_adds) >= 2:
                random.shuffle(valid_adds)
                max_checks = 50
                checks = 0
                for i in range(len(valid_adds)):
                    if checks > max_checks: break
                    x1 = valid_adds[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_adds)):
                        if checks > max_checks: break
                        checks += 1
                        x2 = valid_adds[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            
                            found_swap = True
                            no_progress_count = 0
                            break
                    if found_swap:
                        break

        # 2b. 1-out, 1-in Swap (Escaping Local Optima)
        if not found_swap and current_elements and unused_elements:
            x_out = random.choice(current_elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            # Try to find any valid add
            random.shuffle(unused_elements)
            for x_in in unused_elements:
                if is_valid_addition(x_in, mask_trial):
                    current_mask = mask_trial | (1 << (x_in - 1))
                    # Size unchanged
                    found_swap = True
                    no_progress_count = 0
                    break

        # 2c. 2-out, 3-in Swap (Bigger Growth)
        if not found_swap and len(current_elements) >= 2 and len(unused_elements) >= 3:
            # Try removing 2 random elements and adding 3
            out_indices = random.sample(range(len(current_elements)), 2)
            x_out1 = current_elements[out_indices[0]]
            x_out2 = current_elements[out_indices[1]]
            
            mask_trial = current_mask & ~(1 << (x_out1 - 1)) & ~(1 << (x_out2 - 1))
            
            valid_adds = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_adds.append(x)
            
            if len(valid_adds) >= 3:
                random.shuffle(valid_adds)
                max_checks = 30
                checks = 0
                for i in range(len(valid_adds)):
                    if checks > max_checks: break
                    x1 = valid_adds[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_adds)):
                        if checks > max_checks: break
                        x2 = valid_adds[j]
                        mask_2 = mask_1 | (1 << (x2 - 1))
                        for k in range(j + 1, len(valid_adds)):
                            if checks > max_checks: break
                            checks += 1
                            x3 = valid_adds[k]
                            if is_valid_addition(x3, mask_2):
                                current_mask = mask_2 | (1 << (x3 - 1))
                                current_size += 1
                                if current_size > best_size:
                                    best_size = current_size
                                    best_mask = current_mask
                                
                                found_swap = True
                                no_progress_count = 0
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break

        if found_swap:
            continue

        # 3. Perturbation - remove 1-3 random elements
        if current_elements:
            num_remove = random.randint(1, 3)
            num_remove = min(num_remove, len(current_elements))
            for _ in range(num_remove):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)
            no_progress_count += 1

        if no_progress_count > max_no_progress:
            # Restart from best with some randomization
            current_mask = best_mask
            current_size = best_size
            current_elements = elements_of(current_mask)
            no_progress_count = 0
            
            # Try a few random removals to escape local optimum
            for _ in range(random.randint(1, 5)):
                if current_elements:
                    x_out = random.choice(current_elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    current_elements.remove(x_out)

    # Final polish - multiple passes
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_unused = set(unused_of(final_mask))
        
        for pass_num in range(5):
            if time.time() > deadline - 1.0:
                break
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                if pass_num % 2 == 0:
                    candidates = sorted(final_unused)
                else:
                    candidates = list(final_unused)
                    random.shuffle(candidates)
                for x in candidates:
                    if x not in final_unused:
                        continue
                    if is_valid_addition(x, final_mask):
                        final_mask |= (1 << (x - 1))
                        final_unused.remove(x)
                        changed = True
        
        if popcount(final_mask) > best_size:
            best_mask = final_mask
            best_size = popcount(final_mask)

    # Convert best_mask to list of ints
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)
