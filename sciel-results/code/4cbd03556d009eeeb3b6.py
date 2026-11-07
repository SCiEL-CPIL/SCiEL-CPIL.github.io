import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    max_val = n
    
    # Precompute: for each x, bitmask of y such that x+y <= max_val
    # sum_mask[x] has bit (y-1) set if x+y <= max_val
    sum_mask = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        m = 0
        for y in range(1, max_val + 1):
            if x + y <= max_val:
                m |= (1 << (y - 1))
        sum_mask[x] = m
    
    # Precompute shift_mask[x] = bitmask where bit (b-1) is set if (x+b) is a valid value in 1..max_val
    # i.e., shift_mask[x] has bit (b-1) set iff x+b <= max_val and x+b >= 1
    # This is actually the same as sum_mask[x] for the "b" side
    # But we need: for a given mask, shifted = {b : x+b in mask}
    # We can precompute for each x: shift_mask[x] where bit (b-1) set means b is a candidate
    # Then for a specific mask, the actual A = mask & sum_mask[x] & (mask shifted right by x)
    # Let's precompute the "shift" operation differently:
    # For each x, precompute a list of (from_bit, to_bit) pairs where from_bit = x+b-1, to_bit = b-1
    # Actually, let's just precompute shift_down[x] as a function using bit manipulation
    # shift_down(x, mask) = mask >> x (shift right by x bits)
    # If mask has bit (x+b-1) set, then mask >> x has bit (b-1) set.
    # So A_bits = mask & sum_mask[x] & (mask >> x)
    # This is O(1)!
    
    # Precompute pairs_sum[s] = list of (a,b) with a<b, a+b=s, s<=max_val
    pairs_sum = [[] for _ in range(max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= max_val:
                pairs_sum[s].append((a, b))
    
    def is_valid_addition(x, mask):
        """Check if adding x to valid set 'mask' keeps it valid."""
        if mask & (1 << (x - 1)):
            return True
        
        # Case 1: x is a member.
        # A = {b in S | x+b in S and x+b <= max_val}
        # Need b,c in A, b!=c, with b+c in S.
        
        # A = mask & sum_mask[x] & (mask >> x)
        # mask >> x: if mask has bit (x+b-1) set, result has bit (b-1) set
        A_bits = mask & sum_mask[x] & (mask >> x)
        
        # Check if there exist b,c in A_bits, b!=c, with b+c in S
        m_a = A_bits
        while m_a:
            lsb_b = m_a & -m_a
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m_a ^= lsb_b
            
            # candidates_c = A_bits & sum_mask[b_val] & ~(1 << b_idx)
            candidates_c = A_bits & sum_mask[b_val] & ~(1 << b_idx)
            while candidates_c:
                lsb_c = candidates_c & -candidates_c
                c_val = lsb_c.bit_length()
                sum_bc = b_val + c_val
                if sum_bc <= max_val and (mask & (1 << (sum_bc - 1))):
                    return False
                candidates_c ^= lsb_c
        
        # Case 2: x is a sum. a+b=x, a,b in S, a!=b.
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            
            candidates = mask & sum_mask[a] & sum_mask[b] & ~(1 << (a - 1)) & ~(1 << (b - 1))
            if candidates:
                return False
        
        return True
    
    # Initial solution: upper half
    best_mask = 0
    for i in range(n // 2, n):
        best_mask |= (1 << i)
    best_size = n - n // 2
    
    current_mask = best_mask
    current_size = best_size
    
    # Local search
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        # Try to add a random unused element
        unused = [i for i in range(1, n + 1) if not (current_mask & (1 << (i - 1)))]
        
        if unused:
            random.shuffle(unused)
            added = False
            for x in unused[:50]:  # Try more candidates
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    added = True
                    break
            if added:
                continue
        
        # Try 1-out, 2-in swap
        elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
        if elements and unused:
            x_out = random.choice(elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            valid_x1 = []
            for x in unused:
                if is_valid_addition(x, mask_trial):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                random.shuffle(valid_x1)
                found = False
                for i in range(min(len(valid_x1), 40)):
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, min(len(valid_x1), 40)):
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue
        
        # No add or swap worked; perturb
        no_improve_count += 1
        if elements:
            num_remove = random.randint(1, 3)
            for _ in range(num_remove):
                if elements:
                    x_out = random.choice(elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    elements.remove(x_out)
        
        # If stuck for too long, restart from best
        if no_improve_count > max_no_improve:
            current_mask = best_mask
            current_size = best_size
            no_improve_count = 0
    
    # Final polish: try to add any remaining valid elements
    polish_mask = best_mask
    polish_size = best_size
    for x in range(1, n + 1):
        if not (polish_mask & (1 << (x - 1))):
            if is_valid_addition(x, polish_mask):
                polish_mask |= (1 << (x - 1))
                polish_size += 1
    
    # Convert best_mask to list
    result = []
    m = polish_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
    
    return sorted(result)
