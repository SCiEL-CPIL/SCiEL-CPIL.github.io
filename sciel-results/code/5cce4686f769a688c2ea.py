import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    max_val = n
    
    # Initial valid set: Upper half (65..128)
    best_mask = 0
    for i in range(n//2, n):
        best_mask |= (1 << i)
    best_size = n - n//2
    
    def get_elems(mask):
        elems = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elems.append(idx + 1)
            m ^= lsb
        return elems

    def can_add(x, mask):
        if x > max_val: return False
        x_bit = 1 << (x-1)
        if mask & x_bit: return True
        
        new_mask = mask | x_bit
        
        elems = get_elems(mask)
        n_elems = len(elems)
        
        for i in range(n_elems):
            u = elems[i]
            for j in range(i+1, n_elems):
                v = elems[j]
                su = u + v
                sxu = x + u
                sxv = x + v
                
                if su > max_val or sxu > max_val or sxv > max_val:
                    continue
                
                if (new_mask & (1 << (su-1))) and \
                   (new_mask & (1 << (sxu-1))) and \
                   (new_mask & (1 << (sxv-1))):
                    return False
        return True

    def greedy_add(mask, size, count):
        """Try to add 'count' elements to mask greedily."""
        for _ in range(count):
            if time.time() > deadline: break
            candidates = [i for i in range(1, max_val + 1) if not (mask & (1 << (i-1)))]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(x, mask):
                    mask |= (1 << (x-1))
                    size += 1
                    added = True
                    break
            if not added:
                break
        return mask, size

    # Main search loop
    current_mask = best_mask
    current_size = best_size
    
    for _ in range(5000):
        if time.time() > deadline: break
        
        old_size = current_size
        
        # 1. Try to add elements
        current_mask, current_size = greedy_add(current_mask, current_size, 100)
        
        # 2. If stuck, try 1 out, 2 in
        if current_size == old_size and time.time() < deadline:
            elems = get_elems(current_mask)
            random.shuffle(elems)
            improved = False
            for u_out in elems:
                if time.time() > deadline: break
                mask_trial = current_mask & ~(1 << (u_out-1))
                size_trial = current_size - 1
                
                # Try to add 2
                m_new, s_new = greedy_add(mask_trial, size_trial, 2)
                
                if s_new > current_size:
                    current_mask = m_new
                    current_size = s_new
                    improved = True
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        if not improved and current_size == old_size:
            # Perturb: remove random element
            if current_size > 0:
                elems = get_elems(current_mask)
                if elems:
                    rem = random.choice(elems)
                    current_mask &= ~(1 << (rem-1))
                    current_size -= 1

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)
