import time
import random

DEADLINE_SECONDS = 85.0

def _is_valid(elements):
    if not elements:
        return True
    
    bits = 1
    for e in elements:
        bits |= (bits << e)
        
    sums = set()
    temp_bits = bits
    while temp_bits:
        lsb = temp_bits & -temp_bits
        idx = lsb.bit_length() - 1
        if idx > 0:
            sums.add(idx)
        temp_bits ^= lsb
        
    if len(sums) <= 1:
        return True
        
    sum_list = sorted(sums)
    sum_set = sums
    max_sum = sum_list[-1]
    
    for a in sum_list:
        multiple = a * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += a
    return True

def _greedy_descending(deadline):
    current = []
    for c in range(256, 0, -1):
        if time.time() > deadline:
            break
        candidate = current + [c]
        if _is_valid(candidate):
            current.append(c)
    return current

def _greedy_randomized(deadline, seed):
    random.seed(seed)
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    current = []
    for c in candidates:
        if time.time() > deadline:
            break
        candidate = current + [c]
        if _is_valid(candidate):
            current.append(c)
    return current

def _local_search(initial, deadline, seed):
    random.seed(seed)
    current = list(initial)
    current_set = set(current)
    best = list(current)
    
    all_candidates = [x for x in range(1, 257) if x not in current_set]
    
    while time.time() < deadline:
        # Try to add
        if all_candidates:
            idx = random.randint(0, len(all_candidates) - 1)
            c = all_candidates[idx]
            new_set = current + [c]
            if _is_valid(new_set):
                current = new_set
                current_set.add(c)
                all_candidates.remove(c)
                if len(current) > len(best):
                    best = list(current)
                    
        # Try to swap (remove 1, add 1) to potentially unlock better additions?
        # Or remove 1, add 2 to increase size.
        if len(current) > 1:
            rem_idx = random.randint(0, len(current) - 1)
            removed = current.pop(rem_idx)
            current_set.discard(removed)
            
            # Try to add 2
            if len(all_candidates) >= 2:
                c1 = all_candidates[random.randint(0, len(all_candidates)-1)]
                # Pick c2 distinct from c1
                while True:
                    c2 = all_candidates[random.randint(0, len(all_candidates)-1)]
                    if c2 != c1:
                        break
                
                candidate = current + [c1, c2]
                if _is_valid(candidate):
                    current = candidate
                    current_set.add(c1)
                    current_set.add(c2)
                    all_candidates = [x for x in all_candidates if x != c1 and x != c2]
                    if len(current) > len(best):
                        best = list(current)
                else:
                    current.append(removed)
                    current_set.add(removed)
            else:
                current.append(removed)
                current_set.add(removed)
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_result = []
    best_size = 0
    
    # 1. Deterministic Greedy Descending
    res = _greedy_descending(deadline)
    if len(res) > best_size:
        best_size = len(res)
        best_result = res
        
    # 2. Randomized Greedy + Local Search
    # We have ~85s.
    # Each LS run takes ~0.1s.
    # So we can do ~800 runs.
    
    for i in range(1000):
        if time.time() > deadline:
            break
            
        seed = i
        res_greedy = _greedy_randomized(deadline, seed)
        
        if len(res_greedy) > best_size:
            best_size = len(res_greedy)
            best_result = res_greedy
            
        # Local search
        # Allocate small time slice
        ls_deadline = min(deadline, time.time() + 0.05) # 50ms per LS
        if time.time() < ls_deadline:
            res_ls = _local_search(res_greedy, ls_deadline, seed)
            if len(res_ls) > best_size:
                best_size = len(res_ls)
                best_result = res_ls

    # Fallback
    if not best_result:
        best_result = [256]
        
    # Final verification
    if not _is_valid(best_result):
        best_result = [256]

    return [int(x) for x in best_result]
