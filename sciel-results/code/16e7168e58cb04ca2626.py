import time
import random
import math

def get_score(U):
    if not U:
        return 0.0
    U = set(U)
    if 0 not in U:
        return 0.0
    # Ensure non-negative
    if any(x < 0 for x in U):
        return 0.0
    
    sums = set()
    diffs = set()
    
    # Optimization: Use list for faster iteration if U is small
    U_list = list(U)
    n = len(U_list)
    
    for a in U_list:
        for b in U_list:
            sums.add(a + b)
            diffs.add(a - b)
            
    sum_size = len(sums)
    diff_size = len(diffs)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    max_u = max(U)
    denom = math.log(2 * max_u + 1)
    
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def mian_chowla(n):
    seq = [0, 1]
    for i in range(2, n):
        last = seq[-1]
        candidate = last + 1
        while True:
            ok = True
            # Check if candidate creates duplicate sums
            # Sums of existing pairs + candidate
            # We need to ensure all sums are distinct
            # For Mian-Chowla, we just need to check against existing sums
            existing_sums = set()
            for a in seq:
                for b in seq:
                    existing_sums.add(a + b)
            
            # Check sums involving candidate
            for a in seq:
                s = a + candidate
                if s in existing_sums:
                    ok = False
                    break
            if not ok:
                candidate += 1
                continue
            
            # Also need to ensure candidate + candidate doesn't clash? 
            # The standard definition ensures all sums x+y are distinct for x<=y.
            # The above check covers a+candidate for a in seq.
            # We also need candidate+candidate.
            if candidate + candidate in existing_sums:
                ok = False
            
            if ok:
                seq.append(candidate)
                break
            else:
                candidate += 1
    return seq

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Evaluate Mian-Chowla sets
    for n in range(3, 25):
        if time.time() > deadline:
            break
        seq = mian_chowla(n)
        s = get_score(seq)
        if s > best_score:
            best_score = s
            best_set = seq[:]
            
    # 2. Local Search for small N
    # We suspect N around 5-10 is optimal
    for target_n in range(3, 15):
        if time.time() > deadline:
            break
            
        # Initialize with Mian-Chowla
        current_set = mian_chowla(target_n)
        current_score = get_score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                original = current_set[i]
                # Try changing element i to original +/- delta
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    new_val = original + delta
                    if new_val < 0:
                        continue
                    if new_val in current_set:
                        continue
                    
                    new_set = current_set[:]
                    new_set[i] = new_val
                    # Sort to keep deterministic? Not strictly necessary for set logic
                    # But Mian-Chowla generation assumes sorted.
                    # For scoring, order doesn't matter.
                    
                    s = get_score(new_set)
                    if s > current_score:
                        current_set = new_set
                        current_score = s
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_set = current_set[:]
                        break # Restart loop for this index? Or continue?
                        # Breaking out of delta loop, continue to next i
                
        # Random Restart
        for _ in range(10):
            if time.time() > deadline:
                break
            # Generate random set of size target_n
            # Max value around 50
            rand_set = set([0])
            while len(rand_set) < target_n:
                rand_set.add(random.randint(1, 50))
            rand_list = list(rand_set)
            s = get_score(rand_list)
            if s > best_score:
                best_score = s
                best_set = rand_list[:]
                
            # Local search on random start
            current_set = rand_list
            current_score = s
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(len(current_set)):
                    original = current_set[i]
                    for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                        new_val = original + delta
                        if new_val < 0:
                            continue
                        if new_val in current_set:
                            continue
                        
                        new_set = current_set[:]
                        new_set[i] = new_val
                        s_new = get_score(new_set)
                        if s_new > current_score:
                            current_set = new_set
                            current_score = s_new
                            improved = True
                            if s_new > best_score:
                                best_score = s_new
                                best_set = current_set[:]
                            break

    return best_set
