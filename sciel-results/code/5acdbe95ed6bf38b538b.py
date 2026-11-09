import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    # Precompute 3-APs and 4-APs as bitmasks
    # Also precompute conflict and gain masks for each element
    
    all_3ap_masks = []
    all_4ap_masks = []
    
    # Generate 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            all_3ap_masks.append(mask)
            
    # Generate 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            all_4ap_masks.append(mask)
            
    # Precompute conflicts and gains for each x
    # x is 0-indexed in bitmask, 1-indexed in logic
    # Let's use 0-indexed for bitmask shifts: 1<<x for element x+1
    # Wait, standard is 1<<(val-1). Let's stick to that.
    
    conflict_masks = [[] for _ in range(N + 1)]
    gain_masks = [[] for _ in range(N + 1)]
    
    for mask in all_4ap_masks:
        # Extract the 4 elements
        elems = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            elems.append(idx + 1) # 1-based index
            temp ^= lsb
        # For each element x in this 4-AP, the other 3 form a conflict mask
        for i in range(4):
            x = elems[i]
            others = elems[:i] + elems[i+1:]
            c_mask = 0
            for o in others:
                c_mask |= (1 << (o - 1))
            conflict_masks[x].append(c_mask)
            
    for mask in all_3ap_masks:
        # Extract the 3 elements
        elems = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            elems.append(idx + 1) # 1-based index
            temp ^= lsb
        # For each element x in this 3-AP, the other 2 form a gain mask
        for i in range(3):
            x = elems[i]
            others = elems[:i] + elems[i+1:]
            g_mask = 0
            for o in others:
                g_mask |= (1 << (o - 1))
            gain_masks[x].append(g_mask)
            
    def is_valid_add(x, A_mask):
        for cm in conflict_masks[x]:
            if (A_mask & cm) == cm:
                return False
        return True
        
    def count_gain(x, A_mask):
        gain = 0
        for gm in gain_masks[x]:
            if (A_mask & gm) == gm:
                gain += 1
        return gain
        
    def count_total_3ap(A_mask):
        count = 0
        for mask in all_3ap_masks:
            if (A_mask & mask) == mask:
                count += 1
        return count
        
    best_A_mask = 0
    best_score = 0
    
    # Greedy with Randomization
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
            
        A_mask = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        while remaining:
            best_x = -1
            best_gain = -1
            # Optimization: We can skip validity check if gain is 0? No, we need valid.
            # We can pre-filter valid candidates? 
            # For now, just iterate.
            for x in remaining:
                if is_valid_add(x, A_mask):
                    gain = count_gain(x, A_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        # If gain is high, maybe break? No, we want max.
                        
            if best_x == -1:
                break
                
            A_mask |= (1 << (best_x - 1))
            remaining.remove(best_x)
            
        current_score = count_total_3ap(A_mask)
        if current_score > best_score:
            best_score = current_score
            best_A_mask = A_mask
            
    # Local Search
    if time.time() < deadline - 2.0:
        A_mask = best_A_mask
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try 1-swap
            A_elems = [i+1 for i in range(N) if (A_mask >> i) & 1]
            not_A_elems = [i+1 for i in range(N) if not ((A_mask >> i) & 1)]
            
            # To speed up, we can limit the number of swaps checked per iteration
            # or just check all. N=256, |A|~100, |not A|~150. 15000 swaps.
            # Each swap involves validity check (O(conflicts)) and score update.
            # Validity check is fast with bitmasks.
            # Score update: remove x, add y.
            # New mask = A_mask ^ (1<<(x-1)) ^ (1<<(y-1))
            # Check if new mask is valid.
            # Calculate new score.
            
            random.shuffle(A_elems)
            random.shuffle(not_A_elems)
            
            for x in A_elems:
                if time.time() > deadline - 2.0:
                    break
                for y in not_A_elems:
                    new_mask = A_mask ^ (1 << (x - 1)) ^ (1 << (y - 1))
                    
                    # Check validity of new_mask
                    valid = True
                    # We only need to check 4-APs involving x or y?
                    # Actually, removing x can only make things more valid.
                    # Adding y can only make things less valid.
                    # So we just need to check if adding y to (A - {x}) creates a 4-AP.
                    # This is equivalent to checking if y is valid to add to (A - {x}).
                    base_mask = A_mask ^ (1 << (x - 1))
                    if is_valid_add(y, base_mask):
                        new_score = count_total_3ap(new_mask)
                        if new_score > current_score:
                            current_score = new_score
                            A_mask = new_mask
                            best_score = current_score
                            best_A_mask = A_mask
                            improved = True
                            # Restart loops with new A
                            A_elems = [i+1 for i in range(N) if (A_mask >> i) & 1]
                            not_A_elems = [i+1 for i in range(N) if not ((A_mask >> i) & 1)]
                            break
                if improved:
                    break
            if improved:
                continue
                
    # Convert best_A_mask to list
    result = []
    for i in range(N):
        if (best_A_mask >> i) & 1:
            result.append(i + 1)
            
    return result
