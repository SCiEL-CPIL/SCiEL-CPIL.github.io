import time

def construct():
    start_time = time.time()
    limit = 85  # 85 seconds out of 90
    
    # Greedy initialization
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_len = len(values)
    best_seq = values[:]
    
    # Convert to bitset for speed in DFS
    max_sum_bound = 256 * 30 # Safe upper bound for sum
    sums_bitset = 0
    for s in sums:
        sums_bitset |= (1 << s)
        
    def dfs(seq, sums_bs, last_val):
        nonlocal best_len, best_seq, start_time, limit
        
        if time.time() - start_time > limit:
            raise TimeoutError()
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: if we can't possibly beat best_len, return
        # Max possible length is limited by available numbers and sum distinctness
        # A simple bound: remaining numbers = 256 - last_val
        # If len(seq) + (256 - last_val) <= best_len: return
        # But we also need distinct sums.
        
        # Try to add next element
        # To find longer sequences, we should try smaller values first? 
        # Or larger? 
        # Greedy used smallest. 
        # If greedy got stuck, maybe larger values allow more room?
        # Let's try iterating from last_val + 1 to 256.
        
        for x in range(last_val + 1, 257):
            # Calculate new sums
            new_sums = [x]
            total = x
            for a in reversed(seq):
                total += a
                new_sums.append(total)
                
            # Check collisions with existing sums
            mask = 0
            for s in new_sums:
                if s > max_sum_bound: 
                    # If sum is too big, it's distinct from small ones, 
                    # but we need to handle bitset size.
                    # 256 * 20 = 5120. Python int handles this fine.
                    pass
                mask |= (1 << s)
                
            # Check collision with existing sums_bitset
            if sums_bs & mask:
                continue
                
            # Check collision among new_sums
            # If len(set(new_sums)) != len(new_sums), continue
            # This is O(k^2) or O(k log k). k is small.
            if len(set(new_sums)) != len(new_sums):
                continue
                
            # Valid move
            dfs(seq + [x], sums_bs | mask, x)
            
            # If we found a longer sequence, we might want to stop early?
            # No, we want to maximize. 
            # But if we hit the time limit, we return best.
            
    try:
        dfs([], 0, 0)
    except TimeoutError:
        pass
        
    return best_seq

# EVOLVE-BLOCK-END
