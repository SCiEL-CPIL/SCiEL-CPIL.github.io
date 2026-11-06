import random
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds limit to be safe
    
    best_set = []
    best_len = 0
    
    OFFSET = 25000
    
    # Run multiple random permutations
    # 100 restarts might be too many if each is slow, but 50 should be fine
    for _ in range(100):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        global_bits = [0] * 50
        global_bits[0] = 1 << OFFSET
        
        bits = {}
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
                
            # Check 1: Is x mean of subset of S?
            valid = True
            for k in range(2, len(S) + 1):
                target_sum = k * x
                idx = target_sum + OFFSET
                if 0 <= idx < 50000:
                    if (global_bits[k] >> idx) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Check 2: For each a in S, is a mean of subset of S U {x} including x?
            for a in S:
                target_sum = a - x
                idx = target_sum + OFFSET
                if 0 <= idx < 50000:
                    ba = bits[a]
                    # V_a size is len(S)-1
                    # k can be 1 to len(S)-1
                    for k in range(1, len(S)):
                        if (ba[k] >> idx) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Add x to S
            S.append(x)
            
            # Update global_bits
            new_global_bits = [0] * 50
            for k in range(len(S)):
                new_global_bits[k] = global_bits[k]
                if k > 0:
                    new_global_bits[k] |= (global_bits[k-1] << x)
            global_bits = new_global_bits
            
            # Update bits for existing a in S
            for a in S:
                if a == x:
                    continue
                old_ba = bits[a]
                new_ba = [0] * 50
                # old V_a size was len(S)-2
                max_k_old = len(S) - 2
                for k in range(max_k_old + 1):
                    new_ba[k] = old_ba[k]
                for k in range(1, max_k_old + 2):
                    if k-1 <= max_k_old:
                        new_ba[k] |= (old_ba[k-1] << (x - a))
                bits[a] = new_ba
                
            # Initialize bits for x
            vx = [s - x for s in S if s != x]
            bx = [0] * 50
            bx[0] = 1 << OFFSET
            current_size = 0
            for v in vx:
                for k in range(current_size + 1, 0, -1):
                    bx[k] |= (bx[k-1] << v)
                current_size += 1
            bits[x] = bx
            
            if len(S) > best_len:
                best_len = len(S)
                best_set = S[:]
                
    return best_set
