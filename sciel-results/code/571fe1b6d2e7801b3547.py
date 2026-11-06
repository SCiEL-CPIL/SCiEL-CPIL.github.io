import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 17
    min_dist = 6
    max_size = 340
    
    # Precompute popcounts
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Construct the [17,8,6] quadratic residue code
    gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
    
    def encode(msg):
        cw = 0
        for bit_pos in range(9):
            if (msg >> bit_pos) & 1:
                cw ^= (gen_poly << bit_pos)
        return cw
    
    qr_codewords = []
    seen = set()
    for msg in range(256):
        cw = encode(msg)
        if cw not in seen:
            qr_codewords.append(cw)
            seen.add(cw)
    
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        for i in range(len(qr_codewords)):
            for j in range(i + 1, len(qr_codewords)):
                if popcount[qr_codewords[i] ^ qr_codewords[j]] < min_dist:
                    valid_qr = False
                    break
            if not valid_qr:
                break
    
    if valid_qr:
        best_codewords = qr_codewords
        best_size = 256
    else:
        best_codewords = [0]
        best_size = 1
        for candidate in range(1, 1 << n):
            if time.time() > deadline - 30.0:
                break
            valid = True
            for cw in best_codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                best_codewords.append(candidate)
                best_size += 1

    # Local Search to improve upon the initial code (likely 256 words)
    # We try to add words, and if that fails, we try swaps (remove 1, add 2)
    
    def try_add(codewords, candidate):
        for cw in codewords:
            if popcount[candidate ^ cw] < min_dist:
                return False
        return True

    best_codewords = list(best_codewords)
    best_size = len(best_codewords)
    
    # Candidate pool
    all_candidates = list(range(1 << n))
    
    # Run local search
    while time.time() < deadline - 2.0:
        current = list(best_codewords)
        cw_set = set(current)
        improved = False
        
        # 1. Try to add
        random.shuffle(all_candidates)
        for cand in all_candidates:
            if cand in cw_set:
                continue
            if try_add(current, cand):
                current.append(cand)
                cw_set.add(cand)
                improved = True
                break # Restart scan if we found one
        
        # 2. Try to swap (remove 1, add 2)
        if not improved:
            # Pick a random codeword to remove
            if len(current) > 0:
                idx = random.randint(0, len(current)-1)
                removed = current[idx]
                current.pop(idx)
                cw_set.remove(removed)
                
                # Try to add 2
                added = []
                # Shuffle candidates again
                random.shuffle(all_candidates)
                for cand in all_candidates:
                    if cand in cw_set or cand == removed:
                        continue
                    if try_add(current, cand):
                        current.append(cand)
                        cw_set.add(cand)
                        added.append(cand)
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    improved = True
                else:
                    # Revert
                    for a in added:
                        current.remove(a)
                        cw_set.remove(a)
                    current.append(removed)
                    cw_set.add(removed)
        
        if improved:
            if len(current) > best_size:
                best_size = len(current)
                best_codewords = current
                if best_size >= max_size:
                    break
        else:
            # If no improvement in this round, we might be stuck.
            # We break to avoid wasting time on a local minimum.
            break

    return best_codewords
