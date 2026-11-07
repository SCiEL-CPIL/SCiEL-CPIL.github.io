import time
import random

def construct():
    deadline = time.time() + 55
    best_set = []
    
    def check_validity(bitsets, n, x):
        new_bits = [0] * (n + 2)
        for k in range(n + 1):
            new_bits[k] = bitsets[k]
            if k > 0:
                new_bits[k] |= (bitsets[k-1] << x)
        
        for k in range(n + 1):
            for m in range(k + 1, n + 2):
                if new_bits[k] & new_bits[m]:
                    return False, None
        return True, new_bits

    def greedy(strategy="smallest"):
        S = []
        bitsets = [0] * 61
        bitsets[0] = 1
        current_size = 0
        used = set()
        
        while current_size < 60:
            valid_x = None
            candidates = list(range(1, 257))
            candidates = [x for x in candidates if x not in used]
            
            if strategy == "smallest":
                pass
            elif strategy == "largest":
                candidates.reverse()
            else:
                random.shuffle(candidates)
            
            for x in candidates:
                valid, new_bits = check_validity(bitsets, current_size, x)
                if valid:
                    valid_x = x
                    break
            
            if valid_x is not None:
                S.append(valid_x)
                used.add(valid_x)
                bitsets = new_bits
                current_size += 1
            else:
                break
        return S

    # Try deterministic smallest
    S = greedy("smallest")
    best_set = S
    
    # Try deterministic largest
    S = greedy("largest")
    if len(S) > len(best_set):
        best_set = S
        
    # Try randomized
    start_time = time.time()
    for i in range(1000):
        if time.time() - start_time > 50:
            break
        S = greedy("random")
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 60:
                break

    return [int(x) for x in best_set]
