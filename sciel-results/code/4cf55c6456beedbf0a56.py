import time
import random
import sys

N = 9

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    best_path = []
    best_len = 0
    
    def popcount(x):
        return x.bit_count()

    def smart_greedy(randomize_bits=True):
        path = [0]
        used = {0}
        # Precompute bit masks
        bit_masks = [1 << b for b in range(N)]
        
        while True:
            if time.time() > deadline - 5.0:
                break
            cur = path[-1]
            candidates = []
            bits = list(range(N))
            if randomize_bits:
                random.shuffle(bits)
                
            for b in bits:
                v = cur ^ bit_masks[b]
                if v in used:
                    continue
                
                # Check if v can be appended
                # dist(v, p) >= 2 for all p in path[:-1]
                valid = True
                # path[:-1] are all vertices except the last one (cur)
                for i in range(len(path) - 1):
                    p = path[i]
                    if popcount(v ^ p) == 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate lookahead score
                # Count how many neighbors of v are valid extensions
                score = 0
                for b2 in range(N):
                    nxt = v ^ bit_masks[b2]
                    if nxt in used:
                        continue
                    # Check if nxt can be appended to path + [v]
                    # This requires dist(nxt, p) >= 2 for all p in path[:-1]
                    ok = True
                    for i in range(len(path) - 1):
                        p = path[i]
                        if popcount(nxt ^ p) == 1:
                            ok = False
                            break
                    if ok:
                        score += 1
                
                candidates.append((score, v))
            
            if not candidates:
                break
            
            # Pick best candidate
            candidates.sort(key=lambda x: -x[0])
            max_score = candidates[0][0]
            best_cands = [v for s, v in candidates if s == max_score]
            nxt = random.choice(best_cands)
            
            path.append(nxt)
            used.add(nxt)
            
        return path

    # Initial run
    p = smart_greedy(randomize_bits=False)
    best_path = p
    best_len = len(p)
    
    # Iterative improvement
    while time.time() < deadline - 2.0:
        p = smart_greedy(randomize_bits=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
    return best_path
