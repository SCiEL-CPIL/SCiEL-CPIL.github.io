import time
import random

N = 17
MIN_DIST = 6
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    random.seed(42)
    
    codewords = [0]
    cw_set = {0}
    
    # Greedy random addition
    while time_left() > 10:
        w = random.getrandbits(N)
        if w in cw_set:
            continue
        valid = True
        for c in codewords:
            if (w ^ c).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            codewords.append(w)
            cw_set.add(w)
    
    # Local search: try to add more
    while time_left() > 2:
        improved = False
        for _ in range(500):
            if time_left() < 2:
                break
            # Pick a random existing codeword and mutate it
            base = random.choice(codewords)
            # Flip 1-3 bits
            n_flips = random.randint(1, 3)
            positions = random.sample(range(N), n_flips)
            w = base
            for p in positions:
                w ^= (1 << p)
            
            if w in cw_set:
                continue
            
            valid = True
            for c in codewords:
                if (w ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                codewords.append(w)
                cw_set.add(w)
                improved = True
        
        if not improved and time_left() < 5:
            break
    
    return codewords
