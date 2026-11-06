import time
import random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    
    def score_bitmask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        # d can be 1 to 255 (since a >= 1, a+2d <= 512 => 2d <= 511 => d <= 255)
        for d in range(1, 256):
            # Check if there exists a such that bits a, a+d, a+2d are all set
            # S & (S >> d) & (S >> 2*d) != 0
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count
    
    # Initial set: try a few structured starts
    def make_initial():
        # Start with a random-ish but structured set
        # Use a mixture: some small numbers, some spread out
        s = set()
        # Add a base AP
        for i in range(1, 17):
            s.add(i)
        # Add more spread elements
        for i in range(16, 33):
            s.add(16 + i * 3)  # 19, 22, 25, ..., 113
        # Fill remaining with random
        while len(s) < 32:
            s.add(random.randint(1, 512))
        return list(s)
    
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m
    
    # Try multiple restarts
    best_score = -1
    best_set = None
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Create initial set
        if num_restarts == 1:
            # First restart: structured
            s = set(range(1, 33))  # 1..32
        elif num_restarts == 2:
            # Second: spread out
            s = set()
            for i in range(32):
                s.add(1 + i * 16)  # 1, 17, 33, ..., 505
        elif num_restarts == 3:
            # Third: mixed
            s = set()
            for i in range(16):
                s.add(1 + i * 2)  # odd numbers 1..31
            for i in range(16):
                s.add(32 + i * 28)  # 32, 60, 88, ..., 472
        else:
            s = make_initial()
        
        s = list(s)
        mask = set_to_mask(s)
        cur_score = score_bitmask(mask)
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(s)
        
        # Local search
        temp = 10.0
        cooling = 0.9999
        iters = 0
        max_iters = 50000
        
        while iters < max_iters and time.time() < deadline - 5.0:
            iters += 1
            # Pick a random element to replace
            idx = random.randint(0, 31)
            old_val = s[idx]
            # Pick a new value not in the set
            new_val = random.randint(1, 512)
            if new_val == old_val:
                continue
            if new_val in s:
                continue
            
            # Try the swap
            s[idx] = new_val
            new_mask = set_to_mask(s)
            new_score = score_bitmask(new_mask)
            
            if new_score > cur_score:
                cur_score = new_score
                mask = new_mask
                if cur_score > best_score:
                    best_score = cur_score
                    best_set = set(s)
            else:
                # Accept with probability exp(delta/temp)
                delta = new_score - cur_score
                if random.random() < pow(2.718281828, delta / temp):
                    cur_score = new_score
                    mask = new_mask
                else:
                    s[idx] = old_val
            
            temp *= cooling
        
        # Also try greedy improvement passes
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(32):
                old_val = s[i]
                best_new = old_val
                best_s = cur_score
                # Try all possible replacements (sample)
                for _ in range(20):
                    nv = random.randint(1, 512)
                    if nv == old_val or nv in s:
                        continue
                    s[i] = nv
                    ms = set_to_mask(s)
                    sc = score_bitmask(ms)
                    if sc > best_s:
                        best_s = sc
                        best_new = nv
                    s[i] = old_val
                if best_new != old_val:
                    s[i] = best_new
                    cur_score = best_s
                    mask = set_to_mask(s)
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = set(s)
    
    return sorted(list(best_set))
