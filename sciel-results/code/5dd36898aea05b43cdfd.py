"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Bitset: x in 1..512 maps to bit (x-1), so 0-indexed
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def score_bitset(b):
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def make_mask(elem_list):
        m = 0
        for x in elem_list:
            m |= (1 << (x - 1))
        return m
    
    def mask_to_list(m):
        result = []
        i = 0
        while m:
            if m & 1:
                result.append(i + 1)
            m >>= 1
            i += 1
        return result
    
    # Greedy initialization: build a set element by element, picking
    # candidates that maximize score gain, with randomization among top-k
    def greedy_init(rng):
        mask = 0
        elems = []
        for _ in range(32):
            # Evaluate all candidates
            best_score = -1
            top_candidates = []
            for x in range(1, 513):
                if mask & (1 << (x - 1)):
                    continue
                new_mask = mask | (1 << (x - 1))
                sc = score_bitset(new_mask)
                if sc > best_score:
                    best_score = sc
                    top_candidates = [(sc, x)]
                elif sc == best_score and len(top_candidates) < 20:
                    top_candidates.append((sc, x))
            
            if not top_candidates:
                break
            
            # Pick randomly from top candidates
            sc, x = rng.choice(top_candidates[:min(5, len(top_candidates))])
            mask |= (1 << (x - 1))
            elems.append(x)
        
        return mask, elems
    
    best_mask = 0
    best_score = 0
    
    # Run multiple restarts: greedy init + SA polish, then random restarts
    num_restarts = 0
    
    while time.time() < deadline:
        # Choose initialization strategy
        if num_restarts == 0:
            # First restart: greedy initialization
            rng = random.Random(42)
            mask, elems = greedy_init(rng)
        else:
            # Subsequent restarts: random initialization
            rng = random.Random()
            elems = random.sample(range(1, 513), 32)
            mask = make_mask(elems)
        
        current_score = score_bitset(mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
        
        # Simulated annealing polish
        T = 1.0
        stagnation = 0
        max_stagnation = 15000
        
        while time.time() < deadline and stagnation < max_stagnation:
            # Pick element to replace
            idx_i = rng.randint(0, 31)
            i = elems[idx_i]
            
            # Pick new element not in set
            while True:
                j = rng.randint(1, 512)
                if not (mask & (1 << (j - 1))):
                    break
            
            new_mask = mask ^ (1 << (i - 1)) ^ (1 << (j - 1))
            new_score = score_bitset(new_mask)
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
                mask = new_mask
                current_score = new_score
                elems[idx_i] = j
                stagnation = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                
                if best_score == 255:
                    break
            else:
                stagnation += 1
            
            # Cooling
            T *= 0.999
            if T < 0.001:
                break
        
        num_restarts += 1
    
    return mask_to_list(best_mask)
# EVOLVE-BLOCK-END
