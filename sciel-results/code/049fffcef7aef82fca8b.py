import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin
    
    N = 512
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Precompute coprime masks
    NUM_WORDS = (N + 63) // 64
    coprime_masks = [[0] * NUM_WORDS for _ in range(N)]
    
    for a in range(1, N):
        mask = coprime_masks[a]
        for b in range(1, N):
            if gcd(a, b) == 1:
                mask[b // 64] |= (1 << (b % 64))
                
    def mask_and(masks):
        if not masks:
            return [(1 << 64) - 1] * NUM_WORDS
        result = list(masks[0])
        for m in masks[1:]:
            for i in range(NUM_WORDS):
                result[i] &= m[i]
        return result
        
    def get_random_set_bit(mask):
        # Count bits
        total = 0
        for w in mask:
            total += bin(w).count('1')
        if total == 0:
            return -1
        r = random.randint(0, total - 1)
        for i in range(NUM_WORDS):
            w = mask[i]
            cnt = bin(w).count('1')
            if r < cnt:
                # Find r-th set bit
                idx = 0
                for bit in range(64):
                    if w & (1 << bit):
                        if idx == r:
                            return i * 64 + bit
                        idx += 1
            else:
                r -= cnt
        return -1

    # Greedy Initialization
    chosen = []
    chosen_set = set()
    chosen_bits = [0] * NUM_WORDS
    
    for a in range(N - 1, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            chosen_set.add(a)
            chosen_bits[a // 64] |= (1 << (a % 64))
            
    current_score = sum(weights[a] for a in chosen)
    best_score = current_score
    best_set = list(chosen)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 1e-6
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            T = min_T
            
        # Pick random a to remove
        if not chosen:
            break
        idx = random.randint(0, len(chosen) - 1)
        a = chosen[idx]
        
        # Compute others_mask
        # AND of coprime_masks for all x in chosen except a
        masks = [coprime_masks[x] for x in chosen if x != a]
        others_mask = mask_and(masks)
        
        # Candidates: in others_mask, not in chosen_bits
        candidates = [0] * NUM_WORDS
        for i in range(NUM_WORDS):
            candidates[i] = others_mask[i] & ~chosen_bits[i]
            
        # Pick random b from candidates
        b = get_random_set_bit(candidates)
        if b == -1 or b >= N:
            continue
            
        # Calculate delta
        delta = weights[b] - weights[a]
        
        # Accept?
        accept = False
        if delta > 0:
            accept = True
        else:
            # SA probability
            # exp(delta / T)
            # If T is very small, exp(delta/T) is 0 for delta < 0
            if random.random() < (2.718281828 ** (delta / T)):
                accept = True
                
        if accept:
            # Perform swap
            chosen[idx] = b
            chosen_set.remove(a)
            chosen_set.add(b)
            chosen_bits[a // 64] &= ~(1 << (a % 64))
            chosen_bits[b // 64] |= (1 << (b % 64))
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(chosen)
                
        T *= cooling
        
    return best_set
