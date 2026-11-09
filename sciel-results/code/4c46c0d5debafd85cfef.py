import time
import random
from math import gcd

# Precomputation
def _precompute():
    N = 512
    primes = []
    for i in range(2, N):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    for a in range(1, N):
        m = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= (1 << prime_index[temp])
        masks[a] = m
        
    values = [0.0] * N
    for a in range(1, N):
        values[a] = 1.0 / (512 - a)
        
    return masks, values, primes

def _score(S, values):
    return sum(values[a] for a in S)

def _mask_of_set(S, masks):
    m = 0
    for a in S:
        m |= masks[a]
    return m

def _greedy_fill(S, masks, values, order=None):
    """
    Greedily adds best possible a to S (modifies S and returns new mask).
    S is a list/set.
    If order is None, use descending value order.
    """
    S_set = set(S)
    current_mask = _mask_of_set(S_set, masks)
    
    if order is None:
        candidates = list(range(511, 0, -1))
    else:
        candidates = list(order)
    
    for a in candidates:
        if a in S_set:
            continue
        if masks[a] & current_mask == 0:
            S_set.add(a)
            current_mask |= masks[a]
    
    return list(S_set), current_mask

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    masks, values, primes = _precompute()
    
    # Initial Greedy Solution
    S = []
    current_mask = 0
    for a in range(511, 0, -1):
        if masks[a] & current_mask == 0:
            S.append(a)
            current_mask |= masks[a]
            
    best_S = list(S)
    best_score = _score(best_S, values)
    
    # Local Search
    while time.time() < deadline:
        S = list(best_S)
        current_mask = _mask_of_set(S, masks)
        improved = False
        
        # Phase 1: Fill
        S, current_mask = _greedy_fill(S, masks, values)
        if _score(S, values) > best_score:
            best_S = list(S)
            best_score = _score(best_S, values)
        
        # Phase 2: Remove and Refill
        # Sort S by value ascending to try removing low value items first?
        # Or just iterate.
        S_sorted = sorted(S, key=lambda x: values[x])
        
        for b in S_sorted:
            if time.time() > deadline:
                break
                
            # Try removing b
            S_temp = [x for x in S if x != b]
            mask_temp = _mask_of_set(S_temp, masks)
            
            # Greedy Fill
            S_filled, mask_filled = _greedy_fill(S_temp, masks, values)
            
            score_temp = _score(S_filled, values)
            
            if score_temp > _score(S, values) + 1e-9:
                S = S_filled
                current_mask = mask_filled
                improved = True
                # Update best if better
                if score_temp > best_score:
                    best_S = list(S)
                    best_score = score_temp
                break # Restart the loop with new S
        
        if not improved:
            # Local optimum reached. Try a random perturbation or restart.
            # Simple perturbation: Pick a random element from best_S, remove it, greedily refill.
            if best_S:
                b = best_S[int(time.time() * 1000) % len(best_S)] # Pseudo-random based on time
                S_temp = [x for x in best_S if x != b]
                S_new, mask_new = _greedy_fill(S_temp, masks, values)
                score_new = _score(S_new, values)
                if score_new > best_score:
                    best_S = S_new
                    best_score = score_new
                # Even if not better, we can set S to S_new to explore new area?
                # For now, just update best if better. If stuck, this might not help much.
                # A better perturbation: Randomly swap two elements?
                # Or just restart with a slightly different initial greedy?
                pass
                
    return best_S
