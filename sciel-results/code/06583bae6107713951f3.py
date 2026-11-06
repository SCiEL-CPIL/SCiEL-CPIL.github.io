import time
import random
import math

def _score_mask(mask):
    """
    Fast score using Python big integers as bitsets.
    mask: int where bit i is 1 if (i+1) is in the set.
    Returns the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _mask_to_list(mask):
    """Convert mask to list of integers (1..512)."""
    res = []
    m = mask
    i = 0
    while m:
        if m & 1:
            res.append(i + 1)
        m >>= 1
        i += 1
    return res

def _list_to_mask(S):
    """Convert list of integers (1..512) to mask."""
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    return mask

def _greedy_construct(deadline, rng):
    """
    Greedily construct a set of 32 integers by adding one element at a time,
    choosing the element that maximizes the marginal gain in progression differences.
    """
    N = 32
    MAXV = 512
    
    # Start with an empty set
    current_mask = 0
    current_score = 0
    selected = set()
    
    # Precompute all possible values
    all_values = list(range(1, MAXV + 1))
    rng.shuffle(all_values)
    
    for step in range(N):
        if time.time() > deadline:
            break
        
        best_val = None
        best_new_score = current_score
        best_new_mask = current_mask
        
        # Try all remaining values, but limit to a random subset for speed
        # Actually, let's try all values since scoring is fast
        # But we need to be careful about time
        remaining = [v for v in all_values if v not in selected]
        
        # Shuffle to randomize tie-breaking
        rng.shuffle(remaining)
        
        # Try a subset if too many remain
        if len(remaining) > 100:
            # Just try first 100 in shuffled order
            candidates = remaining[:100]
        else:
            candidates = remaining
        
        for v in candidates:
            if time.time() > deadline:
                break
            new_mask = current_mask | (1 << (v - 1))
            new_score = _score_mask(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_val = v
                best_new_mask = new_mask
        
        if best_val is not None:
            current_mask = best_new_mask
            current_score = best_new_score
            selected.add(best_val)
        else:
            # Fallback: pick a random remaining value
            if remaining:
                v = remaining[0]
                current_mask |= (1 << (v - 1))
                current_score = _score_mask(current_mask)
                selected.add(v)
    
    return current_mask, current_score

def _simulated_annealing(mask, score, deadline, rng):
    """
    Refine a set using simulated annealing with fast bitset scoring.
    """
    N = 32
    MAXV = 512
    
    best_mask = mask
    best_score = score
    
    # Build list from mask for easy element selection
    S = _mask_to_list(mask)
    current_score = score
    current_mask = mask
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T >= T_min:
        # Pick random element to remove
        idx_out = rng.randint(0, len(S) - 1)
        x = S[idx_out]
        
        # Pick random y to add
        y = rng.randint(1, MAXV)
        
        if y == x:
            continue
        
        # Check if y is already in mask
        if (current_mask >> (y - 1)) & 1:
            continue
        
        # Apply move
        new_mask = (current_mask & ~(1 << (x - 1))) | (1 << (y - 1))
        new_score = _score_mask(new_mask)
        delta = new_score - current_score
        
        # Acceptance probability
        if delta > 0 or (delta < 0 and rng.random() < math.exp(delta / T)):
            current_mask = new_mask
            current_score = new_score
            S[idx_out] = y
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        # else: revert (do nothing since we didn't modify in place)
        
        T *= cooling
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 110s.
    deadline = start_time + 105.0
    
    best_score = 0
    best_mask = 0
    
    rng = random.Random(12345)
    
    # Strategy:
    # 1. Multiple greedy constructions with different randomizations
    # 2. Refine each with simulated annealing
    # 3. Keep track of the best result
    
    num_greedy = 8
    
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        # Greedy construction
        mask, score = _greedy_construct(deadline - 30, rng)
        
        if score > best_score:
            best_score = score
            best_mask = mask
        
        # Refine with simulated annealing
        # Give SA some time, but not too much
        sa_deadline = min(deadline, time.time() + 10.0)
        mask2, score2 = _simulated_annealing(mask, score, sa_deadline, rng)
        
        if score2 > best_score:
            best_score = score2
            best_mask = mask2
    
    # If we still have time, do one more SA pass on the best
    if time.time() < deadline - 5:
        mask3, score3 = _simulated_annealing(best_mask, best_score, deadline - 2, rng)
        if score3 > best_score:
            best_score = score3
            best_mask = mask3
    
    return _mask_to_list(best_mask)
