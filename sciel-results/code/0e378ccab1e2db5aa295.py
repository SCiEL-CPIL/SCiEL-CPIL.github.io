import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def list_to_mask(lst):
        mask = 0
        for x in lst:
            mask |= (1 << (x - 1))
        return mask

    def mask_to_list(mask):
        lst = []
        # Iterate through bits
        # Since max is 512, we check bits 0 to 511
        for i in range(512):
            if mask & (1 << i):
                lst.append(i + 1)
        return lst

    def score_mask(mask):
        count = 0
        # d > 0
        # Max d such that 1 + 2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are in mask
            # This is equivalent to checking if mask has a 1 at position a,
            # a+d, and a+2d.
            # mask & (mask << d) & (mask << 2*d) will have a 1 at position a
            # if a, a+d, a+2d are all set in mask.
            if mask & (mask << d) & (mask << (2 * d)):
                count += 1
        return count

    def score_list(lst):
        mask = list_to_mask(lst)
        return score_mask(mask)

    # Initial Candidates
    candidates = []
    
    # 1. Dense interval
    candidates.append(list(range(1, 33)))
    candidates.append(list(range(5, 37)))
    candidates.append(list(range(10, 42)))
    
    # 2. Sparse random
    rng = random.Random(42)
    for _ in range(20):
        s = set()
        while len(s) < 32:
            s.add(rng.randint(1, 512))
        candidates.append(sorted(s))
        
    # 3. Structured: mix of small and large
    for _ in range(10):
        small = set(rng.sample(range(1, 50), 16))
        large = set(rng.sample(range(463, 513), 16))
        if len(small | large) == 32:
            candidates.append(sorted(small | large))

    best_set = None
    best_score = -1
    best_mask = 0

    for cand in candidates:
        if len(cand) != 32 or len(set(cand)) != 32:
            continue
        if min(cand) < 1 or max(cand) > 512:
            continue
        sc = score_list(cand)
        if sc > best_score:
            best_score = sc
            best_set = cand[:]
            best_mask = list_to_mask(cand)

    current_set = best_set[:]
    current_mask = best_mask
    current_score = best_score

    # Local Search / Simulated Annealing
    # We will use a "swap" move: replace one element with a new random one.
    
    # To speed up, we maintain a set of current elements for O(1) lookup
    current_set_obj = set(current_set)
    
    max_iters = 2000000  # High limit, time is the real constraint
    iteration = 0
    temp = 10.0
    cooling = 0.99995
    
    while iteration < max_iters:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        # Pick a random index to replace
        idx = rng.randint(0, 31)
        old_val = current_set[idx]
        
        # Pick a new value not in the set
        # To avoid infinite loops if set is "full" (impossible here), 
        # we just pick random until unique
        new_val = rng.randint(1, 512)
        while new_val in current_set_obj:
            new_val = rng.randint(1, 512)
            
        # Apply move
        current_set[idx] = new_val
        current_set_obj.remove(old_val)
        current_set_obj.add(new_val)
        
        # Update mask
        new_mask = current_mask
        new_mask &= ~(1 << (old_val - 1))
        new_mask |= (1 << (new_val - 1))
        
        # Score
        new_score = score_mask(new_mask)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            current_score = new_score
            current_mask = new_mask
            if new_score > best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Accept with probability exp(delta / temp)
            if delta != 0:
                prob = np.exp(delta / temp)
                if rng.random() < prob:
                    current_score = new_score
                    current_mask = new_mask
                else:
                    # Revert
                    current_set[idx] = old_val
                    current_set_obj.remove(new_val)
                    current_set_obj.add(old_val)
                    current_mask = best_mask if new_mask == best_mask else (current_mask & ~(1 << (new_val - 1)) | (1 << (old_val - 1)))
                    # Actually, simpler revert:
                    current_mask = (current_mask & ~(1 << (new_val - 1))) | (1 << (old_val - 1))
            else:
                # delta == 0, revert to stay in basin or move? Usually revert to avoid noise
                current_set[idx] = old_val
                current_set_obj.remove(new_val)
                current_set_obj.add(old_val)
                current_mask = (current_mask & ~(1 << (new_val - 1))) | (1 << (old_val - 1))
                
        iteration += 1
        if iteration % 1000 == 0:
            temp *= cooling
            if temp < 0.1:
                temp = 0.1

    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    
    # Quick score check
    mask = 0
    for x in result:
        mask |= (1 << (x - 1))
    sc = 0
    for d in range(1, 256):
        if mask & (mask << d) & (mask << (2 * d)):
            sc += 1
    print(f"Score: {sc}")
    print(f"Set: {result}")
