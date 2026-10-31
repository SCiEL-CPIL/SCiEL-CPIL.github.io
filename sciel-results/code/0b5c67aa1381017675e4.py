import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        # Use the seed logic for general fallback if possible
        # The seed logic works for q=3, n>=4.
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial valid code: all words
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as bitmasks
    # sphere[i] is an integer with bits set for all words within distance 1 of all_words[i]
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask
        
    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_code = None
    best_size = N # worst case
    
    # Run multiple greedy restarts
    # We use a fixed number of restarts or until time runs out
    # Each restart is very fast (~50ms?), so we can do many.
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        
        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            best_candidate = -1
            
            for idx in candidates:
                # If this candidate is already chosen, skip? No, we don't remove from candidates list, 
                # but we should avoid picking the same one again? 
                # Actually, once picked, its sphere is fully covered, so gain will be 0.
                # We can skip if gain is 0.
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_candidate = idx
                    
            if best_gain == 0:
                # Should not happen if coverage is complete, but safety
                break
                
            chosen.append(best_candidate)
            covered |= spheres[best_candidate]
            
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_code = [all_words[i] for i in chosen]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]
