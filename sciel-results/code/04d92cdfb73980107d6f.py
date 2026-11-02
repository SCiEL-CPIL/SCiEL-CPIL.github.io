import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1.
    # We assume the evaluator passes these specific parameters as per the task description.
    # If other parameters are passed, we fall back to a simple construction.
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum construction (same as seed)
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial case for small n
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(7,1)
    total_words = 3 ** n
    
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute cover masks for each word
    # cover_masks[i] is an integer where bit j is 1 if word j is within radius of word i
    cover_masks = [0] * total_words
    for idx in range(total_words):
        w = to_tuple(idx)
        mask = 0
        # Self
        mask |= (1 << idx)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = to_index(tuple(new_w))
                    mask |= (1 << new_idx)
        cover_masks[idx] = mask

    best_code_size = float('inf')
    best_centers = []

    # Precompute for faster local search: 
    # For each center c, precompute the mask of words it uniquely covers given a set? No, that's dynamic.
    # Instead, we optimize the local search by using a smarter removal strategy.

    # Run multiple randomized greedy + local search iterations
    start_time = time.time()
    max_time = 115  # Leave 5 seconds buffer
    
    for iteration in range(1000):
        if time.time() - start_time > max_time:
            break
            
        random.seed(iteration)
        uncovered = (1 << total_words) - 1
        centers = []
        
        # Greedy Phase with slight randomization to escape local minima
        while uncovered:
            best_score = -1
            best_idx = -1
            # To speed up, we can just iterate all. 2187 is small.
            for i in range(total_words):
                # Count how many uncovered words this center covers
                score = (cover_masks[i] & uncovered).bit_count()
                # Add small random noise to break ties and diversify
                score += random.random() * 0.1 
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_score < 0.1: # Only noise, no actual coverage
                break
                
            centers.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        # Local Search: Remove redundant centers
        # Optimized: Instead of O(N^2) check for each removal, we can do:
        # 1. Compute union of all masks.
        # 2. For each center, check if its mask is a subset of (union - its mask).
        #    Actually, simpler: 
        #    union_mask = OR of all centers.
        #    For center c, if (cover_masks[c] & ~union_mask) == 0, it's redundant? No.
        #    If (cover_masks[c] & ~ (union_mask ^ cover_masks[c])) == 0? 
        #    Let S be the set of centers.
        #    c is redundant if cover_masks[c] is subset of (OR of masks of S \ {c}).
        #    Let U = OR of masks of S.
        #    Let U_without_c = U & ~cover_masks[c]. This is not OR of others, it's just U minus c's bits.
        #    Wait, U = mask_c | (OR of others).
        #    So (OR of others) = U & ~mask_c? No, that's not true. Bits in mask_c might also be in others.
        #    Actually, if a bit is in U and not in mask_c, it must be in others.
        #    If a bit is in mask_c and in U, it might be in others or not.
        #    So we cannot just use U.
        
        # However, we can speed up the check.
        # Current method: for each c, compute others_mask by OR-ing all other masks.
        # This is O(N^2 * bit_ops). N is ~180. 180^2 = 32400. 2187 bits is 34 64-bit words.
        # Python big ints are fast. 32400 * 34 ops is ~1M ops. Should be fast enough.
        
        # Let's try a different local search: 
        # Try to remove any center, then try to add any center if it helps? No, we want to minimize size.
        # So just removal.
        
        changed = True
        while changed:
            changed = False
            new_centers = []
            for c in centers:
                # Check if c is redundant
                others_mask = 0
                for d in centers:
                    if d != c:
                        others_mask |= cover_masks[d]
                
                # If (cover_masks[c] & ~others_mask) is 0, then c adds no unique coverage
                if (cover_masks[c] & ~others_mask) == 0:
                    changed = True
                else:
                    new_centers.append(c)
            centers = new_centers

        if len(centers) < best_code_size:
            best_code_size = len(centers)
            best_centers = centers.copy()

    # Convert indices back to tuples/lists
    result = [to_tuple(idx) for idx in best_centers]
    return result
