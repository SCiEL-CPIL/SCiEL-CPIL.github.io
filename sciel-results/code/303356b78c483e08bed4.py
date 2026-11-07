"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_valid(S):
    """Check that no three distinct a,b,c in S have all three pair sums in S."""
    s = set(S)
    for i, a in enumerate(S):
        for j in range(i+1, len(S)):
            b = S[j]
            ab = a + b
            if ab not in s:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in s and b + c in s:
                    return False
    return True

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    # Known good seed: 17..32 (16 elements) + 65..128 (64 elements) = 80
    seed_set = list(range(17, 33)) + list(range(65, 129))
    best = list(seed_set)
    best_size = len(best)
    
    # Bitmask representation for fast membership
    def to_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m
    
    def mask_has(m, x):
        return (m >> x) & 1
    
    def mask_add(m, x):
        return m | (1 << x)
    
    def mask_remove(m, x):
        return m & ~(1 << x)
    
    def count_bits(m):
        c = 0
        while m:
            m &= m - 1
            c += 1
        return c
    
    def try_add_fast(S, mask, x):
        """Try to add x to S. Returns True if valid, False otherwise."""
        if mask_has(mask, x):
            return False
        # Adding x creates a bad triple iff there exist distinct a,b in S with
        # a+b in S, a+x in S, and b+x in S.
        # For each a in S where a+x is in S, check if there's b in S (b!=a) 
        # with a+b in S and b+x in S.
        # b+x in S means b in {y-x : y in S, 1<=y-x<=128}
        # a+b in S means b in {y-a : y in S, 1<=y-a<=128}
        
        # Build S_minus_x = {y - x for y in S if 1 <= y-x <= 128} as a mask
        S_minus_x_mask = 0
        for a in S:
            y = a - x
            if 1 <= y <= 128:
                S_minus_x_mask |= (1 << y)
        
        if S_minus_x_mask == 0:
            return True
        
        for a in S:
            ax = a + x
            if ax > 128 or not mask_has(mask, ax):
                continue
            # Build S_minus_a = {y - a for y in S if 1 <= y-a <= 128}
            S_minus_a_mask = 0
            for b in S:
                y = b - a
                if 1 <= y <= 128:
                    S_minus_a_mask |= (1 << y)
            
            # Need b in S ∩ S_minus_x ∩ S_minus_a, b != a
            inter = mask & S_minus_x_mask & S_minus_a_mask
            inter &= ~(1 << a)  # b must be distinct from a
            if inter != 0:
                return False
        
        return True
    
    def greedy_add(S, mask, order=None):
        """Greedy add elements in given order."""
        if order is None:
            order = list(range(1, 129))
        for x in order:
            if time.time() > deadline:
                break
            if mask_has(mask, x):
                continue
            if try_add_fast(S, mask, x):
                S = S + [x]
                mask = mask_add(mask, x)
        return S, mask
    
    def greedy_add_scored(S, mask):
        """Greedy add elements, choosing the one that causes fewest future conflicts."""
        missing = [x for x in range(1, 129) if not mask_has(mask, x)]
        if not missing:
            return S, mask
        
        while missing and time.time() < deadline:
            best_x = None
            best_score = -1
            for x in missing:
                if not try_add_fast(S, mask, x):
                    continue
                # Score: number of elements that become harder to add after adding x
                # Approximate by counting how many y in missing have y-x in S or x-y in S
                conflict_count = 0
                for y in missing:
                    if y == x:
                        continue
                    if mask_has(mask, x + y) or mask_has(mask, y - x) or mask_has(mask, x - y):
                        conflict_count += 1
                score = -conflict_count  # We want to minimize conflicts
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            
            S = S + [best_x]
            mask = mask_add(mask, best_x)
            missing.remove(best_x)
        
        return S, mask
    
    # Phase 1: Greedy add in natural order from seed
    S = list(seed_set)
    mask = to_mask(S)
    S, mask = greedy_add(S, mask)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Phase 2: Multiple random restarts with greedy
    for restart in range(30):
        if time.time() > deadline:
            break
        if restart == 0:
            S = list(best)
        elif restart < 15:
            # Perturb best: remove some random elements
            S = list(best)
            n_remove = random.randint(3, 8)
            to_remove = set(random.sample(range(len(S)), min(n_remove, len(S))))
            S = [S[i] for i in range(len(S)) if i not in to_remove]
        else:
            # Random seed: take random elements
            S = list(random.sample(range(1, 129), random.randint(40, 60)))
        
        mask = to_mask(S)
        S, mask = greedy_add(S, mask)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Phase 3: Swap-based local search with systematic exploration
    # Precompute missing elements
    for iteration in range(50000):
        if time.time() > deadline:
            break
        S = list(best)
        mask = to_mask(S)
        n = len(S)
        if n == 0:
            break
        missing = [x for x in range(1, 129) if not mask_has(mask, x)]
        if not missing:
            break
        
        # Try removing each element and adding the best missing one
        improved = False
        for idx in range(n):
            removed = S[idx]
            mask2 = mask_remove(mask, removed)
            S2 = [x for x in S if x != removed]
            
            # Find the best missing element to add
            best_add = None
            for add_x in missing:
                if mask_has(mask2, add_x):
                    continue
                if try_add_fast(S2, mask2, add_x):
                    best_add = add_x
                    break
            
            if best_add is not None:
                new_S = S2 + [best_add]
                new_mask = mask_add(mask2, best_add)
                if len(new_S) > best_size:
                    best = new_S
                    best_size = best_size + 1
                    improved = True
                    break
        
        if not improved:
            # Try random swaps
            idx = random.randint(0, n - 1)
            removed = S[idx]
            candidates = random.sample(missing, min(20, len(missing)))
            for add_x in candidates:
                mask2 = mask_remove(mask, removed)
                S2 = [x for x in S if x != removed]
                if try_add_fast(S2, mask2, add_x):
                    new_S = S2 + [add_x]
                    if len(new_S) > best_size:
                        best = new_S
                        best_size = best_size + 1
                        break
    
    # Phase 4: Two-swap local search
    for iteration in range(20000):
        if time.time() > deadline:
            break
        S = list(best)
        mask = to_mask(S)
        n = len(S)
        if n < 2:
            break
        
        # Remove two random elements
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 2)
        if idx2 >= idx1:
            idx2 += 1
        removed1 = S[idx1]
        removed2 = S[idx2]
        mask2 = mask_remove(mask_remove(mask, removed1), removed2)
        S2 = [x for x in S if x != removed1 and x != removed2]
        
        missing = [x for x in range(1, 129) if not mask_has(mask2, x)]
        if len(missing) < 2:
            continue
        
        # Try adding pairs of missing elements
        found = False
        for i in range(min(15, len(missing))):
            x1 = missing[i]
            if not try_add_fast(S2, mask2, x1):
                continue
            S3 = S2 + [x1]
            mask3 = mask_add(mask2, x1)
            for j in range(i+1, min(15, len(missing))):
                x2 = missing[j]
                if mask_has(mask3, x2):
                    continue
                if try_add_fast(S3, mask3, x2):
                    new_S = S3 + [x2]
                    if len(new_S) > best_size:
                        best = new_S
                        best_size = best_size + 2
                        found = True
                        break
            if found:
                break
    
    # Phase 5: Final greedy add
    S = list(best)
    mask = to_mask(S)
    S, mask = greedy_add(S, mask)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Final verification
    if not _is_valid(best):
        best = list(seed_set)
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
