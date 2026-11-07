"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

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
    deadline = time.time() + 85.0  # safety margin before 90s kill
    # Known good seed: 17..32 (16 elements) + 65..128 (64 elements) = 80
    seed_set = list(range(17, 33)) + list(range(65, 129))
    best = list(seed_set)
    best_size = len(best)
    
    # Bitmask representation for fast membership
    # bit i set means i is in the set (for i in 1..128)
    def to_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m
    
    def from_mask(m):
        return [i for i in range(1, 129) if (m >> i) & 1]
    
    def mask_has(m, x):
        return (m >> x) & 1
    
    def mask_add(m, x):
        return m | (1 << x)
    
    def mask_remove(m, x):
        return m & ~(1 << x)
    
    def try_add_fast(S, mask, x):
        """Try to add x to S. Returns (new_S, new_mask) or (None, mask) if invalid."""
        if mask_has(mask, x):
            return None, mask
        # Adding x creates a bad triple iff there exist distinct a,b in S with
        # a+b in S, a+x in S, and b+x in S.
        # For each a in S where a+x is in S, check if there's b in S (b!=a) 
        # with a+b in S and b+x in S.
        # b+x in S means b in {y-x : y in S, 1<=y-x<=128}
        # a+b in S means b in {y-a : y in S, 1<=y-a<=128}
        # So b must be in S ∩ (S-x) ∩ (S-a), b != a
        
        # Build S_minus_x = {y - x for y in S if 1 <= y-x <= 128} as a mask
        S_minus_x_mask = 0
        for a in S:
            y = a - x
            if 1 <= y <= 128:
                S_minus_x_mask |= (1 << y)
        
        if S_minus_x_mask == 0:
            return S + [x], mask_add(mask, x)
        
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
            # intersection_mask = mask & S_minus_x_mask & S_minus_a_mask
            # remove a from consideration
            inter = mask & S_minus_x_mask & S_minus_a_mask
            inter &= ~(1 << a)  # b must be distinct from a
            if inter != 0:
                return None, mask
        
        return S + [x], mask_add(mask, x)
    
    def try_remove_add_fast(S, mask, remove_x, add_x):
        """Try to remove remove_x and add add_x. Returns (new_S, new_mask) or (None, mask)."""
        if mask_has(mask, add_x) and add_x != remove_x:
            return None, mask
        if remove_x == add_x:
            return S, mask
        # First remove remove_x
        mask2 = mask_remove(mask, remove_x)
        S2 = [x for x in S if x != remove_x]
        # Now try to add add_x to S2
        return try_add_fast(S2, mask2, add_x)
    
    def greedy_add(S, mask, order=None):
        """Greedy add elements in given order."""
        if order is None:
            order = range(1, 129)
        for x in order:
            if time.time() > deadline:
                break
            if mask_has(mask, x):
                continue
            new_S, new_mask = try_add_fast(S, mask, x)
            if new_S is not None:
                S = new_S
                mask = new_mask
        return S, mask
    
    # Phase 1: Greedy add in natural order from seed
    S = list(seed_set)
    mask = to_mask(S)
    S, mask = greedy_add(S, mask)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Phase 2: Multiple random restarts with greedy
    for restart in range(50):
        if time.time() > deadline:
            break
        if restart == 0:
            S = list(best)
        elif restart < 20:
            # Perturb best: remove some random elements
            S = list(best)
            n_remove = random.randint(3, 10)
            to_remove = set(random.sample(range(len(S)), n_remove))
            S = [S[i] for i in range(len(S)) if i not in to_remove]
        else:
            # Random seed: take random elements
            S = []
            for x in random.sample(range(1, 129), random.randint(30, 50)):
                S.append(x)
        mask = to_mask(S)
        S, mask = greedy_add(S, mask)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Phase 3: Swap-based local search (remove one, add one)
    # This is the key improvement: we can exchange elements to find better configurations
    for iteration in range(20000):
        if time.time() > deadline:
            break
        S = list(best)
        mask = to_mask(S)
        n = len(S)
        if n == 0:
            break
        # Try removing a random element and adding a missing one
        idx = random.randint(0, n - 1)
        removed = S[idx]
        missing = [x for x in range(1, 129) if not mask_has(mask, x)]
        if not missing:
            break
        # Try a few candidates to add
        candidates = random.sample(missing, min(10, len(missing)))
        for add_x in candidates:
            new_S, new_mask = try_remove_add_fast(S, mask, removed, add_x)
            if new_S is not None and len(new_S) > best_size:
                best = new_S
                best_size = len(new_S)
                S = list(best)
                mask = to_mask(S)
                break
    
    # Phase 4: Greedy add again
    S = list(best)
    mask = to_mask(S)
    S, mask = greedy_add(S, mask)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Phase 5: More swap-based local search
    for iteration in range(30000):
        if time.time() > deadline:
            break
        S = list(best)
        mask = to_mask(S)
        n = len(S)
        if n == 0:
            break
        idx = random.randint(0, n - 1)
        removed = S[idx]
        missing = [x for x in range(1, 129) if not mask_has(mask, x)]
        if not missing:
            break
        candidates = random.sample(missing, min(15, len(missing)))
        for add_x in candidates:
            new_S, new_mask = try_remove_add_fast(S, mask, removed, add_x)
            if new_S is not None and len(new_S) > best_size:
                best = new_S
                best_size = len(new_S)
                S = list(best)
                mask = to_mask(S)
                break
    
    # Phase 6: Two-swap local search (remove two, add two)
    for iteration in range(10000):
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
        # Try adding back the removed ones and some new ones
        candidates = [removed1, removed2] + random.sample(missing, min(10, len(missing) - 2))
        best_S2 = None
        best_S2_mask = None
        best_S2_size = len(S2)
        for x in candidates:
            new_S, new_mask = try_add_fast(S2, mask2, x)
            if new_S is not None and len(new_S) > best_S2_size:
                best_S2 = new_S
                best_S2_mask = new_mask
                best_S2_size = len(new_S)
        if best_S2 is not None and best_S2_size > best_size:
            best = best_S2
            best_size = best_S2_size
    
    # Phase 7: Final greedy add
    S = list(best)
    mask = to_mask(S)
    S, mask = greedy_add(S, mask)
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Final verification
    if not _is_valid(best):
        # Fallback to known valid set
        best = list(seed_set)
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
