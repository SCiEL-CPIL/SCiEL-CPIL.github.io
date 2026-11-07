"""Improved program: Precomputed bitmask comparability with Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute comparability bitmasks
    # Comp[i] is an integer where bit j is set if i and j are comparable
    Comp = [0] * N
    
    for i in range(N):
        mask = 0
        # i is comparable with all subsets of i
        sub = i
        while True:
            mask |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        # i is comparable with all supersets of i
        # Supersets of i are masks s such that (s & i) == i
        # This is equivalent to i | x for all x subset of (~i)
        # ~i in n bits is (N-1) ^ i
        super_base = i
        super_comp = (N - 1) ^ i
        sup = 0
        while True:
            s = super_base | sup
            mask |= (1 << s)
            if sup == 0:
                break
            sup = (sup - 1) & super_comp
        Comp[i] = mask
    
    # Initialize family with random distinct masks
    random.seed(42)
    all_masks = list(range(N))
    random.shuffle(all_masks)
    family = all_masks[:m]
    
    # Build family_mask: integer with bits set for masks in family
    fam_mask = 0
    for mask in family:
        fam_mask |= (1 << mask)
    
    # Function to calculate score of current family
    def calc_score(fam, fam_m):
        score = 0
        for mask in fam:
            # Count comparable masks in family
            # Comp[mask] & fam_m gives bits of comparable masks in family
            # bit_count gives the number
            score += (Comp[mask] & fam_m).bit_count()
        return score // 2  # Each pair counted twice
    
    current_score = calc_score(family, fam_mask)
    best_score = current_score
    best_family = family[:]
    
    # Simulated Annealing
    T = 1000.0
    cooling_rate = 0.999
    min_T = 0.1
    
    while T > min_T and time.time() < deadline:
        # Pick a random index in family
        idx = random.randint(0, m - 1)
        old_mask = family[idx]
        
        # Pick a random new mask not in family
        # We can pick a random mask from 0..N-1 and check if it's in fam_mask
        # Or maintain a list of available masks. Given N=4096, m=40, random pick is efficient.
        new_mask = random.randint(0, N - 1)
        if (fam_mask >> new_mask) & 1:
            # Already in family, try again
            continue
        
        # Calculate delta score
        # Remove old_mask from fam_mask temporarily
        new_fam_mask = fam_mask & ~(1 << old_mask)
        new_fam_mask |= (1 << new_mask)
        
        # Score change:
        # Old contribution of old_mask to score: (Comp[old_mask] & (fam_mask ^ (1<<old_mask))).bit_count()
        # New contribution of new_mask to score: (Comp[new_mask] & (new_fam_mask ^ (1<<new_mask))).bit_count()
        # Note: The rest of the family's contributions remain the same relative to each other,
        # except they lose comparability with old_mask and gain with new_mask.
        # Actually, it's easier to just compute the full score change.
        
        # Let's calculate the delta directly.
        # Score = sum_{x in F} comp(x, F\{x}) / 2
        # Delta = (comp(new, F\{old}) - comp(old, F\{old}))
        # comp(x, S) = (Comp[x] & S_mask).bit_count()
        
        # S = F \ {old}
        S_mask = fam_mask & ~(1 << old_mask)
        
        comp_old = (Comp[old_mask] & S_mask).bit_count()
        comp_new = (Comp[new_mask] & S_mask).bit_count()
        
        delta = comp_new - comp_old
        
        # Accept if delta > 0 or with probability exp(delta/T)
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            # Update family
            family[idx] = new_mask
            fam_mask = new_fam_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]
        
        T *= cooling_rate
    
    # Use best family found
    family = best_family
    
    # Convert to required format: list of lists of ints
    result = []
    for mask in family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result

# EVOLVE-BLOCK-END
