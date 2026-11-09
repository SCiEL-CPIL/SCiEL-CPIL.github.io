import time
import random

# Precomputation
N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples (masks)
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

# Precompute supersets for each triple
SUPERSETS = [None] * len(TRIPLES)
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}
# Map from mask to list of triples contained in it? No, that's expensive.
# We just need to find triples subset of a given mask I.
# Since I has size alpha (likely 5-7), we can iterate all triples and check subset.
# 560 triples is small.

def construct():
    deadline = time.time() + 110.0
    
    def calc_alpha_and_set(triple_masks):
        valid = [True] * N_MASKS
        for t in triple_masks:
            idx = MASK_TO_INDEX[t]
            for s in SUPERSETS[idx]:
                valid[s] = False
        
        # Find max popcount of valid mask
        # Optimization: Check from high popcount down
        # We can pre-sort masks by popcount
        for s in SORTED_MASKS:
            if valid[s]:
                return POP_COUNT[s], s
        return 0, 0

    def can_add(triple_masks, new_t):
        for t in triple_masks:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    _random = random.Random(42)
    
    # Initialize with a greedy maximal system
    triples = []
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(triples, t):
            triples.append(t)
            
    current_alpha, current_set = calc_alpha_and_set(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    # Directed Local Search
    while time.time() < deadline:
        # Find triples that are subsets of the current independent set
        # These are the "best" candidates for addition to break the set
        candidates_in_set = []
        for t in TRIPLES:
            if (t & ~current_set) == 0: # t is subset of current_set
                if t not in triples:
                    candidates_in_set.append(t)
        
        moves_made = False
        
        # Try adding candidates directly
        _random.shuffle(candidates_in_set)
        for t in candidates_in_set:
            if can_add(triples, t):
                new_triples = triples + [t]
                new_alpha, new_set = calc_alpha_and_set(new_triples)
                if new_alpha < current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    current_set = new_set
                    moves_made = True
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]
                    break # Restart loop with new state
                # If alpha didn't decrease, we might still accept if we want to explore, 
                # but adding a triple inside the set *must* decrease the size of that specific set, 
                # though another set of same size might exist. 
                # Actually, if we add T subset I, I is no longer independent. 
                # The new alpha is at most current_alpha - 1? Not necessarily, there might be another set of size current_alpha.
                # But it's a good move to try.
        
        if moves_made:
            continue
            
        # If no direct addition worked, try swaps
        # Remove a triple that conflicts with a candidate, and add the candidate
        # Or just random swap with bias towards candidates
        
        # Pick a random candidate from candidates_in_set (if any)
        if candidates_in_set:
            t_in = _random.choice(candidates_in_set)
        else:
            # If no triple is fully inside the independent set, we are stuck or optimal?
            # We might need to change the structure to allow a triple inside a *different* large set.
            # For now, fallback to random move
            t_in = _random.choice(TRIPLES)
            if t_in in triples:
                continue

        # Find a triple to remove that conflicts with t_in
        # We want to remove a triple T_out such that T_out & t_in has >= 2 points
        # This allows t_in to be added after removal
        conflicting = [t for t in triples if (t & t_in).bit_count() >= 2]
        
        if conflicting:
            t_out = _random.choice(conflicting)
            new_triples = [t for t in triples if t != t_out]
            if can_add(new_triples, t_in):
                new_triples.append(t_in)
                new_alpha, new_set = calc_alpha_and_set(new_triples)
                if new_alpha <= current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    current_set = new_set
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]
        else:
            # No conflict, but can't add? 
            # This implies t_in is not in triples and doesn't conflict, so it SHOULD have been added.
            # Unless it was in candidates_in_set but failed can_add?
            # If t_in is not in triples and no conflict, can_add returns True.
            # So this case shouldn't happen if candidates_in_set was filtered correctly.
            # But if candidates_in_set was empty, we picked random t_in.
            if t_in not in triples and can_add(triples, t_in):
                triples.append(t_in)
                new_alpha, new_set = calc_alpha_and_set(triples)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    current_set = new_set
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]

    # Convert best_triples (list of masks) to list of lists of points
    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result

# Precomputations need to be done before construct is called or inside.
# To avoid overhead on first call, do it at module level.
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)

# Fix the construct function to use module-level precomputations
def construct():
    deadline = time.time() + 110.0
    
    def calc_alpha_and_set(triple_masks):
        valid = [True] * N_MASKS
        for t in triple_masks:
            idx = MASK_TO_INDEX[t]
            for s in SUPERSETS[idx]:
                valid[s] = False
        
        # Find max popcount of valid mask
        for s in SORTED_MASKS:
            if valid[s]:
                return POP_COUNT[s], s
        return 0, 0

    def can_add(triple_masks, new_t):
        for t in triple_masks:
            if (t & new_t).bit_count() >= 2:
                return False
        return True

    _random = random.Random(42)
    
    # Initialize with a greedy maximal system
    triples = []
    available = TRIPLES[:]
    _random.shuffle(available)
    for t in available:
        if can_add(triples, t):
            triples.append(t)
            
    current_alpha, current_set = calc_alpha_and_set(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    # Directed Local Search
    while time.time() < deadline:
        # Find triples that are subsets of the current independent set
        candidates_in_set = []
        for t in TRIPLES:
            if (t & ~current_set) == 0: # t is subset of current_set
                if t not in triples:
                    candidates_in_set.append(t)
        
        moves_made = False
        
        # Try adding candidates directly
        _random.shuffle(candidates_in_set)
        for t in candidates_in_set:
            if can_add(triples, t):
                new_triples = triples + [t]
                new_alpha, new_set = calc_alpha_and_set(new_triples)
                if new_alpha < current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    current_set = new_set
                    moves_made = True
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]
                    break # Restart loop with new state
        
        if moves_made:
            continue
            
        # If no direct addition worked, try swaps
        if candidates_in_set:
            t_in = _random.choice(candidates_in_set)
        else:
            # Fallback to random exploration if stuck
            t_in = _random.choice(TRIPLES)
            if t_in in triples:
                continue

        # Find a triple to remove that conflicts with t_in
        conflicting = [t for t in triples if (t & t_in).bit_count() >= 2]
        
        if conflicting:
            t_out = _random.choice(conflicting)
            new_triples = [t for t in triples if t != t_out]
            if can_add(new_triples, t_in):
                new_triples.append(t_in)
                new_alpha, new_set = calc_alpha_and_set(new_triples)
                if new_alpha <= current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    current_set = new_set
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]
        else:
            if t_in not in triples and can_add(triples, t_in):
                triples.append(t_in)
                new_alpha, new_set = calc_alpha_and_set(triples)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    current_set = new_set
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = triples[:]

    result = []
    for mask in best_triples:
        pts = []
        for i in range(N_POINTS):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
