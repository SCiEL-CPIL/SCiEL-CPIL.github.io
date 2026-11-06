import itertools
import random
import time

def construct():
    n = 16
    deadline = time.time() + 115.0
    
    # Precompute all triples and their properties
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute supersets for each triple mask
    supersets_cache = {}
    for t in all_triples:
        mask = triple_masks[t]
        free = (~mask) & 0xFFFF
        subs = []
        sub = free
        while True:
            subs.append(mask | sub)
            if sub == 0:
                break
            sub = (sub - 1) & free
        supersets_cache[t] = subs

    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute masks sorted by popcount descending for fast alpha calculation
    # We want the largest mask with cover_count == 0
    sorted_masks = sorted(range(1 << n), key=lambda x: popcount[x], reverse=True)

    def calculate_alpha(cover_count):
        """
        Find max popcount of mask where cover_count[mask] == 0.
        Iterates masks in descending order of size.
        """
        for mask in sorted_masks:
            if cover_count[mask] == 0:
                return popcount[mask]
        return 0

    def generate_random_packing():
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            # Encode pairs as integers for faster hashing
            p1 = a * 16 + b
            p2 = a * 16 + c
            p3 = b * 16 + c
            ps = {p1, p2, p3}
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        return triples, used_pairs

    def get_pairs(t):
        a, b, c = t
        return {a * 16 + b, a * 16 + c, b * 16 + c}

    def is_valid_addition(t, used_pairs):
        ps = get_pairs(t)
        if ps & used_pairs:
            return False
        return True

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    # Main search loop
    while time.time() < deadline:
        triples, used_pairs = generate_random_packing()
        
        # Initialize cover_count
        cover_count = [0] * (1 << n)
        for t in triples:
            for sup in supersets_cache[t]:
                cover_count[sup] += 1
        
        current_alpha = calculate_alpha(cover_count)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        if best_alpha <= 5:
            break
            
        # Local Search
        # We will perform swaps
        search_deadline = min(time.time() + 1.0, deadline - 2.0)
        
        while time.time() < search_deadline:
            if len(triples) < 2:
                break
                
            idx = rng.randrange(len(triples))
            t_old = triples[idx]
            pairs_old = get_pairs(t_old)
            
            # Remove t_old
            used_pairs.difference_update(pairs_old)
            for sup in supersets_cache[t_old]:
                cover_count[sup] -= 1
                
            # Find a replacement
            # Try random candidates, but limit the number of attempts per step to avoid wasting time
            found = False
            attempts = 0
            max_attempts = 50 # Limit attempts to keep local search fast
            
            # Shuffle a subset of candidates to try
            # Instead of shuffling all 560, we can just pick random ones
            # But we need to ensure we don't pick t_old
            candidates = [t for t in all_triples if t != t_old]
            rng.shuffle(candidates)
            
            for t_new in candidates:
                if attempts >= max_attempts:
                    break
                attempts += 1
                
                if is_valid_addition(t_new, used_pairs):
                    # Valid
                    pairs_new = get_pairs(t_new)
                    triples[idx] = t_new
                    used_pairs.update(pairs_new)
                    
                    for sup in supersets_cache[t_new]:
                        cover_count[sup] += 1
                        
                    new_alpha = calculate_alpha(cover_count)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                        
                    # Accept if better or equal
                    if new_alpha <= current_alpha:
                        current_alpha = new_alpha
                        found = True
                        break
                    else:
                        # Revert
                        used_pairs.difference_update(pairs_new)
                        for sup in supersets_cache[t_new]:
                            cover_count[sup] -= 1
                        triples[idx] = t_old
                        used_pairs.update(pairs_old)
                        for sup in supersets_cache[t_old]:
                            cover_count[sup] += 1
            if not found:
                # Revert if no improvement found
                used_pairs.update(pairs_old)
                for sup in supersets_cache[t_old]:
                    cover_count[sup] += 1
                triples[idx] = t_old

    return best_triples
