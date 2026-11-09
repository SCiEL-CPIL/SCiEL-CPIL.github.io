import time
import random
import itertools

def construct():
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    best_family = None
    best_score = -1
    
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    def is_subset(a, b):
        return (a | b) == b
    
    def score_family(family):
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_subset(family[i], family[j]) or is_subset(family[j], family[i]):
                    count += 1
        return count

    def get_delta(family, set_family, idx, new_mask):
        old_mask = family[idx]
        gain = 0
        loss = 0
        # Optimization: Precompute subset relations for neighbors?
        # No, just calculate on the fly.
        for j in range(len(family)):
            if j == idx:
                continue
            other = family[j]
            if is_subset(old_mask, other) or is_subset(other, old_mask):
                loss += 1
            if is_subset(new_mask, other) or is_subset(other, new_mask):
                gain += 1
        return gain - loss

    def hill_climb(family):
        family = list(family)
        set_family = set(family)
        current_score = score_family(family)
        improved = True
        while improved:
            improved = False
            # Try to improve each position
            for i in range(len(family)):
                old_mask = family[i]
                best_new_mask = None
                best_delta = 0
                # Iterate through all possible candidates not in family
                # To speed up, we can iterate through all_masks and skip those in set_family
                for candidate in all_masks:
                    if candidate in set_family:
                        continue
                    delta = get_delta(family, set_family, i, candidate)
                    if delta > best_delta:
                        best_delta = delta
                        best_new_mask = candidate
                
                if best_new_mask is not None and best_delta > 0:
                    # Update family
                    set_family.remove(old_mask)
                    family[i] = best_new_mask
                    set_family.add(best_new_mask)
                    current_score += best_delta
                    improved = True
        return family, current_score

    def generate_initial_family():
        # Generate 3 chains of 13 masks based on random permutations
        family = []
        used = set()
        
        for _ in range(3):
            perm = list(range(n))
            random.shuffle(perm)
            mask = 0
            for i in range(n + 1):
                if i > 0:
                    # Add the i-th element in permutation
                    # The chain is: {}, {p0}, {p0, p1}, ...
                    # Mask for k elements is sum of 1 << perm[0..k-1]
                    # Let's build incrementally
                    pass
            
            # Re-doing chain generation properly
            mask = 0
            for k in range(n + 1):
                if k > 0:
                    mask |= (1 << perm[k-1])
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                else:
                    # Collision, pick a random unused mask
                    # This is rare with random perms
                    while True:
                        rm = random.choice(all_masks)
                        if rm not in used:
                            family.append(rm)
                            used.add(rm)
                            break
        
        # We have 39 masks. Add one more.
        while len(family) < m:
            rm = random.choice(all_masks)
            if rm not in used:
                family.append(rm)
                used.add(rm)
                
        return family

    # Main loop
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        init_fam = generate_initial_family()
        # Randomize the initial family slightly to help escape local optima?
        # Or just run hill climb.
        final_fam, final_score = hill_climb(init_fam)
        
        if final_score > best_score:
            best_score = final_score
            best_family = final_fam
        
        # If we found a very high score, we might stop early?
        # Max possible score?
        # If we had a chain of 40, score = 780.
        # But max chain is 13.
        # 3 chains of 13 = 234.
        # If we can get close to 300+, it's good.
        # Let's run a few restarts to be safe.
        if restarts > 20 and time.time() > deadline - 10:
             break

    if best_family is None:
        # Fallback: seed program logic
        return [[b for b in range(n) if (mask >> b) & 1] for mask in range(m)]
    
    # Convert masks to lists of ints
    result = []
    for mask in best_family:
        s = [b for b in range(n) if (mask >> b) & 1]
        result.append(s)
    
    return result
