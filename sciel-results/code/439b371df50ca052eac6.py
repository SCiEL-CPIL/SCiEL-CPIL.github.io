import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute bit counts
    bit_counts = [0] * N
    for i in range(1, N):
        bit_counts[i] = bit_counts[i >> 1] + (i & 1)
    
    def score_family(family):
        """Count comparable pairs in family (list of masks)."""
        count = 0
        for i in range(m):
            mi = family[i]
            for j in range(i + 1, m):
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count
    
    # Initial configuration: build a chain-like structure
    # Start with sets of increasing size, sharing a common base
    # Use a "spine" of elements and vary the rest
    
    # Strategy: create sets that form multiple chains
    # Take a base set of size 6, then add elements one by one
    # But we need 40 distinct sets
    
    # Better: start with a greedy construction
    # Pick sets to maximize comparability
    
    # Initial: use sets of sizes 0-12, filling up to 40
    # Size 0: 1 set (empty)
    # Size 1: 12 sets
    # Size 2: C(12,2)=66, take 12
    # Size 3: C(12,3)=220, take 12
    # Size 4: take 4
    # Total: 1+12+12+12+4 = 41, adjust
    
    # Let me just use a random start and local search
    random.seed(42)
    
    # Initial: create a structured start
    # Use a "hub" element approach
    # Pick 6 core elements that appear in most sets
    core = list(range(6))
    periphery = list(range(6, 12))
    
    # Build initial family
    family = []
    # Empty set
    family.append(0)
    # Single core elements
    for i in core:
        family.append(1 << i)
    # Pairs of core
    for i in range(6):
        for j in range(i+1, 6):
            family.append((1 << core[i]) | (1 << core[j]))
    # That's 1 + 6 + 15 = 22
    # Triples of core
    for i in range(6):
        for j in range(i+1, 6):
            for k in range(j+1, 6):
                family.append((1 << core[i]) | (1 << core[j]) | (1 << core[k]))
                if len(family) >= m:
                    break
            if len(family) >= m:
                break
        if len(family) >= m:
            break
    
    # Trim to m
    family = family[:m]
    
    # Ensure distinct
    seen = set()
    final_family = []
    for mask in family:
        if mask not in seen:
            seen.add(mask)
            final_family.append(mask)
    # Fill if needed
    if len(final_family) < m:
        for mask in range(N):
            if mask not in seen:
                final_family.append(mask)
                seen.add(mask)
            if len(final_family) >= m:
                break
    
    family = final_family[:m]
    best_score = score_family(family)
    best_family = family[:]
    
    # Local search
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try random moves: replace one set with a random other set
        for _ in range(200):
            if time.time() > deadline:
                break
            
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Try a random new mask
            new_mask = random.randint(0, N - 1)
            if new_mask == old_mask:
                continue
            if new_mask in family:
                continue
            
            # Quick check: compute delta in score
            # Remove old_mask, add new_mask
            # For each other mask j, check if (old, j) was comparable and (new, j) is comparable
            
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                mj = family[j]
                # Was old comparable with mj?
                old_comp = False
                if (old_mask & mj) == old_mask and old_mask != mj:
                    old_comp = True
                elif (old_mask & mj) == mj and old_mask != mj:
                    old_comp = True
                
                # Is new comparable with mj?
                new_comp = False
                if (new_mask & mj) == new_mask and new_mask != mj:
                    new_comp = True
                elif (new_mask & mj) == mj and new_mask != mj:
                    new_comp = True
                
                if old_comp and not new_comp:
                    delta -= 1
                elif not old_comp and new_comp:
                    delta += 1
            
            if delta > 0:
                family[i] = new_mask
                best_score += delta
                best_family = family[:]
                improved = True
                no_improve = 0
                break
            elif delta == 0:
                # Accept neutral moves occasionally
                if random.random() < 0.1:
                    family[i] = new_mask
                    best_family = family[:]
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Try a different strategy: swap two elements
            if no_improve % 10 == 0:
                # Perturb: randomly replace 2-3 sets
                for _ in range(random.randint(2, 4)):
                    i = random.randint(0, m - 1)
                    new_mask = random.randint(0, N - 1)
                    # Ensure distinct
                    while new_mask in family:
                        new_mask = random.randint(0, N - 1)
                    family[i] = new_mask
        
        # Occasionally try a more structured move
        if no_improve % 50 == 0 and no_improve > 0:
            # Try to find a better set for a random position
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Try all masks that share many bits with old_mask
            candidates = []
            for mask in range(N):
                if mask in family:
                    continue
                # Hamming distance to old_mask
                hd = bin(mask ^ old_mask).count('1')
                if hd <= 2:
                    candidates.append(mask)
            
            random.shuffle(candidates)
            for new_mask in candidates[:50]:
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    mj = family[j]
                    old_comp = False
                    if (old_mask & mj) == old_mask and old_mask != mj:
                        old_comp = True
                    elif (old_mask & mj) == mj and old_mask != mj:
                        old_comp = True
                    
                    new_comp = False
                    if (new_mask & mj) == new_mask and new_mask != mj:
                        new_comp = True
                    elif (new_mask & mj) == mj and new_mask != mj:
                        new_comp = True
                    
                    if old_comp and not new_comp:
                        delta -= 1
                    elif not old_comp and new_comp:
                        delta += 1
                
                if delta > 0:
                    family[i] = new_mask
                    best_score += delta
                    best_family = family[:]
                    no_improve = 0
                    break
    
    # Convert to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
