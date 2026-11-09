import time
import random
import sys

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Pre-compute comparability: comp[i][j] is True if i and j are comparable
    # We represent comp[i] as a Python int bitmask where bit j is set if i and j are comparable
    # This allows fast bitwise operations for counting comparables.
    comp = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                bits |= (1 << j)
        comp[i] = bits

    # Helper to count comparable pairs in a family of masks
    def count_pairs(family):
        c = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            # We only count pairs (i, j) with i < j to avoid double counting
            # But comp[fi] includes all comparable j.
            # We need to mask out indices < i and i itself.
            # Simpler: just iterate j > i and check comp[fi] bit j.
            # Or use bit_count on (comp[fi] & mask_of_indices_greater_than_i)
            # Since m=40, simple loop is fast enough.
            for j in range(i + 1, k):
                fj = family[j]
                if (comp[fi] >> fj) & 1:
                    c += 1
        return c

    # Greedy initialization: build a family by adding the set with max immediate gain
    def greedy_init():
        family = []
        family_set = set()
        
        # Start with empty set (0) as it's comparable to everything
        family.append(0)
        family_set.add(0)
        
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            
            # Optimization: We can pre-compute gains for all candidates?
            # Or just scan. 4096 * 40 checks is ~160k ops, fast.
            for s in range(total_masks):
                if s in family_set:
                    continue
                
                gain = 0
                for f in family:
                    if (comp[f] >> s) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
            else:
                # Should not happen unless m > 4096
                break
        
        return family

    # 2-Opt Local Search: Try replacing each set with any other set in the universe
    # This is the "exhaustive" refinement mentioned in the findings.
    def two_opt_search(family, deadline):
        current_score = count_pairs(family)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                old_mask = family[i]
                best_new_mask = old_mask
                best_new_score = current_score
                
                # Try all possible replacements
                for new_mask in range(total_masks):
                    if new_mask == old_mask:
                        continue
                    # Check if new_mask is already in family (excluding i)
                    # We can do this by checking if new_mask is in family_set
                    # But we don't have a set representation here for speed in the loop?
                    # Let's maintain a set for the current family.
                    
                    # Actually, checking "is in family" for 4096 * 40 * 40 is too slow.
                    # Let's use a boolean array or set.
                    # We'll pass a set or maintain it.
                    
                    # Let's optimize: 
                    # 1. Calculate delta score efficiently?
                    #    Old score contribution of i: sum(1 for j!=i if comparable(family[i], family[j]))
                    #    New score contribution of i: sum(1 for j!=i if comparable(new_mask, family[j]))
                    #    Delta = new_contrib - old_contrib
                    #    This is O(m) per candidate. Total O(m * 4096 * m) = 40 * 4096 * 40 ~ 6.5M ops per pass.
                    #    This is feasible in Python? 6.5M simple ops might take 0.1-0.5s.
                    #    Let's try.
                    
                    # Pre-compute comparability of new_mask with all family members?
                    # comp[new_mask] is a bitmask.
                    # We need to count how many family[j] (j != i) are comparable to new_mask.
                    # Let family_masks = family
                    # We can create a bitmask of the family? No, masks are up to 4095.
                    # We can iterate j and check (comp[new_mask] >> family[j]) & 1.
                    
                    # Let's compute the delta.
                    old_contrib = 0
                    for j in range(m):
                        if j == i: continue
                        if (comp[old_mask] >> family[j]) & 1:
                            old_contrib += 1
                    
                    new_contrib = 0
                    valid = True
                    for j in range(m):
                        if j == i: continue
                        if family[j] == new_mask:
                            valid = False
                            break
                        if (comp[new_mask] >> family[j]) & 1:
                            new_contrib += 1
                    
                    if not valid:
                        continue
                    
                    delta = new_contrib - old_contrib
                    if delta > 0:
                        new_score = current_score + delta
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_mask = new_mask
                
                if best_new_mask != old_mask:
                    family[i] = best_new_mask
                    current_score = best_new_score
                    improved = True
        
        return family, current_score

    # Main Loop
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_family = None
    best_score = -1
    
    # Run multiple restarts with different initializations
    # 1. Greedy
    # 2. Random structured (e.g., subsets of a random base)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        if restarts == 1:
            # Deterministic greedy
            family = greedy_init()
        elif restarts % 2 == 0:
            # Structured: pick a random base set of size 6-8, take all its subsets, shuffle, pick 40
            base_size = random.randint(6, 8)
            base_elements = random.sample(range(n), base_size)
            base_mask = 0
            for e in base_elements:
                base_mask |= (1 << e)
            
            # Generate all subsets of base_mask
            subsets = []
            for i in range(1 << base_size):
                mask = 0
                for b in range(base_size):
                    if (i >> b) & 1:
                        mask |= (1 << base_elements[b])
                subsets.append(mask)
            
            random.shuffle(subsets)
            # Ensure distinctness (they are distinct by construction)
            # If we have more than 40, take 40. If less, fill randomly.
            if len(subsets) >= m:
                family = subsets[:m]
            else:
                family = subsets[:]
                seen = set(family)
                while len(family) < m:
                    new_f = random.randint(0, total_masks - 1)
                    if new_f not in seen:
                        family.append(new_f)
                        seen.add(new_f)
        else:
            # Random
            family = random.sample(range(total_masks), m)
        
        # Ensure distinctness
        if len(set(family)) < m:
            seen = set()
            new_family = []
            for f in family:
                if f not in seen:
                    new_family.append(f)
                    seen.add(f)
            while len(new_family) < m:
                new_f = random.randint(0, total_masks - 1)
                if new_f not in seen:
                    new_family.append(new_f)
                    seen.add(new_f)
            family = new_family
        
        # Refine with 2-Opt
        family, score = two_opt_search(family, deadline)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
            
        # If we've reached a high score, we might stop early?
        # Max possible pairs is 40*39/2 = 780.
        # Current best is around 2.2 * seed. Seed was probably low.
        # Let's just run until deadline.
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Convert best_family to list of lists of ints
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result

# Fallback if construct fails or times out unexpectedly?
# The task requires returning a list of 40 lists.
# If the loop above doesn't finish, we should return a valid result.
# But the deadline guard inside the loop should handle it.
# If best_family is None (e.g., immediate timeout), return a default valid family.
# However, the code above always sets best_family in the first iteration if it completes.
# To be safe:

def construct():
    try:
        return _construct_impl()
    except Exception:
        # Return a valid default: first 40 masks
        return [[b for b in range(12) if (mask >> b) & 1] for mask in range(40)]

# We need to move the logic to _construct_impl or just keep it in construct and ensure it's robust.
# Let's stick to one function to match the signature exactly.

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Pre-compute comparability
    comp = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                bits |= (1 << j)
        comp[i] = bits

    def count_pairs(family):
        c = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (comp[fi] >> fj) & 1:
                    c += 1
        return c

    def greedy_init():
        family = []
        family_set = set()
        family.append(0)
        family_set.add(0)
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            for s in range(total_masks):
                if s in family_set:
                    continue
                gain = 0
                for f in family:
                    if (comp[f] >> s) & 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        return family

    def two_opt_search(family, deadline):
        current_score = count_pairs(family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                old_mask = family[i]
                best_new_mask = old_mask
                best_new_score = current_score
                
                # Calculate old contribution
                old_contrib = 0
                for j in range(m):
                    if j == i: continue
                    if (comp[old_mask] >> family[j]) & 1:
                        old_contrib += 1
                
                for new_mask in range(total_masks):
                    if new_mask == old_mask:
                        continue
                    
                    # Check distinctness
                    valid = True
                    for j in range(m):
                        if j != i and family[j] == new_mask:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate new contribution
                    new_contrib = 0
                    for j in range(m):
                        if j == i: continue
                        if (comp[new_mask] >> family[j]) & 1:
                            new_contrib += 1
                    
                    delta = new_contrib - old_contrib
                    if delta > 0:
                        new_score = current_score + delta
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_mask = new_mask
                
                if best_new_mask != old_mask:
                    family[i] = best_new_mask
                    current_score = best_new_score
                    improved = True
        return family, current_score

    start_time = time.time()
    deadline = start_time + 85.0
    best_family = None
    best_score = -1
    restarts = 0
    
    while time.time() < deadline:
        restarts += 1
        if restarts == 1:
            family = greedy_init()
        elif restarts % 2 == 0:
            base_size = random.randint(6, 8)
            base_elements = random.sample(range(n), base_size)
            subsets = []
            for i in range(1 << base_size):
                mask = 0
                for b in range(base_size):
                    if (i >> b) & 1:
                        mask |= (1 << base_elements[b])
                subsets.append(mask)
            random.shuffle(subsets)
            if len(subsets) >= m:
                family = subsets[:m]
            else:
                family = subsets[:]
                seen = set(family)
                while len(family) < m:
                    new_f = random.randint(0, total_masks - 1)
                    if new_f not in seen:
                        family.append(new_f)
                        seen.add(new_f)
        else:
            family = random.sample(range(total_masks), m)
        
        if len(set(family)) < m:
            seen = set()
            new_family = []
            for f in family:
                if f not in seen:
                    new_family.append(f)
                    seen.add(f)
            while len(new_family) < m:
                new_f = random.randint(0, total_masks - 1)
                if new_f not in seen:
                    new_family.append(new_f)
                    seen.add(new_f)
            family = new_family
        
        family, score = two_opt_search(family, deadline)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
            
        if time.time() > deadline - 5:
            break

    if best_family is None:
        # Fallback
        best_family = list(range(m))

    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    return result
