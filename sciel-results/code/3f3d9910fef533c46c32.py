"""Rewritten program: simulated annealing on QUBO formulation for 40 comparable subsets of {0..11}."""
import time
import random
import sys
import numpy as np

N = 12
M = 40
NUM_MASKS = 1 << N  # 4096
DEADLINE = 85.0  # safety margin before 90s kill

def _precompute_comparability():
    """
    Precompute for each mask m:
      - comp_mask[m]: a bitvector (Python int) where bit i is set iff mask i is comparable with m
        (i.e., i is a strict subset or strict superset of m).
    """
    comp_mask = [0] * NUM_MASKS
    
    for m in range(NUM_MASKS):
        # Strict subsets of m
        sub = m
        while sub:
            if sub != m:
                comp_mask[m] |= (1 << sub)
            sub = (sub - 1) & m
        
        # Strict supersets of m
        complement = (~m) & (NUM_MASKS - 1)
        extra = complement
        while extra:
            s = m | extra
            if s != m:
                comp_mask[m] |= (1 << s)
            extra = (extra - 1) & complement
    
    return comp_mask

def _count_comparable(family):
    """Count unordered pairs where one is strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        mi = family[i]
        for j in range(i + 1, n):
            mj = family[j]
            if (mi & mj) == mi and mi != mj:
                count += 1
            elif (mi & mj) == mj and mi != mj:
                count += 1
    return count

def _greedy_init(seed, comp_mask):
    """Greedy construction to get a good starting family."""
    rng = random.Random(seed)
    
    in_family = [False] * NUM_MASKS
    family = []
    scores = [0] * NUM_MASKS
    
    # Pick first element randomly, preferring intermediate sizes
    first = rng.choice(range(NUM_MASKS))
    family.append(first)
    in_family[first] = True
    
    # Update scores
    cf = comp_mask[first]
    for y in range(NUM_MASKS):
        if y != first and ((cf >> y) & 1):
            scores[y] += 1
    
    for _ in range(M - 1):
        best_mask = -1
        best_score = -1
        candidates = []
        for y in range(NUM_MASKS):
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_mask = y
                    candidates = [y]
                elif scores[y] == best_score:
                    candidates.append(y)
        
        if best_mask == -1:
            break
        
        chosen = rng.choice(candidates)
        family.append(chosen)
        in_family[chosen] = True
        
        cc = comp_mask[chosen]
        for y in range(NUM_MASKS):
            if not in_family[y] and ((cc >> y) & 1):
                scores[y] += 1
    
    # Fill if needed
    if len(family) < M:
        remaining = [m for m in range(NUM_MASKS) if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])
    
    return family

def _local_search(family, comp_mask, deadline, rng):
    """
    Local search: try swapping one element for another to improve the count.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    
    # Precompute scores for all masks not in family
    # scores[m] = number of family members comparable with m
    scores = {}
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        cm = comp_mask[m]
        cnt = 0
        for f in family:
            if (cm >> f) & 1:
                cnt += 1
        scores[m] = cnt
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            
            # Old comparable count for removed with other family members
            old_comp = 0
            cr = comp_mask[removed]
            for j in range(len(family)):
                if j != i and ((cr >> family[j]) & 1):
                    old_comp += 1
            
            # Find best candidate to add
            best_gain = -1
            best_add = -1
            for m, sc in scores.items():
                is_comp = ((comp_mask[m] >> removed) & 1)
                gain = sc - (1 if is_comp else 0) - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m
            
            if best_gain > 0 and best_add != -1:
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)
                
                # Update scores
                ca = comp_mask[best_add]
                for m in list(scores.keys()):
                    if m == removed or m == best_add:
                        continue
                    if (cr >> m) & 1:
                        scores[m] -= 1
                    if (ca >> m) & 1:
                        scores[m] += 1
                
                if best_add in scores:
                    del scores[best_add]
                
                current_count += best_gain
                improved = True
    
    return family, current_count

def _simulated_annealing(family, comp_mask, deadline, rng):
    """
    Simulated annealing: treat the family as a binary state vector over 4096 masks.
    Moves: swap one mask in family with one not in family.
    Objective: maximize number of comparable pairs.
    """
    family_set = set(family)
    current_count = _count_comparable(family)
    best_family = list(family)
    best_count = current_count
    
    T = 50.0  # initial temperature
    cooling = 0.995
    
    # Precompute comp_mask for all masks (already done)
    
    while time.time() < deadline and T > 0.1:
        T *= cooling
        
        # Pick a random element to remove
        i = rng.randint(0, len(family) - 1)
        removed = family[i]
        cr = comp_mask[removed]
        
        # Old comparable count for removed
        old_comp = 0
        for j in range(len(family)):
            if j != i and ((cr >> family[j]) & 1):
                old_comp += 1
        
        # Pick a random candidate to add (not in family)
        # Sample from a subset for efficiency
        candidates = []
        for m in range(0, NUM_MASKS, rng.randint(1, 4)):
            if m not in family_set:
                candidates.append(m)
            if len(candidates) > 100:
                break
        
        if not candidates:
            candidates = [m for m in range(NUM_MASKS) if m not in family_set]
        
        if not candidates:
            continue
        
        added = rng.choice(candidates)
        ca = comp_mask[added]
        
        # New comparable count for added
        new_comp = 0
        is_comp_with_removed = ((ca >> removed) & 1)
        for j in range(len(family)):
            if j != i and ((ca >> family[j]) & 1):
                new_comp += 1
        
        # Net change in total comparable pairs
        # Remove old_comp pairs involving removed
        # Add new_comp pairs involving added (but subtract 1 if added is comparable with removed,
        #   because that pair is already counted in old_comp? No, old_comp counts pairs between
        #   removed and other family members. new_comp counts pairs between added and other family
        #   members (excluding removed). The pair (removed, added) is not in either.)
        # Actually:
        #   Old total includes old_comp pairs (removed, family[j] for j != i)
        #   New total includes new_comp pairs (added, family[j] for j != i)
        #   The pair (removed, added) is not present in either old or new family (since we're swapping)
        #   So delta = new_comp - old_comp
        delta = new_comp - old_comp
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Accept
            family[i] = added
            family_set.remove(removed)
            family_set.add(added)
            current_count += delta
            
            if current_count > best_count:
                best_count = current_count
                best_family = list(family)
    
    return best_family, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute comparability
    comp_mask = _precompute_comparability()
    
    best_count = 0
    best_family = None
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 40.0:
            break
        family = _greedy_init(i, comp_mask)
        count = _count_comparable(family)
        if count > best_count:
            best_count = count
            best_family = family
    
    # Phase 2: Local search on best family
    if best_family is not None:
        rng = random.Random(42)
        polish_deadline = deadline - 30.0
        best_family, best_count = _local_search(best_family, comp_mask, polish_deadline, rng)
    
    # Phase 3: Simulated annealing for further improvement
    if best_family is not None:
        rng = random.Random(123)
        sa_deadline = deadline - 3.0
        best_family, best_count = _simulated_annealing(best_family, comp_mask, sa_deadline, rng)
    
    # Fallback
    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in __import__('itertools').combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = [mask for mask in all_sets[:M]]
        # Convert to masks
        best_family = []
        for s in all_sets[:M]:
            mask = 0
            for v in s:
                mask |= (1 << v)
            best_family.append(mask)
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
