import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Helper: check if a is strict subset of b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    # Count comparable pairs in a family (list of masks)
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            for j in range(i + 1, k):
                b = family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (b & a) == b and b != a:
                    count += 1
        return count
    
    # Generate initial solution: use a greedy chain-based approach
    # Create a main chain: 0, {0}, {0,1}, {0,1,2}, ..., {0..11}
    # Then add sets that are comparable with many existing sets
    
    # Initial: pick 40 sets using a heuristic
    # Use sets sorted by size, and prefer sets that are subsets of/supersets of existing ones
    
    random.seed(42)
    
    # Start with a chain of all sizes 0-12 (13 sets)
    chain = []
    for size in range(13):
        mask = (1 << size) - 1  # {0, 1, ..., size-1}
        chain.append(mask)
    
    # Now add 27 more sets greedily
    family = list(chain)
    
    # Candidate pools: generate random masks
    candidates = list(range(total_masks))
    random.shuffle(candidates)
    
    # Remove masks already in family
    family_set = set(family)
    candidates = [c for c in candidates if c not in family_set]
    
    # Greedily add masks that maximize comparable pairs
    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        for c in candidates[:200]:  # Limit search for speed
            # Compute gain if we add c
            gain = 0
            for f in family:
                if (c & f) == c and c != f:
                    gain += 1
                elif (f & c) == f and f != c:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            candidates.remove(best_mask)
    
    # Local search: try to replace one mask with another to improve
    best_score = count_comparable(family)
    best_family = list(family)
    
    # Precompute for each mask in family, which other masks it's comparable with
    # Use a more efficient local search
    
    # Try multiple random restarts if time allows
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        # Create a perturbed version of best_family
        family = list(best_family)
        
        # Randomly remove 1-3 sets and replace with new ones
        num_replace = random.randint(1, 3)
        to_replace = random.sample(range(m), num_replace)
        family_set = set(family)
        
        for idx in to_replace:
            # Find a good replacement
            best_mask = None
            best_gain = -1
            # Sample candidates
            for _ in range(100):
                c = random.randint(0, total_masks - 1)
                if c in family_set:
                    continue
                # Compute gain
                gain = 0
                for f in family:
                    if f == family[idx]:
                        continue
                    if (c & f) == c and c != f:
                        gain += 1
                    elif (f & c) == f and f != c:
                        gain += 1
                # Subtract loss from removing family[idx]
                loss = 0
                for f in family:
                    if f == family[idx]:
                        continue
                    if (family[idx] & f) == family[idx] and family[idx] != f:
                        loss += 1
                    elif (f & family[idx]) == f and f != family[idx]:
                        loss += 1
                net_gain = gain - loss
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c
            if best_mask is not None:
                family[idx] = best_mask
                family_set.add(best_mask)
        
        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Final polish: try single replacements
    while time.time() < deadline - 2.0:
        improved = False
        family_set = set(best_family)
        for i in range(m):
            # Try to replace best_family[i] with a better mask
            best_mask = None
            best_gain = -1
            # Sample candidates
            for _ in range(50):
                c = random.randint(0, total_masks - 1)
                if c in family_set:
                    continue
                # Compute gain
                gain = 0
                for j in range(m):
                    if j == i:
                        continue
                    f = best_family[j]
                    if (c & f) == c and c != f:
                        gain += 1
                    elif (f & c) == f and f != c:
                        gain += 1
                # Subtract loss from removing best_family[i]
                loss = 0
                for j in range(m):
                    if j == i:
                        continue
                    f = best_family[j]
                    if (best_family[i] & f) == best_family[i] and best_family[i] != f:
                        loss += 1
                    elif (f & best_family[i]) == f and f != best_family[i]:
                        loss += 1
                net_gain = gain - loss
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c
            if best_mask is not None and best_gain > 0:
                family_set.remove(best_family[i])
                best_family[i] = best_mask
                family_set.add(best_mask)
                improved = True
                break
        if not improved:
            break
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    # Ensure types are correct: lists of ints
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result
