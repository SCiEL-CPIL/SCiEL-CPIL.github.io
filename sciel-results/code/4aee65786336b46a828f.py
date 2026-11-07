import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    m = 40
    all_masks = list(range(1 << n))  # 0 to 4095
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def is_subset(a, b):
        # a is subset of b (including equal)
        return (a & b) == a
    
    def is_strict_subset(a, b):
        # a is strict subset of b
        return (a & b) == a and a != b
    
    def count_comparable(family_masks):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        k = len(family_masks)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = family_masks[i], family_masks[j]
                if is_strict_subset(a, b) or is_strict_subset(b, a):
                    count += 1
        return count
    
    def greedy_build(seed):
        """Build a family greedily, starting from a chain-based init."""
        rng = random.Random(seed)
        
        # Start with a long chain: empty set, then add one element at a time
        # Use a random permutation to create the chain
        perm = list(range(n))
        rng.shuffle(perm)
        
        chain = [0]  # empty set
        mask = 0
        for v in perm:
            mask |= (1 << v)
            chain.append(mask)
        # chain has 13 sets: sizes 0,1,2,...,12
        
        family = list(chain)  # 13 sets
        used = set(family)
        
        # Greedily add remaining sets
        remaining = [s for s in all_masks if s not in used]
        rng.shuffle(remaining)
        
        while len(family) < m:
            best_gain = -1
            best_set = None
            for s in remaining:
                if s in used:
                    continue
                gain = 0
                for t in family:
                    if is_strict_subset(s, t) or is_strict_subset(t, s):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
                if time.time() > deadline:
                    break
            if best_set is None:
                # Fallback: just pick any unused
                for s in remaining:
                    if s not in used:
                        best_set = s
                        break
            if best_set is None:
                break
            family.append(best_set)
            used.add(best_set)
            if time.time() > deadline:
                break
        
        return family
    
    def local_search(family, max_iter=500):
        """Improve family via swap-based local search."""
        family = list(family)
        current_score = count_comparable(family)
        used = set(family)
        
        # Precompute all masks not in family
        all_set = set(all_masks)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping each set in family with a set not in family
            for i in range(len(family)):
                if time.time() > deadline:
                    break
                old_set = family[i]
                # Try a few candidate replacements
                # Pick candidates that might be comparable with many existing sets
                candidates = []
                # Sample some candidates
                for _ in range(20):
                    cand = random.choice(all_masks)
                    if cand not in used:
                        candidates.append(cand)
                
                for cand in candidates:
                    if cand in used:
                        continue
                    # Compute new score
                    new_score = 0
                    for j in range(len(family)):
                        if j == i:
                            continue
                        a, b = cand, family[j]
                        if is_strict_subset(a, b) or is_strict_subset(b, a):
                            new_score += 1
                    # Subtract pairs involving old_set
                    old_pairs = 0
                    for j in range(len(family)):
                        if j == i:
                            continue
                        a, b = old_set, family[j]
                        if is_strict_subset(a, b) or is_strict_subset(b, a):
                            old_pairs += 1
                    delta = new_score - old_pairs
                    if delta > 0:
                        family[i] = cand
                        used.remove(old_set)
                        used.add(cand)
                        current_score += delta
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return family, current_score
    
    best_family = None
    best_score = -1
    
    # Try multiple greedy builds with different seeds
    num_restarts = 0
    for seed in range(50):
        if time.time() > deadline:
            break
        
        family = greedy_build(seed)
        if len(family) < m:
            continue
        
        # Local search
        family, score = local_search(family, max_iter=200)
        
        if score > best_score:
            best_score = score
            best_family = family
            num_restarts += 1
        
        if time.time() > deadline:
            break
    
    if best_family is None:
        # Fallback: simple greedy
        best_family = greedy_build(0)
        if len(best_family) < m:
            # Pad with arbitrary unused sets
            used = set(best_family)
            for s in all_masks:
                if s not in used:
                    best_family.append(s)
                    used.add(s)
                if len(best_family) == m:
                    break
    
    # Convert masks to lists of distinct integers
    result = []
    for mask in best_family[:m]:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure exactly 40 distinct lists
    assert len(result) == m
    assert len(set(tuple(x) for x in result)) == m
    
    return result
