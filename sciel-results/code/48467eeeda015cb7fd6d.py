import time
import random
from math import gcd

def construct():
    deadline = time.time() + 82  # Safety margin (evaluator kills at 90s)
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    mask_all = (1 << (n + 1)) - 1  # bits 1..128 set
    
    # Precompute bad pairs using bitmasks for speed
    # conflict_mask[x] is a bitmask where bit y is set if x conflicts with y
    # But we need triple conflicts: x can't be added if both a and b are present
    # So we store for each x: list of (a, b) pairs, but also precompute a "conflict graph"
    
    # Better approach: for each x, store a set of pairs, but use bitmask for membership
    # conflict_pairs[x] = list of (a, b) where {x,a,b} is bad
    # can_add(x, mask) checks: for each (a,b) in conflict_pairs[x], not (mask has a AND mask has b)
    
    # Even faster: precompute for each x a list of masks where each mask represents
    # "if both bits in this mask are set, x cannot be added"
    # But that's the same as pairs. Let's just use the pair list with bitmask checks.
    
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) != ab:
                    continue
                if _lcm(b, c) == ab:
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
    
    # Convert to sorted lists for deterministic iteration and better cache behavior
    bad_pairs_list = [sorted(s) for s in bad_pairs_for]
    
    # Precompute degree (number of conflicting pairs)
    degrees = [len(bad_pairs_list[k]) for k in range(n + 1)]
    
    def can_add(x, mask):
        """Check if x can be added given current set represented as bitmask."""
        for (a, b) in bad_pairs_list[x]:
            if (mask & (1 << a)) and (mask & (1 << b)):
                return False
        return True
    
    def greedy(order, mask=0):
        """Greedy construction in given order, starting from mask."""
        for x in order:
            if not (mask & (1 << x)) and can_add(x, mask):
                mask |= (1 << x)
        return mask
    
    def mask_to_set(mask):
        return {i for i in range(1, n + 1) if mask & (1 << i)}
    
    def local_search(mask):
        """Iterative local search with multiple move types."""
        max_rounds = 30
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # 1. Try to add any single element
            for x in range(1, n + 1):
                if not (mask & (1 << x)) and can_add(x, mask):
                    mask |= (1 << x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 1-for-1 swaps
            found = False
            for x in range(1, n + 1):
                if not (mask & (1 << x)):
                    continue
                mask_trial = mask & ~(1 << x)
                for y in range(1, n + 1):
                    if (mask & (1 << y)):
                        continue
                    if can_add(y, mask_trial):
                        mask_trial |= (1 << y)
                        mask = mask_trial
                        improved = True
                        found = True
                        break
                if found:
                    break
            
            if improved:
                continue
            
            # 3. Try 1-for-2 swaps (remove x, add y and z)
            found = False
            for x in range(1, n + 1):
                if not (mask & (1 << x)):
                    continue
                mask_trial = mask & ~(1 << x)
                added = 0
                for y in range(1, n + 1):
                    if (mask & (1 << y)) or (y == x):
                        continue
                    if can_add(y, mask_trial):
                        mask_trial |= (1 << y)
                        added += 1
                        if added == 1:
                            for z in range(y + 1, n + 1):
                                if (mask & (1 << z)) or (z == x):
                                    continue
                                if can_add(z, mask_trial):
                                    mask_trial |= (1 << z)
                                    added += 1
                                    break
                        if added >= 2:
                            break
                if added >= 2:
                    mask = mask_trial
                    improved = True
                    found = True
                    break
            
            if improved:
                continue
            
            # 4. Try 2-for-3 swaps (remove x,y, add up to 3)
            found = False
            elements = [i for i in range(1, n + 1) if mask & (1 << i)]
            random.shuffle(elements)
            max_pairs = min(len(elements) * (len(elements) - 1) // 2, 300)
            pairs_tried = 0
            for i in range(len(elements)):
                if pairs_tried >= max_pairs or time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(elements)):
                    if pairs_tried >= max_pairs or time.time() > deadline - 5:
                        break
                    pairs_tried += 1
                    x, y = elements[i], elements[j]
                    mask_trial = mask & ~(1 << x) & ~(1 << y)
                    added = 0
                    for z in range(1, n + 1):
                        if (mask & (1 << z)) or (z == x) or (z == y):
                            continue
                        if can_add(z, mask_trial):
                            mask_trial |= (1 << z)
                            added += 1
                            if added >= 3:
                                break
                    if added >= 3:
                        mask = mask_trial
                        improved = True
                        found = True
                        break
                if found:
                    break
            
            if not improved:
                break
        
        return mask
    
    def final_polish(mask):
        """Exhaustive small-scale polish: try all 1-for-1 and all 2-for-2 swaps."""
        if time.time() > deadline - 3:
            return mask
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1-for-1: exhaustive
            for x in range(1, n + 1):
                if not (mask & (1 << x)):
                    continue
                mask_trial = mask & ~(1 << x)
                for y in range(1, n + 1):
                    if (mask & (1 << y)):
                        continue
                    if can_add(y, mask_trial):
                        mask_trial |= (1 << y)
                        if mask_trial != mask:
                            mask = mask_trial
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
            
            # 2-for-2: exhaustive (sample if too many)
            elements = [i for i in range(1, n + 1) if mask & (1 << i)]
            if len(elements) > 40:
                # Sample pairs
                random.shuffle(elements)
                elements = elements[:40]
            
            found = False
            for i in range(len(elements)):
                if time.time() > deadline - 1:
                    break
                for j in range(i + 1, len(elements)):
                    x, y = elements[i], elements[j]
                    mask_trial = mask & ~(1 << x) & ~(1 << y)
                    added = 0
                    for z in range(1, n + 1):
                        if (mask & (1 << z)) or (z == x) or (z == y):
                            continue
                        if can_add(z, mask_trial):
                            mask_trial |= (1 << z)
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2:
                        mask = mask_trial
                        improved = True
                        found = True
                        break
                if found:
                    break
        
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Generate diverse orderings
    def make_orderings():
        orderings = []
        
        # 1. Random
        for _ in range(50):
            o = list(range(1, n + 1))
            random.shuffle(o)
            orderings.append(o)
        
        # 2. Degree-sorted ascending with noise
        for _ in range(30):
            o = list(range(1, n + 1))
            o.sort(key=lambda x: degrees[x] + random.random() * 2)
            orderings.append(o)
        
        # 3. Degree-sorted descending with noise
        for _ in range(30):
            o = list(range(1, n + 1))
            o.sort(key=lambda x: -degrees[x] + random.random() * 2)
            orderings.append(o)
        
        # 4. Conflict-aware: prioritize elements that don't conflict much with early choices
        for _ in range(20):
            o = list(range(1, n + 1))
            # Weight by degree but also by how "central" they are
            o.sort(key=lambda x: degrees[x] * random.random() + random.random())
            orderings.append(o)
        
        # 5. Reverse natural with shuffle
        for _ in range(10):
            o = list(range(n, 0, -1))
            random.shuffle(o)
            orderings.append(o)
        
        # 6. Prime-biased (primes have fewer divisors, potentially fewer conflicts)
        def is_prime(k):
            if k < 2:
                return False
            for p in range(2, int(k**0.5) + 1):
                if k % p == 0:
                    return False
            return True
        primes = [k for k in range(1, n + 1) if is_prime(k)]
        non_primes = [k for k in range(1, n + 1) if not is_prime(k)]
        for _ in range(10):
            random.shuffle(primes)
            random.shuffle(non_primes)
            o = primes + non_primes
            orderings.append(o)
        
        return orderings
    
    orderings = make_orderings()
    random.shuffle(orderings)
    
    for i, order in enumerate(orderings):
        if time.time() > deadline - 20:
            break
        
        mask = greedy(order)
        mask = local_search(mask)
        
        if len(mask_to_set(mask)) > best_size:
            best_size = len(mask_to_set(mask))
            best_mask = mask
    
    # Final polish on best
    best_mask = final_polish(best_mask)
    
    return mask_to_set(best_mask)
