"""Direct construction of maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _verify(subset):
    """Check if all 2^k subset sums of {LCM//x for x in subset} are distinct."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x for x in subset]
    # Generate all subset sums incrementally
    sums = {0}
    for s in scaled:
        new_sums = {old + s for old in sums}
        if len(new_sums) != len(sums):
            return False
        sums |= new_sums
    return True

def _verify_add(subset, x):
    """Check if adding x to a dissociated subset keeps it dissociated. O(2^k) not O(2^(k+1))."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x] + [_LCM // y for y in subset]
    # We need: all 2^(k+1) sums distinct
    # Equivalent: no sum from {0, s} equals any sum from subset-sums, 
    # and no s + a == b where a,b are subset sums
    # Faster: generate all subset sums of subset, check s not in them, and s+s_i != s_j
    old_sums = {0}
    for y in subset:
        v = _LCM // y
        new_sums = {old + v for old in old_sums}
        old_sums |= new_sums
    s = _LCM // x
    # Check s not in old_sums
    if s in old_sums:
        return False
    # Check s + a != b for all a, b in old_sums
    # i.e., s != b - a for any a, b in old_sums
    # i.e., s not in {b - a : a, b in old_sums}
    # This is O(2^(2k)) which is too slow for large k
    # Alternative: just do full verify but with early termination
    # Actually for k <= 12, 2^k <= 4096, so 2^(2k) is too much
    # Let's just do the full check efficiently
    all_sums = set(old_sums)
    for a in old_sums:
        b = a + s
        if b in all_sums and b != a:  # b != a means s != 0, which is always true
            # Wait, we need to check if s+a collides with any existing sum
            # s+a is a new sum. It collides if s+a is in old_sums (already checked via s in old_sums? No)
            # Actually: the new sums are {a + s : a in old_sums}. 
            # Collision if (a1+s) == (a2+s) for a1!=a2 (impossible since a1+a2 distinct)
            # or (a1+s) == a2 for some a1, a2 in old_sums
            # So check: s + a1 not in old_sums for all a1 in old_sums
            pass
    # Simpler: just check all new sums against old sums
    for a in old_sums:
        if (a + s) in old_sums:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Try multiple strategies to find the largest dissociated set
    
    best = []
    
    # Strategy 1: Greedy with multiple orderings
    orderings = [
        sorted(universe, key=lambda x: -(_LCM // x)),  # 1,2,3,...,22
        sorted(universe, key=lambda x: _LCM // x),     # 22,21,...,1
    ]
    
    # Add a few random orderings
    rng = random.Random(42)
    for _ in range(5):
        perm = universe[:]
        rng.shuffle(perm)
        orderings.append(perm)
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        current = []
        for x in order:
            if time.time() > deadline - 10:
                break
            if _verify_add(current, x):
                current.append(x)
        if len(current) > len(best):
            best = list(current)
    
    # Strategy 2: Local search - try to add remaining elements
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    
    # Try adding each remaining element
    for x in remaining:
        if time.time() > deadline - 10:
            break
        if _verify_add(best, x):
            best.append(x)
            in_set.add(x)
            remaining.remove(x)
    
    # Strategy 3: Swap search - try replacing one element with another
    if time.time() < deadline - 20 and len(best) < 22:
        in_set = set(best)
        not_in_set = [x for x in universe if x not in in_set]
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i, a in enumerate(best):
                if time.time() > deadline - 10:
                    break
                for b in not_in_set:
                    if time.time() > deadline - 10:
                        break
                    candidate = best[:]
                    candidate[i] = b
                    if _verify(candidate):
                        # Check if this allows adding more
                        new_in = set(candidate)
                        new_not = [x for x in universe if x not in new_in]
                        extra = 0
                        for c in new_not:
                            if time.time() > deadline - 10:
                                break
                            if _verify_add(candidate, c):
                                candidate.append(c)
                                new_in.add(c)
                                extra += 1
                        if extra > 0 or len(candidate) > len(best):
                            best = candidate
                            in_set = set(best)
                            not_in_set = [x for x in universe if x not in in_set]
                            improved = True
                            break
                if improved:
                    break
    
    # Strategy 4: Final attempt to add any remaining
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 5:
            break
        if _verify_add(best, x):
            best.append(x)
    
    return best
