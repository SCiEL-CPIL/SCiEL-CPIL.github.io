"""Direct construction of maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

# LCM of 1..22
_LCM = 232792560

# Large prime for modular filtering
_P = 10**9 + 7
# Precompute modular weights: LCM // x mod P
_MOD_WEIGHTS = {x: (_LCM // x) % _P for x in range(1, 23)}

def _verify_mod(subset):
    """Fast modular check: all 2^k subset sums mod P must be distinct.
    Necessary but not sufficient for exact distinctness."""
    k = len(subset)
    if k == 0:
        return True
    if k >= 30:  # 2^k would be too large even for modular check
        return False
    mod_weights = [_MOD_WEIGHTS[x] for x in subset]
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        sums[mask] = (sums[mask ^ lb] + mod_weights[idx]) % _P
    return len(set(sums)) == n

def _verify(subset):
    """Check if all 2^k subset sums of {LCM//x for x in subset} are distinct."""
    k = len(subset)
    if k == 0:
        return True
    # Fast modular filter first
    if not _verify_mod(subset):
        return False
    scaled = [_LCM // x for x in subset]
    # Generate all subset sums
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + scaled[idx]
    return len(set(sums)) == n

def _greedy_build(order, deadline):
    """Greedy build from a given order with modular pre-filter."""
    current = []
    for x in order:
        if time.time() > deadline:
            break
        candidate = current + [x]
        # Fast modular check first
        if _verify_mod(candidate):
            # Then exact check
            if _verify(candidate):
                current = candidate
    return current

def _local_search(best, universe, deadline):
    """Local search: try swaps and additions to improve cardinality."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(best)
        remaining = [x for x in universe if x not in in_set]
        
        # Try adding elements
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = best + [x]
            if _verify_mod(candidate) and _verify(candidate):
                best = candidate
                improved = True
                break
        
        # Try single swaps
        if not improved:
            in_set = set(best)
            remaining = [x for x in universe if x not in in_set]
            for i, a in enumerate(best):
                if time.time() > deadline:
                    break
                for b in remaining:
                    candidate = best[:]
                    candidate[i] = b
                    if _verify_mod(candidate) and _verify(candidate):
                        # Check if this swap allows adding more
                        new_in = set(candidate)
                        new_remaining = [x for x in universe if x not in new_in]
                        for c in new_remaining:
                            if time.time() > deadline:
                                break
                            if _verify_mod(candidate + [c]) and _verify(candidate + [c]):
                                best = candidate + [c]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Phase 1: Standard greedy (descending scaled value = ascending x)
    order1 = sorted(universe, key=lambda x: -(_LCM // x))
    best = _greedy_build(order1, deadline)
    
    # Phase 2: Local search
    best = _local_search(best, universe, deadline)
    
    # Phase 3: Multiple random restarts with different orderings
    random.seed(42)
    best_size = len(best)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Create a randomized order with some structure
        # Mix of random and heuristic orderings
        if restart % 3 == 0:
            # Fully random
            order = universe[:]
            random.shuffle(order)
        elif restart % 3 == 1:
            # Random but bias toward smaller x (larger scaled values)
            order = sorted(universe, key=lambda x: random.random() * 5 - _LCM // x / (_LCM / 10))
        else:
            # Random but bias toward larger x (smaller scaled values)
            order = sorted(universe, key=lambda x: random.random() * 5 + _LCM // x / (_LCM / 10))
        
        candidate = _greedy_build(order, deadline)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
            # Local search on the new best
            best = _local_search(best, universe, deadline)
            best_size = len(best)
    
    # Phase 4: Final local search with more aggressive swaps
    if time.time() < deadline - 5:
        best = _local_search(best, universe, deadline)
    
    # Phase 5: Try to add any remaining elements one more time
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 2:
            break
        candidate = best + [x]
        if _verify_mod(candidate) and _verify(candidate):
            best = candidate
    
    return best
