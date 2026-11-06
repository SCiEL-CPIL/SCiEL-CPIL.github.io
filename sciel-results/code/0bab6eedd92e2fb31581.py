"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _pgl2_11_generators():
    """
    Generate the PGL(2,11) group acting on 12 points (0..10, 11=infinity).
    """
    p = 11
    def apply_transform(el, x):
        a, b, c, d = el
        if x == 11:
            if c == 0:
                return 11
            else:
                return (a * pow(c, p-2, p)) % p
        else:
            denom = (c * x + d) % p
            if denom == 0:
                return 11
            else:
                num = (a * x + b) % p
                return (num * pow(denom, p-2, p)) % p
    
    def compose(el1, el2):
        a1, b1, c1, d1 = el1
        a2, b2, c2, d2 = el2
        a = (a1 * a2 + b1 * c2) % p
        b = (a1 * b2 + b1 * d2) % p
        c = (c1 * a2 + d1 * c2) % p
        d = (c1 * b2 + d1 * d2) % p
        return (a, b, c, d)
    
    identity = (1, 0, 0, 1)
    T = (1, 1, 0, 1)
    M = (1, 0, 0, 10)
    S = (2, 0, 0, 1)
    
    group = {identity}
    queue = [identity]
    while queue:
        g = queue.pop(0)
        for gen in [T, M, S]:
            new_g = compose(gen, g)
            if new_g not in group:
                group.add(new_g)
                queue.append(new_g)
    
    return list(group), apply_transform

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    group, apply_transform = _pgl2_11_generators()
    
    # Generate all 6-subsets
    all_blocks = list(combinations(range(12), 6))
    
    # Map each block to its orbit
    # To avoid redundant work, we can process blocks one by one and skip if already in a known orbit
    processed = set()
    orbits = []
    
    for block in all_blocks:
        if block in processed:
            continue
        
        # Generate orbit of this block
        orbit = set()
        for g in group:
            new_block = tuple(sorted(apply_transform(g, x) for x in block))
            orbit.add(new_block)
        
        # Add all blocks in this orbit to processed
        for b in orbit:
            processed.add(b)
        
        orbits.append(frozenset(orbit))
        
        if time.time() > deadline - 10:
            break

    # Filter orbits that are internally valid (no 5-subset repeats within the orbit)
    valid_orbits = []
    for orbit in orbits:
        covered_5sets = set()
        is_valid = True
        for block in orbit:
            for five in combinations(block, 5):
                if five in covered_5sets:
                    is_valid = False
                    break
                covered_5sets.add(five)
            if not is_valid:
                break
        if is_valid:
            valid_orbits.append(orbit)
    
    # Now we need to select a subset of valid_orbits such that no 5-subset is shared
    # Between any two selected orbits.
    # Precompute the set of 5-subsets for each valid orbit
    orbit_5sets = []
    for orbit in valid_orbits:
        s = set()
        for block in orbit:
            for five in combinations(block, 5):
                s.add(five)
        orbit_5sets.append(s)
    
    # Check compatibility between orbits
    # If number of valid orbits is small, we can do a simple search
    # The Witt design is a union of orbits.
    # Let's try to find the combination that maximizes blocks.
    
    best_blocks = []
    best_size = 0
    
    # If the number of valid orbits is small (e.g., < 20), we can use backtracking
    # Or even just check all subsets if very small.
    # Given the symmetry, likely there are very few valid orbits (e.g., 1, 2, or 3).
    
    n_orbits = len(valid_orbits)
    
    if n_orbits == 0:
        # Fallback to greedy
        return _greedy_fallback()
    
    # Try all subsets of valid orbits?
    # If n_orbits is small, 2^n is fine.
    # If n_orbits is large, we need a smarter search.
    # Let's assume n_orbits is small.
    
    for mask in range(1, 1 << n_orbits):
        if time.time() > deadline - 5:
            break
        
        selected = []
        global_covered = set()
        valid = True
        total_blocks = 0
        
        for i in range(n_orbits):
            if mask & (1 << i):
                if not global_covered.isdisjoint(orbit_5sets[i]):
                    valid = False
                    break
                global_covered.update(orbit_5sets[i])
                total_blocks += len(valid_orbits[i])
        
        if valid and total_blocks > best_size:
            best_size = total_blocks
            # Reconstruct blocks
            best_blocks = []
            for i in range(n_orbits):
                if mask & (1 << i):
                    best_blocks.extend(valid_orbits[i])
            
            if best_size == 132:
                break
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    
    return _greedy_fallback()

def _greedy_fallback():
    import random
    best_blocks = []
    for attempt in range(10):
        random.seed(attempt)
        covered = set()
        blocks = []
        all_blocks = list(combinations(range(12), 6))
        random.shuffle(all_blocks)
        for b in all_blocks:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
    return best_blocks
# EVOLVE-BLOCK-END
