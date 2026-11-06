import time
from itertools import combinations

def _pgl2_11_generators():
    """
    Generate the PGL(2,11) group acting on 12 points (0..10, 11=infinity).
    """
    p = 11
    # Represent elements as (a,b,c,d) mod 11
    # x -> (a*x+b)/(c*x+d)
    
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
    deadline = start_time + 110 # Safety margin
    
    group, apply_transform = _pgl2_11_generators()
    
    # We need to find 132 blocks.
    # We will union orbits of base blocks.
    
    # Pre-compute 5-subsets for all possible blocks? No, too many.
    # We will check conflicts on the fly.
    
    all_blocks = []
    covered_5sets = set()
    seen_blocks = set()
    
    # Generate candidate base blocks
    # We can just iterate through all 6-subsets
    # But we want to use the group structure.
    # We will pick random base blocks, generate orbit, add valid ones.
    
    import random
    random.seed(42)
    
    all_6sets = list(combinations(range(12), 6))
    random.shuffle(all_6sets)
    
    for base_tuple in all_6sets:
        if time.time() > deadline:
            break
            
        if len(all_blocks) >= 132:
            break
            
        base = frozenset(base_tuple)
        
        # Generate orbit
        orbit = set()
        for g in group:
            block = frozenset(apply_transform(g, x) for x in base)
            orbit.add(block)
        
        # Add valid blocks from orbit
        for block in orbit:
            if len(all_blocks) >= 132:
                break
                
            if block in seen_blocks:
                continue
                
            # Check 5-subsets
            valid = True
            block_5sets = list(combinations(sorted(block), 5))
            for five in block_5sets:
                if five in covered_5sets:
                    valid = False
                    break
            
            if valid:
                all_blocks.append(list(block))
                seen_blocks.add(block)
                covered_5sets.update(block_5sets)
                
    if len(all_blocks) == 132:
        return all_blocks
        
    # Fallback: Greedy
    # If we didn't get 132, try a pure greedy on the remaining space?
    # Or just return what we have.
    # Given the group structure, we should get 132.
    
    return all_blocks
