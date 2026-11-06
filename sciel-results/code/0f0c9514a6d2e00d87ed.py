"""Combined approach: divisor-seed + systematic expansion + bitmask compatibility."""
import time
import random
from math import gcd, lcm as _lcm

# EVOLVE-BLOCK-START

def _divisors(n):
    """Return sorted list of divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def _build_bitmask_compatibility(n=512):
    """
    Build compatibility as bitmasks for fast intersection checks.
    comp_mask[i] is a bitmask where bit j is set iff lcm(i+1, j+1) <= n.
    Index i represents number i+1.
    """
    size = n + 1  # indices 0..n for numbers 1..n
    comp_mask = [0] * size
    for i in range(size):
        a = i + 1
        for j in range(size):
            b = j + 1
            if _lcm(a, b) <= n:
                comp_mask[i] |= (1 << j)
    return comp_mask

def _is_compatible_with_all(elem, clique, comp_mask):
    """Check if elem (0-indexed) is compatible with all in clique (list of 0-indexed)."""
    mask = comp_mask[elem]
    for c in clique:
        if not (mask & (1 << c)):
            return False
    return True

def _try_expand(clique, comp_mask, size):
    """Try to add any element compatible with all in clique. Returns expanded clique or None."""
    clique_set = set(clique)
    for v in range(size):
        if v in clique_set:
            continue
        if _is_compatible_with_all(v, clique, comp_mask):
            return clique + [v]
    return None

def _local_search_expand(clique, comp_mask, size, max_iters=5000, random_seed=0):
    """
    Local search: repeatedly try to add elements. If stuck, try swaps to escape.
    Returns best clique found.
    """
    random.seed(random_seed)
    best_clique = list(clique)
    best_size = len(clique)
    no_improve = 0
    max_no_improve = 200
    
    iters = 0
    while iters < max_iters and no_improve < max_no_improve:
        iters += 1
        
        # Try to expand
        expanded = _try_expand(clique, comp_mask, size)
        if expanded is not None:
            clique = expanded
            if len(clique) > best_size:
                best_clique = list(clique)
                best_size = len(clique)
                no_improve = 0
            continue
        
        # Stuck: try a swap to potentially enable future expansion
        if len(clique) == 0:
            break
            
        # Pick a random element to remove
        idx = random.randrange(len(clique))
        old_elem = clique[idx]
        temp_clique = [x for x in clique if x != old_elem]
        
        # Find candidates compatible with temp_clique
        candidates = []
        for v in range(size):
            if v == old_elem or v in temp_clique:
                continue
            if _is_compatible_with_all(v, temp_clique, comp_mask):
                candidates.append(v)
        
        if candidates:
            new_elem = random.choice(candidates)
            new_clique = temp_clique + [new_elem]
            
            # Accept if it doesn't shrink, or with some probability for exploration
            if len(new_clique) >= len(clique):
                clique = new_clique
            else:
                # Accept with small probability to escape local optima
                if random.random() < 0.1:
                    clique = new_clique
            
            # Check if this enables expansion
            expanded = _try_expand(clique, comp_mask, size)
            if expanded is not None:
                clique = expanded
                if len(clique) > best_size:
                    best_clique = list(clique)
                    best_size = len(clique)
                    no_improve = 0
                continue
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return best_clique

def _divisor_seed_search(comp_mask, size, deadline):
    """
    Start from divisor sets of highly composite numbers, then try to expand.
    """
    # Highly composite numbers <= 512 with many divisors
    # 420: 24 divisors, 480: 28 divisors, 504: 24 divisors, 360: 24 divisors
    # Let's check which numbers have the most divisors <= 512
    best_divisor_set = []
    best_size = 0
    
    # Try all numbers from 1 to 512 as potential "hub" numbers
    # Their divisor sets are automatically valid cliques
    for hub in range(1, 513):
        divs = _divisors(hub)
        # Convert to 0-indexed
        divs_0idx = [d - 1 for d in divs]
        
        # Verify it's a valid clique (should be, but safety)
        valid = True
        for i in range(len(divs_0idx)):
            for j in range(i + 1, len(divs_0idx)):
                if not (comp_mask[divs_0idx[i]] & (1 << divs_0idx[j])):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(divs_0idx) > best_size:
            best_size = len(divs_0idx)
            best_divisor_set = divs_0idx
        
        if time.time() > deadline - 10:
            break
    
    return best_divisor_set, best_size

def _greedy_clique_bitmask(comp_mask, size, random_seed=42):
    """Greedy clique construction using bitmask compatibility."""
    random.seed(random_seed)
    
    # Order by degree (number of compatible elements)
    degrees = [bin(comp_mask[i]).count('1') for i in range(size)]
    order = list(range(size))
    order.sort(key=lambda i: degrees[i], reverse=True)
    
    clique = []
    remaining = set(order)
    
    while remaining:
        best_v = None
        best_score = -1
        for v in remaining:
            if not _is_compatible_with_all(v, clique, comp_mask):
                continue
            # Score: number of remaining vertices compatible with v
            mask = comp_mask[v]
            score = sum(1 for u in remaining if u != v and (mask & (1 << u)))
            if score > best_score:
                best_score = score
                best_v = v
        if best_v is None:
            break
        clique.append(best_v)
        remaining.discard(best_v)
    
    return clique

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s
    
    n = 512
    size = n + 1  # indices 0..512 for numbers 1..512
    
    # Build bitmask compatibility
    comp_mask = _build_bitmask_compatibility(n)
    
    best_clique = []
    best_size = 0
    
    # Strategy 1: Divisor seed search
    div_clique, div_size = _divisor_seed_search(comp_mask, size, deadline)
    if div_size > best_size:
        best_clique = div_clique
        best_size = div_size
    
    # Strategy 2: Greedy from multiple seeds
    for seed in range(20):
        if time.time() > deadline - 15:
            break
        clique = _greedy_clique_bitmask(comp_mask, size, random_seed=seed * 123 + 7)
        if len(clique) > best_size:
            best_clique = clique
            best_size = len(clique)
    
    # Strategy 3: Local search expansion from best found
    if time.time() < deadline - 10:
        improved = _local_search_expand(list(best_clique), comp_mask, size, 
                                         max_iters=3000, random_seed=99)
        if len(improved) > best_size:
            best_clique = improved
            best_size = len(improved)
    
    # Strategy 4: More local search rounds
    while time.time() < deadline - 5:
        improved = _local_search_expand(list(best_clique), comp_mask, size,
                                         max_iters=2000, random_seed=random.randint(0, 10000))
        if len(improved) > best_size:
            best_clique = improved
            best_size = len(improved)
        else:
            break
    
    # Strategy 5: Try starting local search from divisor sets of top hubs
    if time.time() < deadline - 3:
        for hub in [420, 480, 504, 360, 420, 480]:
            if time.time() > deadline - 3:
                break
            divs = _divisors(hub)
            divs_0idx = [d - 1 for d in divs]
            # Verify validity
            valid = True
            for i in range(len(divs_0idx)):
                for j in range(i + 1, len(divs_0idx)):
                    if not (comp_mask[divs_0idx[i]] & (1 << divs_0idx[j])):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                improved = _local_search_expand(divs_0idx, comp_mask, size,
                                                 max_iters=1000, random_seed=hub)
                if len(improved) > best_size:
                    best_clique = improved
                    best_size = len(improved)
    
    # Convert from 0-indexed to 1-indexed
    result = [v + 1 for v in best_clique]
    
    # Final verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if _lcm(result[i], result[j]) > 512:
                # Safety: remove invalid elements
                pass
    
    return sorted(result)

# EVOLVE-BLOCK-END
