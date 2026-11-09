import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # Pre-compute forbidden triples and pairs
    forbidden = set()
    for a in numbers:
        for b in numbers:
            if a >= b: continue
            g_ab = gcd(a, b)
            for c in numbers:
                if b >= c: continue
                g_bc = gcd(b, c)
                if g_ab != g_bc: continue
                g_ca = gcd(c, a)
                if g_ca == g_ab:
                    forbidden.add((a, b, c))
                    
    pairs_forbidden = {x: set() for x in numbers}
    for a, b, c in forbidden:
        pairs_forbidden[a].add((b, c))
        pairs_forbidden[a].add((c, b))
        pairs_forbidden[b].add((a, c))
        pairs_forbidden[b].add((c, a))
        pairs_forbidden[c].add((a, b))
        pairs_forbidden[c].add((b, a))
        
    # Calculate degree (number of forbidden pairs) for greedy sorting
    degrees = {x: len(pairs_forbidden[x]) for x in numbers}

    def can_add(x, S):
        for y, z in pairs_forbidden[x]:
            if y in S and z in S:
                return False
        return True

    def greedy_solve(seed_val, use_weight=True):
        random.seed(seed_val)
        S = set()
        candidates = numbers.copy()
        if use_weight:
            # Sort by degree, then random tie-breaker
            # To allow randomness, we can shuffle first, then stable sort by degree
            random.shuffle(candidates)
            candidates.sort(key=lambda x: degrees[x])
        else:
            random.shuffle(candidates)
            
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, max_iters=50):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any element
            added = False
            # Try elements not in S
            # Sort by degree? Or random? Random is fine for local search.
            for x in numbers:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        added = True
                        improved = True
                        break
            
            if added:
                continue
                
            # 2. Try swap: Remove x, Add y, then try to add z
            found_swap = False
            xs = list(S)
            random.shuffle(xs)
            for x in xs:
                S_temp = S - {x}
                # Try to add y
                # We can iterate through numbers not in S_temp
                for y in numbers:
                    if y in S_temp:
                        continue
                    if can_add(y, S_temp):
                        S_temp.add(y)
                        # Try to add z
                        for z in numbers:
                            if z in S_temp:
                                continue
                            if can_add(z, S_temp):
                                S = S_temp
                                S.add(z)
                                found_swap = True
                                improved = True
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
        return S

    best_set = set()
    best_size = 0
    
    # Phase 1: Weighted Greedy + Local Search
    for i in range(20):
        S = greedy_solve(i, use_weight=True)
        S = local_search(S, max_iters=50)
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Phase 2: ILS
    for i in range(300):
        S_perturbed = set(best_set)
        if len(S_perturbed) > 3:
            to_remove = random.sample(list(S_perturbed), 3)
            for x in to_remove:
                S_perturbed.remove(x)
        
        # Re-optimize
        # Greedy fill first
        candidates = [x for x in numbers if x not in S_perturbed]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S_perturbed):
                S_perturbed.add(x)
                
        # Then local search
        S_new = local_search(S_perturbed, max_iters=30)
        
        if len(S_new) > best_size:
            best_size = len(S_new)
            best_set = S_new

    return sorted(list(best_set))
