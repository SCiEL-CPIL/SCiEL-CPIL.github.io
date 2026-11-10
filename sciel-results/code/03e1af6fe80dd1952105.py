import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    # Use 181-bit mask (indices 1..180)
    FULL_MASK = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b <= N:
                    triples.append((a, b, c))
    
    # For each element x, precompute the bitmask of "conflict pairs"
    # If x is added, which pairs (u,v) in S would create a triple?
    # x as 'a': conflicts with (b,c) where (a,b,c) is triple
    # x as 'b': conflicts with (a,c) where (a,b,c) is triple
    # x as 'c': conflicts with (a,b) where (a,b,c) is triple
    # We'll store for each x, a list of (u,v) pairs such that if both u,v in S, x can't be added
    conflict_pairs = [set() for _ in range(N + 1)]
    for a, b, c in triples:
        # x=a conflicts with (b,c)
        conflict_pairs[a].add((b, c))
        # x=b conflicts with (a,c)
        conflict_pairs[b].add((a, c))
        # x=c conflicts with (a,b)
        conflict_pairs[c].add((a, b))
    
    # Convert conflict pairs to bitmask pairs for fast checking
    # For each x, store list of (mask_u, mask_v) where mask_u = 1<<u, mask_v = 1<<v
    conflict_masks = [list() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u, v in conflict_pairs[x]:
            conflict_masks[x].append((1 << u, 1 << v))
    
    def can_add_mask(S_mask, x):
        """Check if we can add x to S (represented as bitmask) without creating a triple."""
        for mask_u, mask_v in conflict_masks[x]:
            if (S_mask & mask_u) and (S_mask & mask_v):
                return False
        return True
    
    def mask_to_set(S_mask):
        return {x for x in range(1, N + 1) if (S_mask >> x) & 1}
    
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m
    
    def has_triple_mask(S_mask):
        """Check if bitmask S contains any forbidden triple."""
        for a in range(1, N + 1):
            if (S_mask >> a) & 1:
                for mask_b, mask_c in conflict_masks[a]:
                    if (S_mask & mask_b) and (S_mask & mask_c):
                        return True
        return False
    
    def greedy_build_mask(order):
        """Build a set by greedily adding elements in the given order, using bitmask."""
        S_mask = 0
        for x in order:
            if can_add_mask(S_mask, x):
                S_mask |= (1 << x)
        return S_mask
    
    def local_search_add_mask(S_mask, max_iters=2000):
        """Try to add any missing element repeatedly, using bitmask."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            for x in range(1, N + 1):
                if not ((S_mask >> x) & 1):
                    if can_add_mask(S_mask, x):
                        S_mask |= (1 << x)
                        improved = True
                        break
        return S_mask
    
    def removal_search_mask(S_mask, max_iters=1000):
        """Try removing elements that block the most additions, then re-add."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            best_removal = None
            best_gain = 0
            for y in range(1, N + 1):
                if not ((S_mask >> y) & 1):
                    continue
                S_without_y = S_mask & ~(1 << y)
                gain = 0
                for x in range(1, N + 1):
                    if not ((S_without_y >> x) & 1):
                        if can_add_mask(S_without_y, x):
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_removal = y
            if best_removal is not None and best_gain > 0:
                S_mask &= ~(1 << best_removal)
                S_mask = local_search_add_mask(S_mask, max_iters=200)
                improved = True
        return S_mask
    
    def swap_search_mask(S_mask, rng, max_iters=50000):
        """Try swapping elements to find better configurations."""
        iters = 0
        while iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            iters += 1
            # Pick a random y in S
            S_list = [x for x in range(1, N + 1) if (S_mask >> x) & 1]
            if not S_list:
                break
            y = rng.choice(S_list)
            x = rng.randint(1, N)
            if x == y or ((S_mask >> x) & 1):
                continue
            S_without_y = S_mask & ~(1 << y)
            if can_add_mask(S_without_y, x):
                S_mask = S_without_y | (1 << x)
        return S_mask
    
    def swap2_search_mask(S_mask, rng, max_iters=20000):
        """Try 2-for-2 swaps to find better configurations."""
        iters = 0
        while iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            iters += 1
            S_list = [x for x in range(1, N + 1) if (S_mask >> x) & 1]
            if len(S_list) < 2:
                break
            y1, y2 = rng.sample(S_list, 2)
            missing = [x for x in range(1, N + 1) if not ((S_mask >> x) & 1)]
            if len(missing) < 2:
                break
            x1, x2 = rng.sample(missing, 2)
            S_removed = S_mask & ~(1 << y1) & ~(1 << y2)
            if can_add_mask(S_removed, x1):
                S_tmp = S_removed | (1 << x1)
                if can_add_mask(S_tmp, x2):
                    S_mask = S_tmp | (1 << x2)
            elif can_add_mask(S_removed, x2):
                S_tmp = S_removed | (1 << x2)
                if can_add_mask(S_tmp, x1):
                    S_mask = S_tmp | (1 << x1)
        return S_mask
    
    # Compute constraint density for each element (number of conflict pairs)
    constraint_density = [0] * (N + 1)
    for x in range(1, N + 1):
        constraint_density[x] = len(conflict_pairs[x])
    
    # Strategy 1: Start with high numbers, add lower ones greedily
    S1_mask = 0
    for x in range(91, N + 1):
        S1_mask |= (1 << x)
    for x in range(90, 0, -1):
        if can_add_mask(S1_mask, x):
            S1_mask |= (1 << x)
    S1_mask = local_search_add_mask(S1_mask)
    S1_mask = removal_search_mask(S1_mask, max_iters=200)
    S1_mask = local_search_add_mask(S1_mask)
    
    best_mask = S1_mask
    best_size = bin(best_mask).count('1')
    
    rng = random.Random(42)
    
    # Strategy 2: Try various greedy orderings
    # Descending order
    order_desc = list(range(N, 0, -1))
    S2_mask = greedy_build_mask(order_desc)
    S2_mask = local_search_add_mask(S2_mask)
    S2_mask = removal_search_mask(S2_mask, max_iters=100)
    S2_mask = local_search_add_mask(S2_mask)
    if bin(S2_mask).count('1') > best_size:
        best_mask = S2_mask
        best_size = bin(S2_mask).count('1')
    
    # Ascending order
    order_asc = list(range(1, N + 1))
    S3_mask = greedy_build_mask(order_asc)
    S3_mask = local_search_add_mask(S3_mask)
    S3_mask = removal_search_mask(S3_mask, max_iters=100)
    S3_mask = local_search_add_mask(S3_mask)
    if bin(S3_mask).count('1') > best_size:
        best_mask = S3_mask
        best_size = bin(S3_mask).count('1')
    
    # Order by constraint density (least constrained first)
    order_density = sorted(range(1, N + 1), key=lambda x: constraint_density[x])
    S4_mask = greedy_build_mask(order_density)
    S4_mask = local_search_add_mask(S4_mask)
    S4_mask = removal_search_mask(S4_mask, max_iters=100)
    S4_mask = local_search_add_mask(S4_mask)
    if bin(S4_mask).count('1') > best_size:
        best_mask = S4_mask
        best_size = bin(S4_mask).count('1')
    
    # Multiple random orderings
    for iteration in range(400):
        if time.time() >= deadline - 10.0:
            break
        
        # Mix: 50% random, 30% density-ordered with noise, 20% reverse density
        r = rng.random()
        if r < 0.5:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif r < 0.8:
            order = list(range(1, N + 1))
            # Add noise to density ordering
            scores = [(constraint_density[x] + rng.random() * 2, x) for x in range(1, N + 1)]
            order = [x for _, x in sorted(scores)]
        else:
            order = sorted(range(1, N + 1), key=lambda x: -constraint_density[x])
        
        S_try = greedy_build_mask(order)
        S_try = local_search_add_mask(S_try, max_iters=500)
        
        sz = bin(S_try).count('1')
        if sz > best_size:
            best_mask = S_try
            best_size = sz
    
    # Swap search on best solution
    best_mask = swap_search_mask(best_mask, rng, max_iters=30000)
    best_mask = local_search_add_mask(best_mask)
    best_mask = swap2_search_mask(best_mask, rng, max_iters=10000)
    best_mask = local_search_add_mask(best_mask)
    
    # Try more random restarts with swap
    for iteration in range(150):
        if time.time() >= deadline - 5.0:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_try = greedy_build_mask(order)
        S_try = local_search_add_mask(S_try, max_iters=300)
        S_try = swap_search_mask(S_try, rng, max_iters=5000)
        S_try = local_search_add_mask(S_try)
        
        sz = bin(S_try).count('1')
        if sz > best_size:
            best_mask = S_try
            best_size = sz
    
    # Final polish: removal + local + swap2
    best_mask = removal_search_mask(best_mask, max_iters=500)
    best_mask = local_search_add_mask(best_mask)
    best_mask = swap2_search_mask(best_mask, rng, max_iters=15000)
    best_mask = local_search_add_mask(best_mask)
    
    # Convert to set and verify
    best_set = mask_to_set(best_mask)
    if has_triple_mask(best_mask):
        # Fallback: just use numbers > 90
        best_set = set(range(91, N + 1))
    
    return list(best_set)
