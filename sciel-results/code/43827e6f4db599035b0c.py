import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # safety margin from 90s kill
    n = limit

    # Precompute LCM compatibility matrix using bitmasks for speed
    compat = [0] * (n + 1)  # bitmask of compatible elements
    for a in range(1, n + 1):
        mask = 0
        for b in range(1, n + 1):
            l = (a * b) // gcd(a, b)
            if l <= limit:
                mask |= (1 << b)
        compat[a] = mask

    def is_valid_add(S_mask, x):
        """Check if x is compatible with all elements in S_mask using bitmask."""
        # S_mask has bits set for elements in S
        # x is compatible with all in S iff (compat[x] & S_mask) == S_mask
        return (compat[x] & S_mask) == S_mask

    def complete_from_mask(S_mask):
        """Complete a set (given as bitmask) by adding all compatible elements."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 3:
                    break
                if not (S_mask & (1 << x)):
                    if (compat[x] & S_mask) == S_mask:
                        S_mask |= (1 << x)
                        changed = True
        return S_mask

    def mask_to_set(S_mask):
        return set(i for i in range(1, n + 1) if S_mask & (1 << i))

    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def greedy_color_bound(candidates_mask):
        """Greedy coloring upper bound on clique size in candidate subgraph."""
        if candidates_mask == 0:
            return 0
        candidates = [i for i in range(1, n + 1) if candidates_mask & (1 << i)]
        color = {}
        num_colors = 0
        for x in candidates:
            used = set()
            cm = compat[x] & candidates_mask
            cm &= ~(1 << x)
            for y in candidates:
                if y != x and (cm & (1 << y)) and y in color:
                    used.add(color[y])
            for c in range(num_colors + 1):
                if c not in used:
                    color[x] = c
                    break
            else:
                num_colors += 1
                color[x] = num_colors
        return num_colors

    best_set = set()
    best_size = 0
    best_mask = 0

    def get_divisors(m):
        divs = set()
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
        return divs

    # Try divisors of various highly composite numbers
    hc_numbers = [504, 480, 420, 360, 336, 288, 240, 210, 180, 168, 144, 120, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2]
    for m in hc_numbers:
        if time.time() > deadline - 20:
            break
        divs = get_divisors(m)
        dm = set_to_mask(divs)
        cm = complete_from_mask(dm)
        if cm.bit_count() > best_size:
            best_size = cm.bit_count()
            best_mask = cm
            best_set = mask_to_set(cm)

    # Also try ranges
    for r in [16, 12, 10, 8]:
        if time.time() > deadline - 20:
            break
        dm = set_to_mask(set(range(1, r + 1)))
        cm = complete_from_mask(dm)
        if cm.bit_count() > best_size:
            best_size = cm.bit_count()
            best_mask = cm
            best_set = mask_to_set(cm)

    # Branch and bound with bitmask
    def branch_and_bound(S_mask, candidates_mask, deadline):
        nonlocal best_mask, best_size, best_set
        if time.time() > deadline:
            return
        if S_mask.bit_count() + candidates_mask.bit_count() <= best_size:
            return
        if candidates_mask == 0:
            if S_mask.bit_count() > best_size:
                best_size = S_mask.bit_count()
                best_mask = S_mask
                best_set = mask_to_set(S_mask)
            return
        # Upper bound via greedy coloring
        ub = greedy_color_bound(candidates_mask)
        if S_mask.bit_count() + ub <= best_size:
            return
        # Order candidates by degree (number of compatible neighbors in candidates)
        candidates = [i for i in range(1, n + 1) if candidates_mask & (1 << i)]
        candidates.sort(key=lambda x: (compat[x] & candidates_mask).bit_count(), reverse=True)
        
        for x in candidates:
            if time.time() > deadline:
                return
            new_S = S_mask | (1 << x)
            new_cands = candidates_mask & compat[x] & ~(1 << x)
            branch_and_bound(new_S, new_cands, deadline)
            candidates_mask &= ~(1 << x)
            if S_mask.bit_count() + candidates_mask.bit_count() <= best_size:
                return

    # Run B&B from best seed
    if time.time() < deadline - 15:
        remaining = ((1 << (n + 1)) - 1) & ~best_mask
        # Filter to compatible with all of best_mask
        candidates = 0
        for x in range(1, n + 1):
            if remaining & (1 << x):
                if (compat[x] & best_mask) == best_mask:
                    candidates |= (1 << x)
        branch_and_bound(best_mask, candidates, deadline - 5)

    # Local search with bitmask operations
    def local_search(S_mask, deadline):
        nonlocal best_mask, best_size, best_set
        S_mask = complete_from_mask(S_mask)
        if S_mask.bit_count() > best_size:
            best_size = S_mask.bit_count()
            best_mask = S_mask
            best_set = mask_to_set(S_mask)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_set = mask_to_set(S_mask)
            # 1-for-1 swaps
            for e in S_set:
                if time.time() > deadline - 5:
                    break
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if not (S_mask & (1 << x)):
                        if (compat[x] & S_minus) == S_minus:
                            S2 = complete_from_mask(S_minus | (1 << x))
                            if S2.bit_count() > S_mask.bit_count():
                                S_mask = S2
                                improved = True
                                if S_mask.bit_count() > best_size:
                                    best_size = S_mask.bit_count()
                                    best_mask = S_mask
                                    best_set = mask_to_set(S_mask)
                            break
            # 1-for-2 swaps
            for e in S_set:
                if time.time() > deadline - 5:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = S_minus
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if not (S2 & (1 << x)):
                        if (compat[x] & S2) == S2:
                            S2 |= (1 << x)
                            added += 1
                            if added >= 2:
                                break
                if S2.bit_count() > S_mask.bit_count():
                    S_mask = complete_from_mask(S2)
                    improved = True
                    if S_mask.bit_count() > best_size:
                        best_size = S_mask.bit_count()
                        best_mask = S_mask
                        best_set = mask_to_set(S_mask)
        return S_mask

    # Run local search
    if time.time() < deadline - 10:
        local_search(best_mask, deadline)

    # Random restarts
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        m = rng.choice(hc_numbers)
        divs = get_divisors(m)
        seed = set()
        for d in divs:
            if rng.random() < 0.7:
                seed.add(d)
        if not seed:
            seed = {1}
        sm = set_to_mask(seed)
        sm = complete_from_mask(sm)
        local_search(sm, deadline)

    return sorted(best_set)
