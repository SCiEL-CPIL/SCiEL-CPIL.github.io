import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 80  # safety margin before 90s kill
    n = limit

    # Precompute compatibility using LCM directly for speed
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            lcm_val = i * j // gcd(i, j)
            if lcm_val <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    # Greedy coloring for upper bound - optimized with bit_count
    def color_bound(P_mask):
        if P_mask == 0:
            return 0
        verts = []
        m = P_mask
        while m:
            v = (m & -m).bit_length() - 1
            verts.append(v)
            m &= m - 1
        if not verts:
            return 0
        # Sort by degree in subgraph (using bit_count for speed)
        verts.sort(key=lambda v: (compat_mask[v] & P_mask).bit_count(), reverse=True)
        color = {}
        max_c = 0
        for i, v in enumerate(verts):
            used = set()
            for j in range(i):
                u = verts[j]
                if (compat_mask[v] & (1 << u)):
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c + 1 > max_c:
                max_c = c + 1
        return max_c

    best_mask = 0
    best_size = 0
    stopped = [False]

    def bb(P_mask, clique_mask, depth):
        nonlocal best_mask, best_size
        if stopped[0]:
            return
        if time.time() > deadline - 5:
            stopped[0] = True
            return

        cur_size = clique_mask.bit_count()
        if P_mask == 0:
            if cur_size > best_size:
                best_mask = clique_mask
                best_size = cur_size
            return

        ub = cur_size + color_bound(P_mask)
        if ub <= best_size:
            return

        P_list = []
        m = P_mask
        while m:
            v = (m & -m).bit_length() - 1
            P_list.append(v)
            m &= m - 1
        P_list.sort(key=lambda v: (compat_mask[v] & P_mask).bit_count(), reverse=True)

        for v in P_list:
            if stopped[0]:
                return
            new_clique = clique_mask | (1 << v)
            new_P = P_mask & ~(1 << v) & compat_mask[v]
            bb(new_P, new_clique, depth + 1)
            if stopped[0]:
                return
            P_mask &= ~(1 << v)
            if P_mask == 0:
                if cur_size > best_size:
                    best_mask = clique_mask
                    best_size = cur_size
                return
            ub = cur_size + color_bound(P_mask)
            if ub <= best_size:
                return
            # Additional pruning: if remaining vertices can't beat best
            remaining_count = P_mask.bit_count()
            if cur_size + remaining_count <= best_size:
                return

    # Initialize with greedy solution - use bit_count for speed
    comp_counts = [compat_mask[i].bit_count() for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if (S & ~compat_mask[x]) == 0:
            S |= (1 << x)
    best_mask = S
    best_size = S.bit_count()

    # Try divisor sets - more candidates and faster validation
    for m in [360, 420, 480, 504, 240, 336, 384, 432, 468, 512, 288, 360, 420, 480, 504, 252, 336, 384, 432, 468, 504, 512, 320, 360, 420, 480, 504, 240, 288, 336, 384, 432, 468, 504, 512]:
        if m > n:
            continue
        divs = [d for d in range(1, m + 1) if m % d == 0]
        # Fast check: max LCM in divisor set is m itself
        if m <= n:
            dm = 0
            for d in divs:
                dm |= (1 << d)
            if len(divs) > best_size:
                best_size = len(divs)
                best_mask = dm

    # Run branch and bound
    all_mask = 0
    for i in range(1, n + 1):
        all_mask |= (1 << i)
    bb(all_mask, 0, 0)

    # Local search polish - more aggressive
    if time.time() < deadline - 8 and not stopped[0]:
        def can_add(S_mask, x):
            return (S_mask & ~compat_mask[x]) == 0
        def complete(S_mask):
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for x in range(1, n + 1):
                    if time.time() > deadline - 5:
                        break
                    if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                        S_mask |= (1 << x)
                        changed = True
            return S_mask
        S = complete(best_mask)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
        
        # Try swapping: remove one element, try to add two
        if time.time() < deadline - 5:
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if (best_mask & (1 << x)) == 0:
                    continue
                # Try removing x
                S2 = best_mask & ~(1 << x)
                # Try adding any compatible element
                for y in range(1, n + 1):
                    if (S2 & (1 << y)) == 0 and (S2 & ~compat_mask[y]) == 0:
                        S3 = S2 | (1 << y)
                        if S3.bit_count() > best_size:
                            best_size = S3.bit_count()
                            best_mask = S3
                            break

    # Final validation and return
    result = mask_to_set(best_mask)
    # Quick sanity check
    assert len(result) > 0
    return result
