import time
import random
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    # Precompute compatibility
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    # Greedy coloring for upper bound
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
        verts.sort(key=lambda v: bin(compat_mask[v] & P_mask).count('1'), reverse=True)
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

        cur_size = bin(clique_mask).count('1')
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
        P_list.sort(key=lambda v: bin(compat_mask[v] & P_mask).count('1'), reverse=True)

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

    # Initialize with greedy solution
    comp_counts = [bin(compat_mask[i]).count('1') for i in range(n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -comp_counts[x])
    S = 0
    for x in order:
        if (S & ~compat_mask[x]) == 0:
            S |= (1 << x)
    best_mask = S
    best_size = bin(S).count('1')

    # Try divisor sets
    for m in [360, 420, 480, 504, 240, 336, 384, 432, 468, 512]:
        if m > n:
            continue
        divs = [d for d in range(1, m + 1) if m % d == 0]
        if all(a * b // gcd(a, b) <= n for a in divs for b in divs):
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

    # Local search polish
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
        if bin(S).count('1') > best_size:
            best_size = bin(S).count('1')
            best_mask = S

    return mask_to_set(best_mask)
