import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    n = 128

    # Precompute all 4-term APs (0-indexed positions)
    aps = []
    for a in range(1, n + 1):
        md = (n - a) // 3
        for d in range(1, md + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    m = len(aps)

    # For each position, list of AP indices that include it
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)

    # Fast distinct-count for 4 values (inline branch logic)
    def distinct4(a, b, c, d):
        if a == b:
            if a == c:
                if a == d:
                    return 1
                return 2
            if a == d:
                return 2
            if c == d:
                return 2
            return 3
        else:
            if c == a:
                if d == a:
                    return 2
                if d == b:
                    return 3
                return 3
            elif c == b:
                if d == b:
                    return 2
                if d == a:
                    return 3
                return 3
            else:
                if d == a:
                    return 3
                if d == b:
                    return 3
                if d == c:
                    return 3
                return 4

    def is_bad_at(colors, idx):
        ap = aps[idx]
        return distinct4(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) < 3

    def compute_vset(colors):
        vset = set()
        for i in range(m):
            if is_bad_at(colors, i):
                vset.add(i)
        return vset

    def v2_coloring(k):
        # color[i] = v2(i+1) mod k
        colors = [0] * n
        for i in range(n):
            x = i + 1
            v = 0
            while x % 2 == 0:
                x //= 2
                v += 1
            colors[i] = v % k
        return colors

    def random_coloring(k, rng):
        return [rng.randrange(k) for _ in range(n)]

    def try_eliminate_color(colors, k, end):
        """
        Given a valid k-coloring, try to eliminate one color class via Kempe-chain swaps.
        Returns (new_colors, new_k) or (colors, k) if no improvement found within time.
        """
        used = set(colors)
        if len(used) < k:
            return colors, len(used)
        rng = random
        best = colors[:]
        best_k = k
        # Try to eliminate each color class
        for target in list(used):
            if time.time() > end:
                break
            # Count how many positions use target
            positions = [i for i in range(n) if colors[i] == target]
            if not positions:
                continue
            # For each position, try reassigning to a non-target color
            # We do a greedy pass: for each position in target class, pick a color
            # that keeps the coloring valid.
            trial = colors[:]
            success = True
            # Process positions in an order that minimizes disruption
            # Try to reassign each position to some color != target
            for p in positions:
                if time.time() > end:
                    success = False
                    break
                # Find a valid color for p (not target)
                found = False
                for c in range(k):
                    if c == target:
                        continue
                    if trial[p] == c:
                        continue
                    # Check all APs containing p
                    ok = True
                    for idx in pos_aps[p]:
                        ap = aps[idx]
                        # Compute distinct count with trial[p] replaced by c
                        a0 = ap[0]; a1 = ap[1]; a2 = ap[2]; a3 = ap[3]
                        v0 = c if a0 == p else trial[a0]
                        v1 = c if a1 == p else trial[a1]
                        v2 = c if a2 == p else trial[a2]
                        v3 = c if a3 == p else trial[a3]
                        if distinct4(v0, v1, v2, v3) < 3:
                            ok = False
                            break
                    if ok:
                        trial[p] = c
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                # Verify the trial is fully valid
                vset = compute_vset(trial)
                if not vset:
                    new_used = set(trial)
                    if len(new_used) < best_k:
                        best = trial[:]
                        best_k = len(new_used)
                        if best_k == 3:
                            return best, best_k
        return best, best_k

    def local_search(colors, k, end):
        """
        Standard local search: pick a bad AP, change one of its positions to a
        color not currently in the AP, minimize delta in bad-count.
        """
        rng = random
        vset = compute_vset(colors)
        stall = 0
        while time.time() < end and vset:
            ei = rng.choice(list(vset))
            ap = aps[ei]
            cur = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            cand = []
            for p in ap:
                for c in range(k):
                    if c not in cur:
                        cand.append((p, c))
            if not cand:
                cand = [(ap[0], rng.randrange(k))]
            best_delta = 10**9
            best_mv = None
            for (p, c) in cand:
                dlt = 0
                for idx in pos_aps[p]:
                    a2 = aps[idx]
                    was = idx in vset
                    # Compute distinct count after changing p to c
                    a0 = a2[0]; a1 = a2[1]; a2_ = a2[2]; a3 = a2[3]
                    v0 = c if a0 == p else colors[a0]
                    v1 = c if a1 == p else colors[a1]
                    v2 = c if a2_ == p else colors[a2_]
                    v3 = c if a3 == p else colors[a3]
                    now = distinct4(v0, v1, v2, v3) < 3
                    if was and not now:
                        dlt -= 1
                    elif (not was) and now:
                        dlt += 1
                if dlt < best_delta:
                    best_delta = dlt
                    best_mv = (p, c)
            if best_mv is None:
                break
            p, c = best_mv
            colors[p] = c
            for idx in pos_aps[p]:
                a2 = aps[idx]
                now = is_bad_at(colors, idx)
                was = idx in vset
                if was and not now:
                    vset.discard(idx)
                elif (not was) and now:
                    vset.add(idx)
            if best_delta < 0:
                stall = 0
            else:
                stall += 1
            if stall > 500:
                # Kick: random perturbations
                for _ in range(6):
                    p2 = rng.randrange(n)
                    c2 = rng.randrange(k)
                    if c2 == colors[p2]:
                        continue
                    colors[p2] = c2
                    for idx in pos_aps[p2]:
                        now = is_bad_at(colors, idx)
                        was = idx in vset
                        if was and not now:
                            vset.discard(idx)
                        elif (not was) and now:
                            vset.add(idx)
                stall = 0
        return colors

    def try_k(k, budget, rng):
        end = t0 + budget
        best = None
        best_du = k + 1
        while time.time() < end:
            # Initialize with 2-adic coloring (structurally valid) or random
            if rng.random() < 0.7:
                colors = v2_coloring(k)
            else:
                colors = random_coloring(k, rng)
            vset = compute_vset(colors)
            if vset:
                # Local search to fix
                colors = local_search(colors, k, end)
                vset = compute_vset(colors)
            if not vset:
                # Valid coloring: try to eliminate colors
                colors, du = try_eliminate_color(colors, k, end)
                if du < best_du:
                    best_du = du
                    best = colors[:]
                if best_du == 3:
                    return best
        return best

    # Main search loop
    result = None
    rng = random
    budgets = {3: 20, 4: 18, 5: 12, 6: 8, 7: 5, 8: 4}
    for k in range(3, 9):
        if time.time() > DEADLINE - 3:
            break
        b = budgets.get(k, 5)
        rem = DEADLINE - 3 - time.time()
        b = min(b, rem)
        if b <= 0.5:
            break
        res = try_k(k, b, rng)
        if res is not None:
            # Verify
            ok = True
            for ap in aps:
                if len({res[ap[0]], res[ap[1]], res[ap[2]], res[ap[3]]}) < 3:
                    ok = False
                    break
            if ok:
                result = res
                break

    if result is not None:
        return result

    # Fallback: greedy (known valid)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
