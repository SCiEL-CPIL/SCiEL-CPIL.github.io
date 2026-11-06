import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 128

    all_aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(all_aps):
        for p in ap:
            pos_aps[p].append(i)

    def _ap_violation(ap, colors):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        distinct = 1
        if c1 != c0:
            distinct += 1
        if c2 != c0 and c2 != c1:
            distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2:
            distinct += 1
        return distinct < 3

    def _count_violations(colors):
        v = 0
        for ap in all_aps:
            if _ap_violation(ap, colors):
                v += 1
        return v

    def _local_search(num_colors, init_colors, max_iters=30000):
        colors = init_colors[:]
        viol = _count_violations(colors)
        if viol == 0:
            return colors

        best_colors = colors[:]
        best_viol = viol

        for it in range(max_iters):
            if it % 2000 == 0:
                if time.time() > deadline:
                    break
                if viol < best_viol:
                    best_viol = viol
                    best_colors = colors[:]
                    if best_viol == 0:
                        return best_colors

            p = random.randint(0, n - 1)
            old_color = colors[p]

            best_color = old_color
            best_cost = None

            for nc in range(num_colors):
                colors[p] = nc
                cost = 0
                for ap_idx in pos_aps[p]:
                    ap = all_aps[ap_idx]
                    if _ap_violation(ap, colors):
                        cost += 1
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_color = nc

            colors[p] = best_color

            if best_color != old_color:
                viol = _count_violations(colors)
                if viol < best_viol:
                    best_viol = viol
                    best_colors = colors[:]
                    if best_viol == 0:
                        return best_colors

        viol = _count_violations(colors)
        if viol < best_viol:
            best_viol = viol
            best_colors = colors[:]

        return best_colors if best_viol == 0 else None

    # Try 3 colors with various inits
    inits = []
    for seed in range(15):
        rng = random.Random(seed)
        inits.append([rng.randint(0, 2) for _ in range(n)])
    inits.append([x % 3 for x in range(n)])
    inits.append([(x * 7) % 3 for x in range(n)])
    inits.append([(x * 13) % 3 for x in range(n)])

    for init in inits:
        if time.time() > deadline:
            break
        result = _local_search(3, init, max_iters=30000)
        if result is not None:
            return result

    # Try 4 colors
    for seed in range(15):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        init = [rng.randint(0, 3) for _ in range(n)]
        result = _local_search(4, init, max_iters=30000)
        if result is not None:
            return result

    # Fallback greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - j * d for j in (1, 2, 3)]
                if all(0 <= i < len(colors) for i in idxs):
                    c = {color} | {colors[i] for i in idxs}
                    if len(c) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
