import time
import random

N = 256

# ---- Precompute per-element 3-AP pair lists and 4-AP conflict triples ----
f3 = [[] for _ in range(N + 1)]   # x as FIRST term of a 3-AP: (x+d, x+2d)
m3 = [[] for _ in range(N + 1)]   # x as MIDDLE term: (x-d, x+d)
l3 = [[] for _ in range(N + 1)]   # x as LAST term: (x-2d, x-d)
f4 = [[] for _ in range(N + 1)]   # 4-APs containing x: tuple of the OTHER three elements

for x in range(1, N + 1):
    # x first: x+2d <= N
    for d in range(1, (N - x) // 2 + 1):
        f3[x].append((x + d, x + 2 * d))
    # x middle: x-d >= 1 and x+d <= N
    for d in range(1, min(x - 1, N - x) + 1):
        m3[x].append((x - d, x + d))
    # x last: x-2d >= 1
    for d in range(1, (x - 1) // 2 + 1):
        l3[x].append((x - 2 * d, x - d))
    # x in a 4-AP, four positions
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4[x].append((x + d, x + 2 * d, x + 3 * d))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4[x].append((x - d, x + d, x + 2 * d))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4[x].append((x - 2 * d, x - d, x + d))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4[x].append((x - 3 * d, x - 2 * d, x - d))

# Precompute bitmask of the other 3 elements for each 4-AP conflict
f4_mask = [0] * (N + 1)
f4_mask_pairs = [[] for _ in range(N + 1)]  # list of (mask1, mask2) where mask1|mask2 = full 3-element mask
for x in range(1, N + 1):
    for (a, b, c) in f4[x]:
        f4_mask[x] |= (1 << a) | (1 << b) | (1 << c)
        f4_mask_pairs[x].append(((1 << a) | (1 << b), 1 << c))

# Precompute bitmask for gain: for each x, list of pair-bitmasks (a|b) for each 3-AP where x participates
gain_masks = [[] for _ in range(N + 1)]
for x in range(1, N + 1):
    for (a, b) in f3[x]:
        gain_masks[x].append((1 << a) | (1 << b))
    for (a, b) in m3[x]:
        gain_masks[x].append((1 << a) | (1 << b))
    for (a, b) in l3[x]:
        gain_masks[x].append((1 << a) | (1 << b))


def count_3AP_total(present):
    s = 0
    for x in range(1, N + 1):
        if present[x]:
            for (a, b) in m3[x]:
                if present[a] and present[b]:
                    s += 1
    return s


def count_3AP_mask(mask):
    s = 0
    for x in range(1, N + 1):
        if mask & (1 << x):
            for (a, b) in m3[x]:
                if (mask >> a) & 1 and (mask >> b) & 1:
                    s += 1
    return s


def greedy_seed(rng):
    mask = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        # check if adding x creates a 4-AP: any 4-AP conflict triple all present
        ok = True
        for (a, b, c) in f4[x]:
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                ok = False
                break
        if ok:
            mask |= (1 << x)
    return mask


def local_search(mask, deadline):
    while time.time() < deadline:
        improved = False
        # --- best positive-gain addition ---
        bx, bg = None, 0
        for x in range(1, N + 1):
            if mask & (1 << x):
                continue
            # check 4-AP conflicts
            conflict = False
            for (a, b, c) in f4[x]:
                if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                    conflict = True
                    break
            if conflict:
                continue
            g = 0
            for pm in gain_masks[x]:
                if mask & pm:
                    g += 1
            if g > bg:
                bg, bx = g, x
        if bx is not None:
            mask |= (1 << bx)
            improved = True
        # --- best net-positive swap (remove r, add x) ---
        if not improved:
            bs, bsx, bnet = None, None, 0
            for r in range(1, N + 1):
                if not (mask & (1 << r)):
                    continue
                # loss from removing r
                loss = 0
                for pm in gain_masks[r]:
                    if mask & pm:
                        loss += 1
                mask_without_r = mask & ~(1 << r)
                for x in range(1, N + 1):
                    if mask_without_r & (1 << x):
                        continue
                    # check 4-AP conflicts with r removed
                    conflict = False
                    for (a, b, c) in f4[x]:
                        if (mask_without_r >> a) & 1 and (mask_without_r >> b) & 1 and (mask_without_r >> c) & 1:
                            conflict = True
                            break
                    if conflict:
                        continue
                    g = 0
                    for pm in gain_masks[x]:
                        if mask_without_r & pm:
                            g += 1
                    net = g - loss
                    if net > bnet:
                        bnet, bs, bsx = net, r, x
            if bs is not None and bnet > 0:
                mask &= ~(1 << bs)
                mask |= (1 << bsx)
                improved = True
        if not improved:
            break
    return count_3AP_mask(mask)


def polish_2swap(mask, deadline):
    """Try 2-removal + 2-addition swaps."""
    while time.time() < deadline:
        improved = False
        best_net = 0
        best_r1 = best_r2 = best_x1 = best_x2 = None
        present_list = [x for x in range(1, N + 1) if mask & (1 << x)]
        absent_list = [x for x in range(1, N + 1) if not (mask & (1 << x))]
        
        for i in range(len(present_list)):
            r1 = present_list[i]
            for j in range(i + 1, len(present_list)):
                r2 = present_list[j]
                mask_no = mask & ~(1 << r1) & ~(1 << r2)
                # compute loss
                loss = 0
                for pm in gain_masks[r1]:
                    if mask_no & pm:
                        loss += 1
                for pm in gain_masks[r2]:
                    if mask_no & pm:
                        loss += 1
                # also count 3-APs that had both r1 and r2
                # (already handled since gain_masks counts pairs where both endpoints present)
                
                for x1 in absent_list:
                    if mask_no & (1 << x1):
                        continue
                    conflict1 = False
                    for (a, b, c) in f4[x1]:
                        if (mask_no >> a) & 1 and (mask_no >> b) & 1 and (mask_no >> c) & 1:
                            conflict1 = True
                            break
                    if conflict1:
                        continue
                    g1 = 0
                    for pm in gain_masks[x1]:
                        if mask_no & pm:
                            g1 += 1
                    mask_temp = mask_no | (1 << x1)
                    for x2 in absent_list:
                        if x2 == x1 or mask_temp & (1 << x2):
                            continue
                        conflict2 = False
                        for (a, b, c) in f4[x2]:
                            if (mask_temp >> a) & 1 and (mask_temp >> b) & 1 and (mask_temp >> c) & 1:
                                conflict2 = True
                                break
                        if conflict2:
                            continue
                        g2 = 0
                        for pm in gain_masks[x2]:
                            if mask_temp & pm:
                                g2 += 1
                        net = g1 + g2 - loss
                        if net > best_net:
                            best_net = net
                            best_r1, best_r2, best_x1, best_x2 = r1, r2, x1, x2
        if best_r1 is not None and best_net > 0:
            mask &= ~(1 << best_r1) & ~(1 << best_r2)
            mask |= (1 << best_x1) | (1 << best_x2)
            improved = True
        if not improved:
            break
    return count_3AP_mask(mask)


def construct():
    start = time.time()
    # safety margin well under the 60s eval / 90s kill
    deadline = start + 52.0
    rng = random.Random(12345)

    best_mask = 0
    best_score = -1

    # main multi-restart search
    while time.time() < deadline - 6.0:
        mask = greedy_seed(rng)
        score = local_search(mask, deadline)
        if score > best_score:
            best_score = score
            best_mask = mask

    # final polish with remaining time on the incumbent
    if time.time() < deadline - 2.0 and best_mask != 0:
        local_search(best_mask, time.time() + 2.0)
        best_score = count_3AP_mask(best_mask)
    
    # 2-swap polish
    if time.time() < deadline - 0.5 and best_mask != 0:
        score2 = polish_2swap(best_mask, deadline)
        if score2 > best_score:
            best_score = score2

    if best_mask == 0:
        best_mask = greedy_seed(random.Random(0))

    A = [x for x in range(1, N + 1) if best_mask & (1 << x)]
    if not A:
        A = [1]

    # hard validity guarantee: drop any element that sits in a 4-AP
    present = [False] * (N + 1)
    for x in A:
        present[x] = True
    for x in list(A):
        for (a, b, c) in f4[x]:
            if present[a] and present[b] and present[c]:
                present[x] = False
                A.remove(x)
                break
    if not A:
        A = [1]
    return A
