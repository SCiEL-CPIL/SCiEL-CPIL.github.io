"""Combined: bitmask representation with systematic 2-opt local search and repair."""
import time
import random
import math

N = 256
DEADLINE = 85.0  # safety margin under 90s kill


def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Precompute APs (0-indexed internally, convert to 1-indexed at end)
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element, store the OTHER elements in each 4-AP it participates in
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # For each element, store the OTHER elements in each 3-AP it participates in
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

    # Precompute weights
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]

    def is_safe_add(S, x):
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                return False
        return True

    def gain_3aps(S, x):
        gain = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        return gain

    def loss_3aps(S, x):
        loss = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                loss += 1
        return loss

    def count_3aps(S):
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def repair_4aps(S):
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    elems = (a, b, c, d)
                    worst_idx = min(range(4), key=lambda i: (elem_3ap_weight[elems[i]], -elem_4ap_weight[elems[i]]))
                    worst = elems[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def greedy_construct(order):
        S = 0
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def systematic_2opt(S, max_iters=100):
        """Systematic 2-opt: try all single adds, then all swaps to improve."""
        current_S = S
        current_score = count_3aps(current_S)

        for it in range(max_iters):
            if time.time() > DEADLINE - 3:
                break

            improved = False
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds first (fastest wins)
            best_add = None
            best_add_gain = 0
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                g = gain_3aps(current_S, x)
                if g > best_add_gain:
                    best_add_gain = g
                    best_add = x

            if best_add is not None and best_add_gain > 0:
                current_S |= (1 << best_add)
                current_score += best_add_gain
                improved = True
                continue

            # Try swaps: remove x, add y
            best_swap = None
            best_swap_gain = 0
            for x_remove in members:
                S_after = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after, x_add):
                        continue
                    gain = gain_3aps(S_after, x_add)
                    net = gain - loss
                    if net > best_swap_gain:
                        best_swap_gain = net
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_swap_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_swap_gain
                improved = True
                continue

            # Try 2-swap: remove two, add two
            if len(members) >= 2 and len(non_members) >= 2:
                best_2swap = None
                best_2swap_gain = 0
                # Sample a subset for speed
                sample_size = min(len(members), 20)
                sample_remove = rng.sample(members, sample_size)
                sample_add_size = min(len(non_members), 20)
                sample_add = rng.sample(non_members, sample_add_size)

                for x1 in sample_remove:
                    S_after1 = current_S & ~(1 << x1)
                    loss1 = loss_3aps(current_S, x1)
                    for x2 in sample_remove:
                        if x2 == x1:
                            continue
                        S_after12 = S_after1 & ~(1 << x2)
                        loss2 = loss_3aps(S_after1, x2)
                        total_loss = loss1 + loss2
                        for y1 in sample_add:
                            if not is_safe_add(S_after12, y1):
                                continue
                            gain1 = gain_3aps(S_after12, y1)
                            S_after12y1 = S_after12 | (1 << y1)
                            for y2 in sample_add:
                                if y2 == y1:
                                    continue
                                if not is_safe_add(S_after12y1, y2):
                                    continue
                                gain2 = gain_3aps(S_after12y1, y2)
                                total_gain = gain1 + gain2
                                net = total_gain - total_loss
                                if net > best_2swap_gain:
                                    best_2swap_gain = net
                                    best_2swap = (x1, x2, y1, y2)

                if best_2swap is not None and best_2swap_gain > 0:
                    x1, x2, y1, y2 = best_2swap
                    current_S = (current_S & ~(1 << x1) & ~(1 << x2)) | (1 << y1) | (1 << y2)
                    current_score += best_2swap_gain
                    improved = True
                    continue

            if not improved:
                break

        return current_S, current_score

    def initial_greedy():
        S = 0
        order = list(range(N))
        rng.shuffle(order)
        order.sort(key=lambda x: -elem_3ap_weight[x])
        return greedy_construct(order)

    # Run multiple restarts with systematic 2-opt
    best_S = 0
    best_score = -1

    # First greedy start
    S0 = initial_greedy()
    S0 = repair_4aps(S0)
    s0 = count_3aps(S0)
    if s0 > best_score:
        best_S, best_score = S0, s0

    # Multiple restarts
    n_restarts = 12
    for i in range(n_restarts):
        if time.time() > DEADLINE - 15:
            break

        # Vary the greedy order
        S = 0
        order = list(range(N))
        rng.shuffle(order)
        # Mix weight-based and random
        if i % 3 == 0:
            order.sort(key=lambda x: -elem_3ap_weight[x])
        elif i % 3 == 1:
            order.sort(key=lambda x: -elem_4ap_weight[x])
        # else: random order

        S = greedy_construct(order)
        S = repair_4aps(S)

        # Systematic 2-opt
        S2, sc = systematic_2opt(S, max_iters=50)

        # Final repair and count
        S2 = repair_4aps(S2)
        sc = count_3aps(S2)

        if sc > best_score:
            best_S, best_score = S2, sc

    # Final long 2-opt from best
    if time.time() < DEADLINE - 5:
        S2, sc = systematic_2opt(best_S, max_iters=30)
        S2 = repair_4aps(S2)
        sc = count_3aps(S2)
        if sc > best_score:
            best_S, best_score = S2, sc

    # Convert to 1-indexed and verify
    result = [x + 1 for x in range(N) if (best_S >> x) & 1]

    # Safety: verify no 4-AP
    valid = True
    for a, b, c, d in four_aps:
        if (best_S >> a) & 1 and (best_S >> b) & 1 and (best_S >> c) & 1 and (best_S >> d) & 1:
            valid = False
            break

    if not valid or not result:
        # Fallback: simple greedy
        S = 0
        for x in range(N):
            if is_safe_add(S, x):
                S |= (1 << x)
        result = [x + 1 for x in range(N) if (S >> x) & 1]

    return sorted(result)
