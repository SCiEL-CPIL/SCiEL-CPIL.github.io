"""Improved 4-AP-free set construction: faster 2-opt, iterated local search, and better greedy."""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s


def construct():
    start_time = time.time()
    rng = random.Random(12345)

    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

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

    # Precompute 3-AP weights
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    # Precompute 4-AP weights
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]

    def is_safe_add(S, x):
        """Check if adding x to S (bitmask) creates any 4-AP involving x."""
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
        """Number of new 3-APs formed by adding x to S."""
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
        """Number of 3-APs lost by removing x from S (x must be in S)."""
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

    def count_3aps_bitmask(S):
        """Count 3-APs in bitmask S (0-indexed)."""
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def count_4aps_bitmask(S):
        """Count 4-APs fully contained in bitmask S."""
        cnt = 0
        for a, b, c, d in four_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs, prioritizing low 3-AP weight."""
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
        """Greedy: add elements in given order, skip if creates 4-AP."""
        S = 0
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def greedy_max_gain(S):
        """Greedy: repeatedly add the element with maximum 3-AP gain that is safe."""
        S_cur = S
        while True:
            best_x = -1
            best_gain = 0
            for x in range(N):
                if (S_cur >> x) & 1:
                    continue
                if not is_safe_add(S_cur, x):
                    continue
                g = gain_3aps(S_cur, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            S_cur |= (1 << best_x)
        return S_cur

    def local_search_2opt(S, max_iters=300):
        """2-opt local search with efficient 2-swap exploration."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False

            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                continue

            # Try swaps: remove x, add y
            best_swap = None
            best_gain = 0
            for x_remove in members:
                S_after_remove = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after_remove, x_add):
                        continue
                    gain = gain_3aps(S_after_remove, x_add)
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue

            # Try 2-swap with larger sample
            if len(members) >= 2 and len(non_members) >= 2:
                sample_size = min(len(members), 40)
                sample_non_size = min(len(non_members), 40)
                sample_members = rng.sample(members, sample_size)
                sample_non_members = rng.sample(non_members, sample_non_size)

                best_2swap = None
                best_2swap_gain = 0
                for i, x1 in enumerate(sample_members):
                    S_after1 = current_S & ~(1 << x1)
                    loss1 = loss_3aps(current_S, x1)
                    for x2 in sample_members[i+1:]:
                        S_after2 = S_after1 & ~(1 << x2)
                        loss2 = loss_3aps(S_after1, x2)
                        total_loss = loss1 + loss2

                        for y1 in sample_non_members:
                            if not is_safe_add(S_after2, y1):
                                continue
                            S_after2_y1 = S_after2 | (1 << y1)
                            gain1 = gain_3aps(S_after2, y1)

                            for y2 in sample_non_members:
                                if y2 == y1:
                                    continue
                                if not is_safe_add(S_after2_y1, y2):
                                    continue
                                gain2 = gain_3aps(S_after2_y1, y2)
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

    def local_search_2opt_heavy(S, max_iters=150):
        """Enhanced 2-opt with larger 2-swap samples and 3-swap moves."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False

            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                continue

            # Try swaps
            best_swap = None
            best_gain = 0
            for x_remove in members:
                S_after_remove = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after_remove, x_add):
                        continue
                    gain = gain_3aps(S_after_remove, x_add)
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue

            # Try 2-swap with even larger sample
            if len(members) >= 2 and len(non_members) >= 2:
                sample_size = min(len(members), 50)
                sample_non_size = min(len(non_members), 50)
                sample_members = rng.sample(members, sample_size)
                sample_non_members = rng.sample(non_members, sample_non_size)

                best_2swap = None
                best_2swap_gain = 0
                for i, x1 in enumerate(sample_members):
                    S_after1 = current_S & ~(1 << x1)
                    loss1 = loss_3aps(current_S, x1)
                    for x2 in sample_members[i+1:]:
                        S_after2 = S_after1 & ~(1 << x2)
                        loss2 = loss_3aps(S_after1, x2)
                        total_loss = loss1 + loss2

                        for y1 in sample_non_members:
                            if not is_safe_add(S_after2, y1):
                                continue
                            S_after2_y1 = S_after2 | (1 << y1)
                            gain1 = gain_3aps(S_after2, y1)

                            for y2 in sample_non_members:
                                if y2 == y1:
                                    continue
                                if not is_safe_add(S_after2_y1, y2):
                                    continue
                                gain2 = gain_3aps(S_after2_y1, y2)
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

    def perturbation_rebuild(S, k_remove):
        """Perturb: remove k random elements, then greedily re-add best elements."""
        members = [x for x in range(N) if (S >> x) & 1]
        if len(members) < k_remove:
            return S
        to_remove = rng.sample(members, k_remove)
        S_pert = S
        for x in to_remove:
            S_pert &= ~(1 << x)
        # Greedily add back best elements
        S_pert = greedy_max_gain(S_pert)
        return S_pert

    # === PHASE 1: Multi-start greedy with structured orderings ===
    best_S = 0
    best_score = -1

    # Ordering 1: by 3-AP weight descending
    order1 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    S1 = greedy_construct(order1)
    S1 = repair_4aps(S1)
    score1 = count_3aps_bitmask(S1)
    if score1 > best_score:
        best_score = score1
        best_S = S1

    # Ordering 2: by 3-AP weight descending, tie by 4-AP weight ascending
    order2 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], elem_4ap_weight[x], x))
    S2 = greedy_construct(order2)
    S2 = repair_4aps(S2)
    score2 = count_3aps_bitmask(S2)
    if score2 > best_score:
        best_score = score2
        best_S = S2

    # Ordering 3: residue class mod 3
    order3 = []
    for r in range(3):
        for x in range(r, N, 3):
            order3.append(x)
    S3 = greedy_construct(order3)
    S3 = repair_4aps(S3)
    score3 = count_3aps_bitmask(S3)
    if score3 > best_score:
        best_score = score3
        best_S = S3

    # Ordering 4: residue class mod 4
    order4 = []
    for r in range(4):
        for x in range(r, N, 4):
            order4.append(x)
    S4 = greedy_construct(order4)
    S4 = repair_4aps(S4)
    score4 = count_3aps_bitmask(S4)
    if score4 > best_score:
        best_score = score4
        best_S = S4

    # Ordering 5: random shuffle
    order5 = list(range(N))
    rng.shuffle(order5)
    S5 = greedy_construct(order5)
    S5 = repair_4aps(S5)
    score5 = count_3aps_bitmask(S5)
    if score5 > best_score:
        best_score = score5
        best_S = S5

    # Ordering 6: weighted random
    remaining = list(range(N))
    rem_weights = [elem_3ap_weight[x] + 1 for x in range(N)]
    order6 = []
    for _ in range(N):
        total = sum(rem_weights)
        r = rng.random() * total
        cum = 0
        for i, w in enumerate(rem_weights):
            cum += w
            if r <= cum:
                order6.append(remaining[i])
                remaining.pop(i)
                rem_weights.pop(i)
                break
    S6 = greedy_construct(order6)
    S6 = repair_4aps(S6)
    score6 = count_3aps_bitmask(S6)
    if score6 > best_score:
        best_score = score6
        best_S = S6

    # Ordering 7: max-gain greedy from empty
    S7 = greedy_max_gain(0)
    score7 = count_3aps_bitmask(S7)
    if score7 > best_score:
        best_score = score7
        best_S = S7

    # Ordering 8: another random
    order8 = list(range(N))
    rng.shuffle(order8)
    S8 = greedy_construct(order8)
    S8 = repair_4aps(S8)
    score8 = count_3aps_bitmask(S8)
    if score8 > best_score:
        best_score = score8
        best_S = S8

    # === PHASE 2: 2-opt local search on best ===
    if time.time() - start_time < TIME_LIMIT - 15:
        S_ls, score_ls = local_search_2opt(best_S, max_iters=300)
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls

    # === PHASE 3: Heavy local search ===
    if time.time() - start_time < TIME_LIMIT - 8:
        S_hs, score_hs = local_search_2opt_heavy(best_S, max_iters=150)
        if score_hs > best_score:
            best_score = score_hs
            best_S = S_hs

    # === PHASE 4: Iterated local search with perturbation ===
    deadline = start_time + TIME_LIMIT - 3
    ils_count = 0
    while time.time() < deadline:
        ils_count += 1

        # Perturb the best solution
        k_remove = rng.randint(3, 8)
        S_pert = perturbation_rebuild(best_S, k_remove)
        score_pert = count_3aps_bitmask(S_pert)

        # Local search on perturbed
        S_ils, score_ils = local_search_2opt(S_pert, max_iters=150)
        if score_ils > best_score:
            best_score = score_ils
            best_S = S_ils

        # Also try a fresh random start with local search
        if time.time() < deadline - 5:
            order_r = list(range(N))
            rng.shuffle(order_r)
            S_r = greedy_construct(order_r)
            S_r = repair_4aps(S_r)
            S_ils2, score_ils2 = local_search_2opt(S_r, max_iters=100)
            if score_ils2 > best_score:
                best_score = score_ils2
                best_S = S_ils2

    # === PHASE 5: Final heavy local search ===
    if time.time() - start_time < deadline - 1:
        S_final, score_final = local_search_2opt_heavy(best_S, max_iters=80)
        if score_final > best_score:
            best_score = score_final
            best_S = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(best_S) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (best_S >> x) & 1]

    if not result:
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result
