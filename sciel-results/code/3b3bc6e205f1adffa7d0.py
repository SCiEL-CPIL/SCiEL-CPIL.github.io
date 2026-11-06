import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128

    # Precompute all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))

    # Build conflict structures
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))

    # Precompute conflict bitmasks for fast can_add
    pair_mask = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            ub = 1 << (u - 1)
            vb = 1 << (v - 1)
            pair_mask[x].append((ub, vb))

    def can_add(x, mask):
        for ub, vb in pair_mask[x]:
            if (mask & ub) and (mask & vb):
                return False
        return True

    def popcount(x):
        return bin(x).count('1')

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def greedy_complete(mask, remaining_order):
        """Greedy add from mask using remaining_order, stopping when no more can be added."""
        selected = mask
        for x in remaining_order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def local_search(mask, max_iters=120):
        improved = True
        iters = 0

        while improved and time.time() < deadline - 3 and iters < max_iters:
            improved = False
            iters += 1

            # 1. Try to add any missing element (fast path)
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            best_new_mask = mask
            best_size = popcount(mask)

            for x in s_list:
                if time.time() > deadline - 3:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                added_count = 0
                for y in range(1, n + 1):
                    if y == x:
                        continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    sz = popcount(new_mask)
                    if sz > best_size:
                        best_size = sz
                        best_new_mask = new_mask

            if best_size > popcount(mask):
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap
            if time.time() < deadline - 4:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                best_new_mask = mask
                best_size = popcount(mask)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 4:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 6)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        new_mask = mask_no_xy
                        added_count = 0
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            sz = popcount(new_mask)
                            if sz > best_size:
                                best_size = sz
                                best_new_mask = new_mask
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    continue

            # 4. Random kick-and-refill if stuck
            if not improved and iters > 4:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                if len(s_list) > 5:
                    k = random.randint(1, min(3, len(s_list) - 2))
                    remove_set = set(random.sample(s_list, k))
                    perturb_mask = mask
                    for idx in remove_set:
                        perturb_mask &= ~(1 << (idx - 1))
                    # Refill greedily
                    order = list(range(1, n + 1))
                    random.shuffle(order)
                    for x in order:
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break

        return mask

    def branch_and_bound(selected_mask, candidate_mask, best_size, depth=0, max_depth=18):
        if time.time() > deadline - 6:
            return selected_mask, best_size

        upper_bound = popcount(selected_mask) + popcount(candidate_mask)
        if upper_bound <= best_size:
            return selected_mask, best_size

        if candidate_mask == 0:
            if popcount(selected_mask) > best_size:
                best_size = popcount(selected_mask)
                return selected_mask, best_size
            return selected_mask, best_size

        if depth >= max_depth:
            # Greedy completion with degree ordering
            greedy_mask = selected_mask
            remaining = candidate_mask
            while remaining and time.time() < deadline - 6:
                best_x = -1
                best_conflicts = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    x = lsb.bit_length()
                    conf = 0
                    for ub, vb in pair_mask[x]:
                        if remaining & ub:
                            conf += 1 if (remaining & vb) else 0
                    if conf < best_conflicts:
                        best_conflicts = conf
                        best_x = x
                    temp &= temp - 1
                if best_x == -1:
                    break
                if can_add(best_x, greedy_mask):
                    greedy_mask |= (1 << (best_x - 1))
                    remaining &= ~(1 << (best_x - 1))
                else:
                    remaining &= ~(1 << (best_x - 1))

            if popcount(greedy_mask) > best_size:
                best_size = popcount(greedy_mask)
                return greedy_mask, best_size
            return selected_mask, best_size

        # Choose the next candidate to branch on (most constrained)
        best_x = -1
        best_score = -1
        temp = candidate_mask
        while temp:
            lsb = temp & (-temp)
            x = lsb.bit_length()
            score = 0
            for ub, vb in pair_mask[x]:
                if candidate_mask & ub:
                    score += 1 if (candidate_mask & vb) else 0
            if score > best_score:
                best_score = score
                best_x = x
            temp &= temp - 1

        if best_x == -1:
            return selected_mask, best_size

        # Branch 1: Include best_x
        if can_add(best_x, selected_mask):
            new_selected = selected_mask | (1 << (best_x - 1))
            new_candidates = candidate_mask & ~(1 << (best_x - 1))
            # Remove conflicting pairs from candidates
            for ub, vb in pair_mask[best_x]:
                if new_candidates & ub:
                    new_candidates &= ~vb
                if new_candidates & vb:
                    new_candidates &= ~ub
            mask1, size1 = branch_and_bound(new_selected, new_candidates, best_size, depth + 1, max_depth)
            if size1 > best_size:
                best_size = size1
                selected_mask = mask1

        # Branch 2: Exclude best_x
        new_candidates = candidate_mask & ~(1 << (best_x - 1))
        mask2, size2 = branch_and_bound(selected_mask, new_candidates, best_size, depth + 1, max_depth)
        if size2 > best_size:
            best_size = size2
            selected_mask = mask2

        return selected_mask, best_size

    best_mask = 0
    best_size = 0

    # Precompute degrees
    degrees = [len(conflict_pairs[k]) for k in range(n + 1)]

    # Run multiple randomized greedy passes with local search
    num_restarts = 500

    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break

        strategy = i % 8
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.1)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.1)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif strategy == 4:
            # Weight by degree squared
            order = sorted(range(1, n + 1), key=lambda x: degrees[x]**2 + random.random())
        elif strategy == 5:
            # Reverse degree
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x]**2 + random.random())
        elif strategy == 6:
            # Random with slight bias toward low degree
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] * 0.5 + random.random() * 5)
        else:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)

        mask = greedy(order)
        mask = local_search(mask)

        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Branch-and-bound from empty set
    if time.time() < deadline - 12:
        bb_mask, bb_size = branch_and_bound(0, (1 << n) - 1, best_size, max_depth=14)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask

    # B&B starting from current best
    if time.time() < deadline - 6:
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask2, bb_size2 = branch_and_bound(best_mask, candidates, best_size, max_depth=10)
        if bb_size2 > best_size:
            best_size = bb_size2
            best_mask = bb_mask2

    # Final polish
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=150)

    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]

    if not result:
        result = [1]

    return result
