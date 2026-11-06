import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute all 3-APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))

    # Precompute all 4-APs
    aps_4 = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            aps_4.append((s, s + d, s + 2 * d, s + 3 * d))

    # For each element x, precompute:
    # - three_ap_pairs[x]: list of (y, z) such that {x,y,z} forms a 3-AP
    # - four_ap_triples[x]: list of (y1, y2, y3) such that {x,y1,y2,y3} forms a 4-AP
    three_ap_pairs = [[] for _ in range(n + 1)]
    four_ap_triples = [[] for _ in range(n + 1)]

    for a, b, c in aps_3:
        three_ap_pairs[a].append((b, c))
        three_ap_pairs[b].append((a, c))
        three_ap_pairs[c].append((a, b))

    for s, s1, s2, s3 in aps_4:
        four_ap_triples[s].append((s1, s2, s3))
        four_ap_triples[s1].append((s, s2, s3))
        four_ap_triples[s2].append((s, s1, s3))
        four_ap_triples[s3].append((s, s1, s2))

    # Use a set for fast membership testing
    # Represent S as a Python set for clarity, but use precomputed structures for speed

    def can_add(x, S):
        """Check if x can be added to S without creating a 4-AP."""
        for y1, y2, y3 in four_ap_triples[x]:
            if y1 in S and y2 in S and y3 in S:
                return False
        return True

    def gain_if_add(x, S):
        """Count new 3-APs formed by adding x to S."""
        gain = 0
        for y, z in three_ap_pairs[x]:
            if y in S and z in S:
                gain += 1
        return gain

    def score_set(S):
        """Count total 3-APs in S."""
        cnt = 0
        for a, b, c in aps_3:
            if a in S and b in S and c in S:
                cnt += 1
        return cnt

    def greedy_construct(order, use_score=False, min_gain=0):
        """Greedy construction following the given order."""
        S = set()
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                g = gain_if_add(x, S)
                if g >= min_gain or len(S) < 20:
                    S.add(x)
            else:
                S.add(x)
        return S

    def local_search(S, max_iters=100):
        """Steepest-ascent local search with add/remove/swap moves."""
        S = set(S)
        current_score = score_set(S)

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False

            # Phase 1: Try adding elements (steepest ascent)
            best_add = None
            best_add_gain = 0
            for x in range(1, n + 1):
                if x in S:
                    continue
                if can_add(x, S):
                    g = gain_if_add(x, S)
                    if g > best_add_gain:
                        best_add_gain = g
                        best_add = x

            if best_add is not None and best_add_gain > 0:
                S.add(best_add)
                current_score += best_add_gain
                improved = True

            # Phase 2: Try removing elements (if it allows better additions)
            # Only do this every few iterations to save time
            if iteration % 2 == 0:
                best_remove = None
                best_remove_net = 0

                for y in list(S):
                    if time.time() > deadline - 5:
                        break

                    # Tentatively remove y
                    S.remove(y)

                    # Find best additions possible after removal
                    add_gain = 0
                    added = []
                    for x in range(1, n + 1):
                        if x in S:
                            continue
                        if can_add(x, S):
                            g = gain_if_add(x, S)
                            if g > 0:
                                S.add(x)
                                added.append(x)
                                add_gain += g

                    # Compute loss from removing y
                    # y was contributing to 3-APs where it appeared
                    loss = 0
                    for a, b, c in aps_3:
                        if y in (a, b, c):
                            others = [v for v in (a, b, c) if v != y]
                            if all(o in S for o in others):
                                loss += 1

                    net = add_gain - loss
                    if net > best_remove_net:
                        best_remove_net = net
                        best_remove = y
                        best_added = added[:]

                    # Revert
                    S.add(y)
                    for x in added:
                        S.discard(x)

                if best_remove is not None and best_remove_net > 0:
                    S.remove(best_remove)
                    for x in best_added:
                        S.add(x)
                    current_score = score_set(S)
                    improved = True

            if not improved:
                break

        return S, current_score

    def hill_climbing(S, max_iters=200):
        """More aggressive local search with random restarts within the search."""
        S = set(S)
        current_score = score_set(S)
        best_S = set(S)
        best_score = current_score

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False

            # Try all single additions
            best_add = None
            best_add_gain = 0
            for x in range(1, n + 1):
                if x in S:
                    continue
                if can_add(x, S):
                    g = gain_if_add(x, S)
                    if g > best_add_gain:
                        best_add_gain = g
                        best_add = x

            if best_add is not None and best_add_gain > 0:
                S.add(best_add)
                current_score += best_add_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                continue

            # Try removals with re-optimization
            if iteration % 3 == 0:
                for y in list(S):
                    if time.time() > deadline - 5:
                        break

                    S.remove(y)
                    add_gain = 0
                    added = []
                    for x in range(1, n + 1):
                        if x in S:
                            continue
                        if can_add(x, S):
                            g = gain_if_add(x, S)
                            if g > 0:
                                S.add(x)
                                added.append(x)
                                add_gain += g

                    new_score = score_set(S)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                    else:
                        S.add(y)
                        for x in added:
                            S.discard(x)

            if not improved and iteration > 0:
                break

        return best_S, best_score

    # Main: run multiple strategies and keep the best
    best_S = set()
    best_score = 0

    # Strategy 1: Natural order greedy + local search
    if time.time() < deadline - 15:
        S1 = greedy_construct(range(1, n + 1), use_score=False)
        S1, sc1 = hill_climbing(S1, max_iters=100)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Score-aware greedy with multiple random restarts
    random.seed(123)
    for restart in range(15):
        if time.time() > deadline - 15:
            break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        S_r = greedy_construct(random_order, use_score=True, min_gain=1)
        S_r, sc_r = hill_climbing(S_r, max_iters=80)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 3: Start from full set, prune 4-APs, then optimize
    if time.time() < deadline - 15:
        S_p = set(range(1, n + 1))
        # Remove elements to break all 4-APs
        for ap in aps_4:
            if all(elem in S_p for elem in ap):
                # Remove element that contributes least to 3-APs
                min_contrib = None
                min_val = float('inf')
                for elem in ap:
                    if elem in S_p:
                        # Count 3-APs this element is part of (rough estimate)
                        contrib = len(three_ap_pairs[elem])
                        if contrib < min_val:
                            min_val = contrib
                            min_contrib = elem
                if min_contrib:
                    S_p.discard(min_contrib)

        S_p, sc_p = hill_climbing(S_p, max_iters=100)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 4: Start with a random subset of moderate size, then optimize
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        random.seed(42 + restart)
        # Start with ~100 random elements
        S_init = set(random.sample(range(1, n + 1), 100))
        # Prune 4-APs
        for ap in aps_4:
            if all(elem in S_init for elem in ap):
                S_init.discard(random.choice(ap))
        S_init, sc_init = hill_climbing(S_init, max_iters=80)
        if sc_init > best_score:
            best_score = sc_init
            best_S = S_init

    # Final hill climbing on best
    if time.time() < deadline - 5 and best_S:
        best_S, best_score = hill_climbing(best_S, max_iters=50)

    return sorted(best_S)

# EVOLVE-BLOCK-END
