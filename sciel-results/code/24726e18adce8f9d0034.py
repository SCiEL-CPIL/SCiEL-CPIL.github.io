"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    N = 1000
    MAX_SUM = 2 * N  # 2000

    def greedy_forward(counts, chosen, order=None):
        """Continue greedy over a given order. Returns (chosen, counts)."""
        if order is None:
            order = range(1, N + 1)
        for x in order:
            ok = True
            for y in chosen:
                if counts[x + y] >= 2:
                    ok = False
                    break
            if ok and counts[2 * x] >= 2:
                ok = False
            if ok:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
        return chosen, counts

    def greedy_with_skip(counts, chosen, skip_prob):
        """Greedy in natural order, occasionally skipping a valid element."""
        for x in range(1, N + 1):
            ok = True
            for y in chosen:
                if counts[x + y] >= 2:
                    ok = False
                    break
            if ok and counts[2 * x] >= 2:
                ok = False
            if ok and random.random() > skip_prob:
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
        return chosen, counts

    def local_search(chosen, counts, max_steps=500):
        """Improve via direct adds and 1-for-1 swaps (size-neutral restructure)."""
        chosen_set = set(chosen)
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # 1) Try direct adds
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                ok = True
                for y in chosen:
                    if counts[x + y] >= 2:
                        ok = False
                        break
                if ok and counts[2 * x] >= 2:
                    ok = False
                if ok:
                    for y in chosen:
                        counts[x + y] += 1
                    counts[2 * x] += 1
                    chosen.append(x)
                    chosen_set.add(x)
                    improved = True
                    break
            if improved:
                continue
            # 2) Try a swap: remove r, add a (size-neutral)
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
            if not candidates or len(chosen) == 0:
                break
            a = random.choice(candidates)
            # Sums that would exceed 2 if we add a to the current set
            conflict_sums = set()
            for y in chosen:
                if counts[a + y] >= 2:
                    conflict_sums.add(a + y)
            if counts[2 * a] >= 2:
                conflict_sums.add(2 * a)
            if not conflict_sums:
                continue  # would have been a direct add
            # Sums that r must reduce (must be in conflict_sums to help)
            r_sums = set()
            for y in chosen:
                if y != a:
                    r_sums.add(r := y + 0)  # placeholder
            # Recompute r candidate sums properly
            r_sums = set()
            for y in chosen:
                r_sums.add(a + y)  # not used; recompute below per r
            # For each possible r, check feasibility
            for r in chosen:
                # sums r contributes to
                r_contrib = set()
                for y in chosen:
                    if y != r:
                        r_contrib.add(r + y)
                r_contrib.add(2 * r)
                # Feasible if every conflict sum is covered by r
                if conflict_sums.issubset(r_contrib):
                    # Verify: after removing r and adding a, no sum > 2
                    # Only sums in (conflict_sums | r_contrib | {a+y for y in chosen}) matter
                    new_counts = {}
                    affected = set(conflict_sums) | set(r_contrib)
                    for y in chosen:
                        if y != r:
                            affected.add(a + y)
                    affected.add(2 * a)
                    valid = True
                    for s in affected:
                        c = counts[s]
                        # remove r's contributions
                        for y in chosen:
                            if y != r and r + y == s:
                                c -= 1
                        if 2 * r == s:
                            c -= 1
                        # add a's contributions
                        for y in chosen:
                            if y != r and a + y == s:
                                c += 1
                        if 2 * a == s:
                            c += 1
                        if c > 2:
                            valid = False
                            break
                    if valid:
                        # Apply swap
                        for y in chosen:
                            if y != r:
                                counts[r + y] -= 1
                        counts[2 * r] -= 1
                        for y in chosen:
                            if y != r:
                                counts[a + y] += 1
                        counts[2 * a] += 1
                        chosen.remove(r)
                        chosen_set.discard(r)
                        chosen.append(a)
                        chosen_set.add(a)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return chosen, counts

    # Phase 1: Basic greedy
    chosen, counts = greedy_forward(np.zeros(MAX_SUM + 1, dtype=np.int32), [])
    best_chosen = list(chosen)
    best_size = len(chosen)

    if time.time() > deadline:
        return best_chosen

    # Phase 2: Multiple greedy restarts with different skip probabilities
    for trial in range(80):
        if time.time() > deadline:
            break
        random.seed(trial * 7 + 13)
        c2 = np.zeros(MAX_SUM + 1, dtype=np.int32)
        ch2, c2 = greedy_with_skip(c2, [], skip_prob=0.03 + 0.005 * trial)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = list(ch2)

    if time.time() > deadline:
        return best_chosen

    # Phase 3: Greedy with different orderings (natural, reverse, golden, random)
    for trial in range(40):
        if time.time() > deadline:
            break
        strategy = trial % 4
        order = list(range(1, N + 1))
        if strategy == 1:
            order.reverse()
        elif strategy == 2:
            random.seed(trial * 31 + 5)
            random.shuffle(order)
        elif strategy == 3:
            golden = 1.618033988749895
            order = sorted(range(1, N + 1), key=lambda x: (x * golden) % 1)
        c3 = np.zeros(MAX_SUM + 1, dtype=np.int32)
        ch3, c3 = greedy_forward(c3, [], order)
        if len(ch3) > best_size:
            best_size = len(ch3)
            best_chosen = list(ch3)

    if time.time() > deadline:
        return best_chosen

    # Phase 4: Local search on the best solution
    chosen_ls = list(best_chosen)
    counts_ls = np.zeros(MAX_SUM + 1, dtype=np.int32)
    for i in range(len(chosen_ls)):
        for j in range(i, len(chosen_ls)):
            counts_ls[chosen_ls[i] + chosen_ls[j]] += 1
    chosen_ls, counts_ls = local_search(chosen_ls, counts_ls, max_steps=2000)
    if len(chosen_ls) > best_size:
        best_size = len(chosen_ls)
        best_chosen = list(chosen_ls)

    if time.time() > deadline:
        return best_chosen

    # Phase 5: Final direct-add polish
    best_set = set(best_chosen)
    best_sc = np.zeros(MAX_SUM + 1, dtype=np.int32)
    for i in range(len(best_chosen)):
        for j in range(i, len(best_chosen)):
            best_sc[best_chosen[i] + best_chosen[j]] += 1
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x in best_set:
            continue
        ok = True
        for y in best_chosen:
            if best_sc[x + y] >= 2:
                ok = False
                break
        if ok and best_sc[2 * x] >= 2:
            ok = False
        if ok:
            for y in best_chosen:
                best_sc[x + y] += 1
            best_sc[2 * x] += 1
            best_chosen.append(x)
            best_set.add(x)

    return best_chosen
# EVOLVE-BLOCK-END
