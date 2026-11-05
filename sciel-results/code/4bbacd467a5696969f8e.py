# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s kill

    total = q ** n  # 729
    # Precompute the cover set (as a list of vertex indices) for every codeword.
    # A codeword c covers c itself and every word at Hamming distance 1.
    covers = []
    for c in itertools.product(range(q), repeat=n):
        s = [c]
        for pos in range(n):
            for val in range(q):
                if val == c[pos]:
                    continue
                lst = list(c)
                lst[pos] = val
                s.append(tuple(lst))
        covers.append(s)

    # Bitset representation: total bits -> words of 64 bits.
    nwords = (total + 63) // 64  # 729 -> 12

    def make_bitset(indices):
        words = [0] * nwords
        for idx in indices:
            words[idx >> 6] |= (1 << (idx & 63))
        return words

    # cover_bits[i] for codeword i
    cover_bits = [make_bitset(covers[i]) for i in range(total)]

    # Full mask
    full_mask = [0] * nwords
    for w in range(nwords):
        full_mask[w] = (1 << 64) - 1
    if total & 63:
        full_mask[-1] = (1 << (total & 63)) - 1

    def bitset_andnot(a, b):
        # a & ~b
        return [aw & ~bw for aw, bw in zip(a, b)]

    def bitset_count(words):
        return sum(bin(w).count("1") for w in words)

    def is_fully_covered(uncovered):
        for w in range(nwords):
            if uncovered[w] != 0:
                return False
        return True

    def greedy_cover(seed=None):
        if seed is not None:
            rng = random.Random(seed)
            weights = [rng.random() for _ in range(total)]
        else:
            weights = None

        uncovered = full_mask[:]
        chosen = []
        chosen_set = set()

        while not is_fully_covered(uncovered):
            if time.time() > deadline - 3.0:
                break
            best = -1
            best_key = -1.0
            for i in range(total):
                if i in chosen_set:
                    continue
                cnt = bitset_count([cw & uw for cw, uw in zip(cover_bits[i], uncovered)])
                if cnt <= 0:
                    continue
                key = float(cnt)
                if weights is not None:
                    key += weights[i] * 0.001
                if key > best_key:
                    best_key = key
                    best = i
            if best < 0:
                break
            chosen.append(best)
            chosen_set.add(best)
            cb = cover_bits[best]
            for w in range(nwords):
                uncovered[w] &= ~cb[w]
        return chosen

    def compute_count_only(chosen):
        co = [0] * total
        cnt = [0] * total
        last = [-1] * total
        for i in chosen:
            for v in covers[i]:
                cnt[v] += 1
                last[v] = i
        for v in range(total):
            if cnt[v] == 1:
                co[last[v]] += 1
        return co

    def full_covered(chosen):
        cov = [0] * nwords
        for i in chosen:
            cb = cover_bits[i]
            for w in range(nwords):
                cov[w] |= cb[w]
        for w in range(nwords):
            if cov[w] != full_mask[w]:
                return False
        return True

    def local_search(chosen, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5.0
        chosen = list(chosen)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            co = compute_count_only(chosen)
            # Try removals first
            for idx in range(len(chosen)):
                i = chosen[idx]
                if co[i] == 0:
                    newchosen = chosen[:idx] + chosen[idx+1:]
                    if full_covered(newchosen):
                        chosen = newchosen
                        improved = True
                        break
            if time.time() > time_limit:
                break
            # Try swaps to enable future removals
            co2 = compute_count_only(chosen)
            rem_old = sum(1 for k in chosen if co2[k] == 0)
            for idx in range(len(chosen)):
                if time.time() > time_limit:
                    break
                i = chosen[idx]
                # Only try swaps for codewords that are "hard to remove" (high count_only)
                if co2[i] == 0:
                    continue
                # Find vertices uniquely covered by i
                cnt = [0] * total
                last = [-1] * total
                for k in chosen:
                    for v in covers[k]:
                        cnt[v] += 1
                        last[v] = k
                unique_v = [v for v in covers[i] if cnt[v] == 1 and last[v] == i]
                if not unique_v:
                    continue
                # Find candidates j that cover all unique_v
                for j in range(total):
                    if j in chosen:
                        continue
                    ok = all(v in covers[j] for v in unique_v)
                    if not ok:
                        continue
                    newchosen = chosen[:idx] + [j] + chosen[idx+1:]
                    if not full_covered(newchosen):
                        continue
                    co_new = compute_count_only(newchosen)
                    rem_new = sum(1 for k in newchosen if co_new[k] == 0)
                    if rem_new > rem_old:
                        chosen = newchosen
                        improved = True
                        break
                if improved:
                    break
        return chosen

    # Main search: multiple greedy restarts with local search
    best = greedy_cover(0)
    best = local_search(best, deadline - 10.0)
    best_size = len(best)

    # Try more restarts with random seeds
    max_restarts = 30
    for seed in range(1, max_restarts + 1):
        if time.time() > deadline - 15.0:
            break
        cand = greedy_cover(seed)
        if len(cand) >= best_size:
            # Still try local search, might improve
            if time.time() > deadline - 10.0:
                continue
        cand = local_search(cand, deadline - 8.0)
        if len(cand) < best_size:
            best_size = len(cand)
            best = cand

    return [list(covers[i]) for i in best]
# EVOLVE-BLOCK-END
