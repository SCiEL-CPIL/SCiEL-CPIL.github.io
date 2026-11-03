"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 140.0  # wall-clock safety margin under the 150s kill
_START = [time.time()]

def _elapsed():
    return time.time() - _START[0]

def _time_left():
    return _DEADLINE - _elapsed()

def construct(q, n, radius):
    _START[0] = time.time()
    if (q, n, radius) != (3, 8, 1):
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    rng = random.Random(12345)
    N = 3 ** n

    # Precompute digits and indices
    def idx_to_digits(i):
        d = [0] * n
        for p in range(n):
            d[p] = i % 3
            i //= 3
        return d

    def digits_to_idx(d):
        v = 0
        mul = 1
        for p in range(n):
            v += d[p] * mul
            mul *= 3
        return v

    # Precompute balls using numpy for speed
    # For each word i, its ball contains i and all words at distance 1
    # We'll compute this efficiently
    ball = np.empty(N, dtype=object)
    
    # Precompute all indices
    all_indices = np.arange(N)
    
    # For each position p, compute the three variants
    # For position p, the digit is (i // 3**p) % 3
    # The two neighbors change that digit to the other two values
    
    # Precompute powers of 3
    pow3 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow3[i] = pow3[i-1] * 3
    
    for i in range(N):
        d = idx_to_digits(i)
        s = [i]
        for p in range(n):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    s.append(digits_to_idx(d))
            d[p] = orig
        ball[i] = np.array(s, dtype=np.int32)

    # Precompute coverers: for each word w, list of codewords c such that w in ball(c)
    coverers = [[] for _ in range(N)]
    for i in range(N):
        for w in ball[i]:
            coverers[w].append(i)

    # Convert coverers to numpy arrays for faster operations
    coverers_np = [np.array(c, dtype=np.int32) for c in coverers]

    # 1. Criticality-based Greedy Construction
    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_set = []
    for h in tetracode:
        for t in tetracode:
            w = h + t
            seed_set.append(digits_to_idx(w))
    seed_set = list(set(seed_set))

    covered = np.zeros(N, dtype=bool)
    covcount = np.zeros(N, dtype=np.int32)
    chosen = set(seed_set)
    for i in seed_set:
        covcount[ball[i]] += 1
        covered[ball[i]] = True

    # Criticality-based greedy fill
    while not covered.all() and _time_left() > 20.0:
        # Find uncovered words
        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break

        # Find the uncovered word with minimum criticality (fewest potential coverers)
        # Sample a subset for speed if there are many
        sample_size = min(len(unc_idx), 300)
        if len(unc_idx) > sample_size:
            sample = unc_idx[rng.sample(range(len(unc_idx)), sample_size)]
        else:
            sample = unc_idx

        min_crit = N + 1
        target_word = -1
        for w in sample:
            crit = len(coverers_np[w])
            if crit < min_crit:
                min_crit = crit
                target_word = w
            if min_crit == 1:
                break

        if target_word == -1:
            target_word = sample[0]

        # Find the best codeword that covers target_word and maximizes new coverage
        candidates = coverers_np[target_word]
        # Filter out already chosen
        chosen_arr = np.array(list(chosen), dtype=np.int32) if chosen else np.array([], dtype=np.int32)
        mask = np.isin(candidates, chosen_arr, invert=True)
        candidates = candidates[mask]

        if len(candidates) == 0:
            # Fallback: pick any uncovered word's coverer
            for w in unc_idx[:10]:
                candidates = coverers_np[w]
                mask = np.isin(candidates, chosen_arr, invert=True)
                candidates = candidates[mask]
                if len(candidates) > 0:
                    target_word = w
                    break
            if len(candidates) == 0:
                break

        best_c = -1
        best_gain = 0
        # Evaluate top candidates by gain
        eval_size = min(len(candidates), 80)
        eval_candidates = candidates[:eval_size] if len(candidates) > eval_size else candidates
        for c in eval_candidates:
            b = ball[c]
            gain = int(np.sum(~covered[b]))
            if gain > best_gain:
                best_gain = gain
                best_c = c

        if best_c == -1:
            best_c = eval_candidates[0]

        chosen.add(best_c)
        covcount[ball[best_c]] += 1
        covered[ball[best_c]] = True

    # Ensure valid
    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.add(i)
                covcount[ball[i]] += 1
                covered[ball[i]] = True

    # 2. Local Search: Remove redundant codewords
    def try_remove_all():
        improved = True
        while improved and _time_left() > 10.0:
            improved = False
            order = list(chosen)
            rng.shuffle(order)
            for i in order:
                if i not in chosen:
                    continue
                b = ball[i]
                if int(np.min(covcount[b])) >= 2:
                    covcount[b] -= 1
                    chosen.discard(i)
                    improved = True
                    if _time_left() < 8.0:
                        return

    try_remove_all()

    # 3. Local Search: 1-exchange and 2-exchange moves
    def do_1_exchange():
        """Try to swap one codeword for another to enable removal"""
        if not chosen or _time_left() < 8.0:
            return False

        i = rng.choice(list(chosen))
        b_i = ball[i]
        would_uncover = b_i[covcount[b_i] == 1]

        if len(would_uncover) == 0:
            # Can remove without adding
            covcount[b_i] -= 1
            chosen.discard(i)
            return True

        # Find candidates j that cover all would_uncover
        if len(would_uncover) == 1:
            w = would_uncover[0]
            cands = coverers_np[w]
            cands = cands[~np.isin(cands, np.array(list(chosen), dtype=np.int32))]
        else:
            # Need intersection of coverers for all would_uncover words
            cands_set = set(coverers_np[would_uncover[0]].tolist())
            for w in would_uncover[1:]:
                cands_set &= set(coverers_np[w].tolist())
            cands = np.array([c for c in cands_set if c not in chosen], dtype=np.int32)

        if len(cands) == 0:
            return False

        # Pick candidate that minimizes new unique coverage
        best_j = -1
        best_score = float('inf')
        eval_size = min(len(cands), 40)
        eval_cands = cands[:eval_size] if len(cands) > eval_size else cands
        for j in eval_cands:
            b_j = ball[j]
            # Score: number of words that would become uniquely covered by j
            new_unique = int(np.sum((covcount[b_j] == 0)))
            if new_unique < best_score:
                best_score = new_unique
                best_j = j

        if best_j == -1:
            return False

        # Perform swap
        covcount[b_i] -= 1
        covcount[ball[best_j]] += 1
        chosen.discard(i)
        chosen.add(best_j)
        return True

    def do_2_exchange():
        """Try to swap two codewords for two different ones"""
        if len(chosen) < 2 or _time_left() < 8.0:
            return False

        # Pick two codewords to potentially remove
        i, j = rng.sample(list(chosen), 2)
        b_i = ball[i]
        b_j = ball[j]

        # Find words that would become uncovered
        would_uncover = set()
        for w in b_i:
            if covcount[w] == 1:
                would_uncover.add(int(w))
        for w in b_j:
            if covcount[w] == 1:
                would_uncover.add(int(w))

        if not would_uncover:
            # Can remove both
            covcount[b_i] -= 1
            covcount[b_j] -= 1
            chosen.discard(i)
            chosen.discard(j)
            return True

        # Find pairs (k, l) that cover all would_uncover
        would_uncover_list = list(would_uncover)
        if len(would_uncover_list) > 8:
            return False

        # Find candidates that cover at least one would_uncover word
        all_cands = set()
        for w in would_uncover_list:
            all_cands.update(coverers_np[w].tolist())
        all_cands -= chosen
        all_cands = list(all_cands)

        if len(all_cands) < 2:
            return False

        # Sample pairs
        sample_size = min(len(all_cands), 25)
        if len(all_cands) > sample_size:
            sampled = rng.sample(all_cands, sample_size)
        else:
            sampled = all_cands

        best_pair = None
        best_score = float('inf')

        for idx1 in range(len(sampled)):
            for idx2 in range(idx1 + 1, len(sampled)):
                k, l = sampled[idx1], sampled[idx2]
                b_k = ball[k]
                b_l = ball[l]

                # Check if k and l together cover all would_uncover
                covers_all = True
                for w in would_uncover_list:
                    if w not in b_k and w not in b_l:
                        covers_all = False
                        break
                if not covers_all:
                    continue

                # Score: minimize new unique coverage
                new_unique = int(np.sum((covcount[b_k] == 0))) + \
                              int(np.sum((covcount[b_l] == 0)))
                if new_unique < best_score:
                    best_score = new_unique
                    best_pair = (k, l)

        if best_pair is None:
            return False

        k, l = best_pair
        # Perform swap
        covcount[b_i] -= 1
        covcount[b_j] -= 1
        covcount[ball[k]] += 1
        covcount[ball[l]] += 1
        chosen.discard(i)
        chosen.discard(j)
        chosen.add(k)
        chosen.add(l)
        return True

    # 4. Simulated Annealing with 1-exchange and 2-exchange
    T = 1.0
    T_min = 0.01
    cooling = 0.9998

    while _time_left() > 8.0:
        # Try to remove redundant codewords
        try_remove_all()

        if _time_left() < 12.0:
            break

        # Try 1-exchange
        if do_1_exchange():
            T *= cooling
            if T < T_min:
                T = T_min
            continue

        # Try 2-exchange (less frequent)
        if rng.random() < 0.35:
            if do_2_exchange():
                T *= cooling
                if T < T_min:
                    T = T_min
                continue

        # If no move succeeded, cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final cleanup - multiple passes
    for _ in range(3):
        if _time_left() < 5.0:
            break
        try_remove_all()

    # Verify and fix if needed
    final_covered = np.zeros(N, dtype=bool)
    for i in chosen:
        final_covered[ball[i]] = True
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen.add(i)
                final_covered[ball[i]] = True

    return [idx_to_digits(i) for i in chosen]
