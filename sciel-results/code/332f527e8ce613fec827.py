"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0  # wall-clock safety margin under the 150s kill
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

    # Precompute balls
    ball = np.empty(N, dtype=object)
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
    while not covered.all() and _time_left() > 15.0:
        # Find uncovered words
        unc_idx = np.where(~covered)[0]
        if len(unc_idx) == 0:
            break

        # Find the uncovered word with minimum criticality (fewest potential coverers)
        # Sample a subset for speed if there are many
        sample_size = min(len(unc_idx), 200)
        if len(unc_idx) > sample_size:
            sample = unc_idx[rng.sample(range(len(unc_idx)), sample_size)]
        else:
            sample = unc_idx

        min_crit = N + 1
        target_word = -1
        for w in sample:
            crit = len(coverers[w])
            if crit < min_crit:
                min_crit = crit
                target_word = w
            if min_crit == 1:
                break

        if target_word == -1:
            target_word = sample[0]

        # Find the best codeword that covers target_word and maximizes new coverage
        candidates = [c for c in coverers[target_word] if c not in chosen]
        if not candidates:
            # Fallback: pick any uncovered word's coverer
            for w in unc_idx[:10]:
                candidates = [c for c in coverers[w] if c not in chosen]
                if candidates:
                    target_word = w
                    break
            if not candidates:
                break

        best_c = -1
        best_gain = 0
        # Evaluate top candidates by gain
        eval_candidates = candidates[:50] if len(candidates) > 50 else candidates
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
        while improved and _time_left() > 8.0:
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
                    if _time_left() < 5.0:
                        return

    try_remove_all()

    # 3. Optimized 1-Exchange (Reduction-oriented)
    def do_1_exchange_optimized():
        """
        Try to swap one codeword for another to enable future removals.
        Strategy: Remove c, find c' that covers unique words of c.
        Score c' by how many words it 'demotes' from unique coverage (count==1) to shared (count==2).
        """
        if not chosen or _time_left() < 5.0:
            return False

        # Pick a random codeword to try removing
        i = rng.choice(list(chosen))
        b_i = ball[i]
        
        # Words uniquely covered by i
        unique_mask = (covcount == 1)
        would_uncover = b_i[unique_mask[b_i]]
        
        if len(would_uncover) == 0:
            # Can remove without adding
            covcount[b_i] -= 1
            chosen.discard(i)
            return True

        # Find candidates j that cover all would_uncover
        # Intersection of coverers of all words in would_uncover
        if len(would_uncover) == 1:
            w = would_uncover[0]
            cands = [c for c in coverers[w] if c not in chosen]
        else:
            # Efficient intersection
            cands_set = set(coverers[int(would_uncover[0])])
            for w in would_uncover[1:]:
                cands_set &= set(coverers[int(w)])
            cands = [c for c in cands_set if c not in chosen]
            
        if not cands:
            return False

        # Evaluate candidates
        # We want to maximize the number of words w in ball(c) such that:
        # 1. w is NOT in would_uncover (those are forced)
        # 2. covcount[w] == 1 (currently uniquely covered by some other c_other)
        # This increases the redundancy of c_other.
        
        # Precompute unique mask excluding i? 
        # Actually, if w is in would_uncover, covcount[w] is 1 (by i). 
        # If we add c, covcount[w] becomes 2. 
        # For w NOT in would_uncover, if covcount[w] == 1, the unique coverer is NOT i.
        # So we just count sum(covcount[b_c] == 1) for b_c = ball(c).
        # But we must ensure c covers would_uncover.
        
        best_c = -1
        best_score = -1
        
        # Sample candidates if too many
        eval_cands = cands[:50] if len(cands) > 50 else cands
        
        for j in eval_cands:
            b_j = ball[j]
            # Score: number of words in b_j that are currently uniquely covered (by someone else)
            # Note: words in would_uncover are covered by i (count 1). 
            # If j covers them, they become count 2. 
            # The "benefit" is reducing the unique coverage burden on other codewords.
            # So we count words in b_j where covcount == 1.
            # This includes words in would_uncover (covered by i) and words covered by others.
            # If a word in would_uncover is covered by j, it's good (we need to cover it).
            # If a word NOT in would_uncover is covered by j and has count 1, it's good (creates redundancy for the other coverer).
            
            score = int(np.sum(covcount[b_j] == 1))
            
            if score > best_score:
                best_score = score
                best_c = j

        if best_c == -1:
            return False

        # Perform swap
        covcount[b_i] -= 1
        covcount[ball[best_c]] += 1
        chosen.discard(i)
        chosen.add(best_c)
        return True

    # 4. Simulated Annealing / Iterated Local Search
    # We alternate between removals and swaps.
    # If a swap doesn't immediately allow a removal, it might still be useful 
    # if it sets up a future removal. 
    # To keep it simple and fast, we accept swaps if they don't increase size (they don't) 
    # and we rely on the subsequent try_remove_all to harvest the benefits.
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995

    while _time_left() > 5.0:
        # Try to remove redundant codewords
        try_remove_all()

        if _time_left() < 10.0:
            break

        # Try 1-exchange
        if do_1_exchange_optimized():
            T *= cooling
            if T < T_min:
                T = T_min
            continue

        # If no move succeeded, cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Final cleanup
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
