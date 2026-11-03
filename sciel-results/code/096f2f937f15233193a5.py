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
    # This is the transpose of the ball relation.
    # coverers[w] = list of i such that w in ball[i]
    coverers = [[] for _ in range(N)]
    for i in range(N):
        for w in ball[i]:
            coverers[w].append(i)

    # 1. Initial Greedy Construction
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

    # Greedy fill
    while not covered.all() and _time_left() > 20.0:
        # Find best candidate
        # To speed up, we can just pick a random uncovered word and find a coverer, or greedy max gain.
        # Greedy max gain is O(N * ball_size). N=6561, ball_size=17. ~100k ops.
        # Doing this many times might be slow in Python.
        # Let's do a few rounds of greedy, then random.
        
        # Optimization: Only consider candidates that cover at least 1 new word.
        # We can maintain a list of uncovered words.
        # For each candidate i, gain = sum(1 for w in ball[i] if not covered[w])
        
        # Fast heuristic: Pick a random uncovered word w.
        # Pick a random coverer i of w that is not in chosen.
        # Add i.
        # This is O(1) per step roughly.
        
        # Let's try this random-greedy hybrid.
        if _time_left() > 10.0:
             # Find an uncovered word
             # To be efficient, we can keep a list of uncovered indices.
             # But updating it is costly.
             # Just scan? 6561 is small.
             unc_idx = np.where(~covered)[0]
             if len(unc_idx) == 0:
                 break
             w = unc_idx[rng.randint(0, len(unc_idx)-1)]
             candidates = [c for c in coverers[w] if c not in chosen]
             if not candidates:
                 # Should not happen if coverers is correct and we are adding
                 break
             # Pick the one that covers most new words?
             # Just pick random for speed, or greedy if time allows.
             # Let's do a small greedy choice among candidates
             best_c = -1
             best_gain = 0
             for c in candidates:
                 b = ball[c]
                 gain = int(np.sum(~covered[b]))
                 if gain > best_gain:
                     best_gain = gain
                     best_c = c
             if best_c != -1:
                 chosen.add(best_c)
                 covcount[ball[best_c]] += 1
                 covered[ball[best_c]] = True
             else:
                 # Fallback: just add any
                 c = candidates[0]
                 chosen.add(c)
                 covcount[ball[c]] += 1
                 covered[ball[c]] = True
        else:
             break

    # Ensure valid
    if not covered.all():
        for i in range(N):
            if not covered[i]:
                chosen.add(i)
                covcount[ball[i]] += 1
                covered[ball[i]] = True

    # 2. Local Search: Remove
    in_chosen = np.zeros(N, dtype=bool)
    for i in chosen:
        in_chosen[i] = True

    def try_remove_all():
        improved = True
        while improved and _time_left() > 2.0:
            improved = False
            order = list(chosen)
            rng.shuffle(order)
            for i in order:
                if not in_chosen[i]:
                    continue
                b = ball[i]
                if int(np.min(covcount[b])) >= 2:
                    covcount[b] -= 1
                    chosen.discard(i)
                    in_chosen[i] = False
                    improved = True
                    if _time_left() < 1.0:
                        return

    try_remove_all()

    # 3. Local Search: Swaps
    # We maintain a valid cover.
    # Move: Swap i (in) with j (out) if valid.
    # Validity: All words in ball(i) with covcount==1 must be in ball(j).
    
    while _time_left() > 2.0:
        try_remove_all()
        
        if _time_left() < 3.0:
            break
            
        if len(chosen) == 0:
            break
            
        i = rng.choice(list(chosen))
        b_i = ball[i]
        
        # Check if i is removable
        if int(np.min(covcount[b_i])) >= 2:
            covcount[b_i] -= 1
            chosen.discard(i)
            in_chosen[i] = False
            continue
            
        # Find words that would become uncovered if we remove i
        would_uncover = b_i[covcount[b_i] == 1]
        
        if len(would_uncover) == 0:
             covcount[b_i] -= 1
             chosen.discard(i)
             in_chosen[i] = False
             continue

        # Find candidates j
        # j must cover all w in would_uncover
        # j must not be in chosen
        
        w0 = would_uncover[0]
        cands = coverers[w0]
        
        valid_cands = []
        # Optimization: If len(would_uncover) == 1, we just filter cands by in_chosen
        if len(would_uncover) == 1:
            for j in cands:
                if not in_chosen[j]:
                    valid_cands.append(j)
        else:
            # For each j in cands, check if it covers all other w in would_uncover
            # and is not in chosen
            for j in cands:
                if in_chosen[j]:
                    continue
                ok = True
                for w in would_uncover[1:]:
                    # Check if j covers w
                    # j covers w iff w is in ball[j]
                    # Equivalent to j in coverers[w]
                    # coverers[w] is a list of size 17.
                    if j not in coverers[w]:
                        ok = False
                        break
                if ok:
                    valid_cands.append(j)
                    
        if valid_cands:
            j = rng.choice(valid_cands)
            b_j = ball[j]
            
            # Perform swap
            covcount[b_i] -= 1
            covcount[b_j] += 1
            chosen.discard(i)
            in_chosen[i] = False
            chosen.add(j)
            in_chosen[j] = True

    # Final cleanup
    try_remove_all()
    
    # Verify
    final_covered = np.zeros(N, dtype=bool)
    for i in chosen:
        final_covered[ball[i]] = True
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                chosen.add(i)
                final_covered[ball[i]] = True
                # Update counts if we care, but we are returning soon

    return [idx_to_digits(i) for i in chosen]
