"""Covering code K_3(8,1): minimize codewords so every word is within Hamming distance 1."""
import time
import random
import numpy as np

_DEADLINE = 110.0
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

    N = 3 ** n  # 6561
    ball_size = 1 + n * 2  # 17

    # Precompute ball using vectorized operations
    # For each index i, ball[i] contains i and all indices at Hamming distance 1
    # Represent indices in base-3 digits
    all_indices = np.arange(N, dtype=np.int32)
    digits = np.zeros((N, n), dtype=np.int32)
    for p in range(n):
        digits[:, p] = (all_indices // (3 ** p)) % 3

    # Precompute ball as a flat array with offsets (CSR-like)
    ball_offsets = np.zeros(N + 1, dtype=np.int32)
    ball_offsets[1:] = np.cumsum([ball_size] * N)
    ball_flat = np.empty(N * ball_size, dtype=np.int32)

    # For each position p, the two alternative values
    for p in range(n):
        pos_val = (all_indices // (3 ** p)) % 3
        # For each alternative value nv != pos_val, compute the new index
        for nv in range(3):
            mask = pos_val != nv
            if mask.any():
                new_idx = all_indices[mask].copy()
                # Replace digit p with nv
                digit_p_val = (new_idx // (3 ** p)) % 3
                new_idx -= digit_p_val * (3 ** p)
                new_idx += nv * (3 ** p)
                # These go into ball_flat at positions corresponding to ball[i][1+2p] and ball[i][2+2p]
                # We need to figure out which slot
                # For index i, the alternatives at position p are at offsets 1+2*p and 2+2*p
                # Let's just build it properly
                pass

    # Simpler: just build ball_flat directly
    ball_flat = np.empty(N * ball_size, dtype=np.int32)
    for i in range(N):
        base = i * ball_size
        ball_flat[base] = i
        d = digits[i]
        for p in range(n):
            orig = d[p]
            for nv in range(3):
                if nv != orig:
                    d[p] = nv
                    new_idx = int(np.sum(d * (3 ** np.arange(n))))
                    ball_flat[base + 1 + 2 * p + (1 if nv > orig else 0)] = new_idx
            d[p] = orig

    # Build coverers as lists (not sets) for faster intersection
    coverers = [[] for _ in range(N)]
    for i in range(N):
        base = i * ball_size
        for k in range(ball_size):
            w = ball_flat[base + k]
            coverers[w].append(i)

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

    def get_ball(i):
        base = i * ball_size
        return ball_flat[base:base + ball_size]

    def try_remove_all(chosen, covcount, rng):
        improved = True
        while improved and _time_left() > 5.0:
            improved = False
            order = list(chosen)
            rng.shuffle(order)
            for i in order:
                if i not in chosen:
                    continue
                b = get_ball(i)
                if np.min(covcount[b]) >= 2:
                    covcount[b] -= 1
                    chosen.discard(i)
                    improved = True

    def do_1_exchange(chosen, covcount, rng):
        """Try to replace one codeword with another to enable removals."""
        if not chosen:
            return False
        i = rng.choice(list(chosen))
        b_i = get_ball(i)
        would_uncover = b_i[covcount[b_i] == 1]
        if len(would_uncover) == 0:
            # Can just remove
            covcount[b_i] -= 1
            chosen.discard(i)
            return True
        # Find candidates that cover all would_uncover words
        if len(would_uncover) == 1:
            w = would_uncover[0]
            cands = [c for c in coverers[w] if c not in chosen]
        else:
            # Intersection of coverers for all would_uncover
            cands = set(coverers[would_uncover[0]])
            for w in would_uncover[1:]:
                cands &= set(coverers[w])
            cands = [c for c in cands if c not in chosen]
        if not cands:
            return False
        j = rng.choice(cands)
        b_j = get_ball(j)
        # Check validity: after swap, all words still covered
        # Remove i: covcount[b_i] -= 1
        # Add j: covcount[b_j] += 1
        # Words that were only covered by i (covcount==1) must be in b_j
        for w in would_uncover:
            if covcount[w] == 1 and w not in b_j:
                return False
        # Perform swap
        covcount[b_i] -= 1
        covcount[b_j] += 1
        chosen.discard(i)
        chosen.add(j)
        return True

    def do_2_exchange(chosen, covcount, rng):
        """Try to replace two codewords with two others."""
        if len(chosen) < 2:
            return False
        # Pick two random codewords to remove
        sample = rng.sample(list(chosen), 2)
        i1, i2 = sample
        b1 = get_ball(i1)
        b2 = get_ball(i2)
        # Words that would be uncovered
        would_uncover = set()
        for w in b1:
            if covcount[w] == 1:
                would_uncover.add(w)
        for w in b2:
            if covcount[w] == 1 and w not in b1:
                would_uncover.add(w)
        if not would_uncover:
            # Can just remove both
            covcount[b1] -= 1
            covcount[b2] -= 1
            chosen.discard(i1)
            chosen.discard(i2)
            return True
        # Find two candidates that cover all would_uncover words
        # This is expensive; limit to small would_uncover
        if len(would_uncover) > 6:
            return False
        # Get candidate pool: union of coverers for would_uncover words
        cand_pool = set()
        for w in would_uncover:
            cand_pool.update(coverers[w])
        cand_pool = [c for c in cand_pool if c not in chosen]
        if len(cand_pool) < 2:
            return False
        # Try random pairs
        for _ in range(20):
            j1, j2 = rng.sample(cand_pool, 2)
            bj1 = get_ball(j1)
            bj2 = get_ball(j2)
            # Check if all would_uncover words are covered by j1 or j2
            ok = True
            for w in would_uncover:
                if w not in bj1 and w not in bj2:
                    ok = False
                    break
            if ok:
                # Check that removing i1,i2 and adding j1,j2 keeps coverage
                # Words in b1 or b2 with covcount==1 must be in bj1 or bj2
                for w in b1:
                    if covcount[w] == 1 and w not in bj1 and w not in bj2:
                        ok = False
                        break
                if ok:
                    for w in b2:
                        if covcount[w] == 1 and w not in bj1 and w not in bj2:
                            ok = False
                            break
                if ok:
                    covcount[b1] -= 1
                    covcount[b2] -= 1
                    covcount[bj1] += 1
                    covcount[bj2] += 1
                    chosen.discard(i1)
                    chosen.discard(i2)
                    chosen.add(j1)
                    chosen.add(j2)
                    return True
        return False

    best_size = N
    best_result = None

    for seed in [12345, 54321, 99999, 11111, 77777]:
        if _time_left() < 30.0:
            break
        rng = random.Random(seed)
        
        # Initial construction: structured seed + greedy
        # Tetracode seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_set = set()
        for h in tetracode:
            for t in tetracode:
                w = h + t
                seed_set.add(digits_to_idx(w))
        
        chosen = set(seed_set)
        covcount = np.zeros(N, dtype=np.int32)
        covered = np.zeros(N, dtype=bool)
        for i in seed_set:
            b = get_ball(i)
            covcount[b] += 1
            covered[b] = True

        # Greedy fill: pick codeword with max gain
        while not covered.all() and _time_left() > 25.0:
            # Find uncovered words
            unc = np.where(~covered)[0]
            if len(unc) == 0:
                break
            # For each uncovered word, pick a coverer with max gain
            # To speed up, pick a few random uncovered words and take best coverer
            gains = {}
            for _ in range(min(5, len(unc))):
                w = unc[rng.randint(0, len(unc) - 1)]
                for c in coverers[w]:
                    if c in chosen:
                        continue
                    b = get_ball(c)
                    g = int(np.sum(~covered[b]))
                    if g > 0:
                        if c not in gains or g > gains[c]:
                            gains[c] = g
            if not gains:
                # Fallback: add any coverer of a random uncovered word
                w = unc[0]
                c = coverers[w][0]
                chosen.add(c)
                covcount[get_ball(c)] += 1
                covered[get_ball(c)] = True
                continue
            best_c = max(gains, key=gains.get)
            chosen.add(best_c)
            b = get_ball(best_c)
            covcount[b] += 1
            covered[b] = True

        # Ensure valid
        if not covered.all():
            for i in range(N):
                if not covered[i]:
                    chosen.add(i)
                    covcount[get_ball(i)] += 1
                    covered[get_ball(i)] = True

        # Local search
        for iteration in range(500):
            if _time_left() < 20.0:
                break
            try_remove_all(chosen, covcount, rng)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = set(chosen)
            # Try exchanges
            moved = False
            for _ in range(10):
                if do_1_exchange(chosen, covcount, rng):
                    moved = True
                    break
            if not moved:
                for _ in range(5):
                    if do_2_exchange(chosen, covcount, rng):
                        moved = True
                        break
            try_remove_all(chosen, covcount, rng)

        if len(chosen) < best_size:
            best_size = len(chosen)
            best_result = set(chosen)

    # Final answer
    if best_result is None:
        # Fallback: full space
        return [idx_to_digits(i) for i in range(N)]
    
    # Verify
    final_covered = np.zeros(N, dtype=bool)
    for i in best_result:
        final_covered[get_ball(i)] = True
    if not final_covered.all():
        for i in range(N):
            if not final_covered[i]:
                best_result.add(i)
                final_covered[get_ball(i)] = True

    return [idx_to_digits(i) for i in best_result]
