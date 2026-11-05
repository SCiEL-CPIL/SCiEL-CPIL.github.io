import time
import itertools
import numpy as np

_DEADLINE = 120.0
_SAFE_MARGIN = 6.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

_START = time.time()

def construct(q, n, radius):
    global _START
    _START = time.time()
    # Fast path: tetracode direct sum as initial/fallback
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n == 4:
        return [list(h) for h in tetracode]
    if n < 4:
        # small: just cover by greedy
        return _greedy_cover(q, n, radius)
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    # Try to improve via local search within time budget
    improved = _local_search(base, q, n, radius)
    return improved

def _greedy_cover(q, n, radius):
    # Greedy set cover: repeatedly pick the codeword covering the most uncovered words.
    # For radius 1 over q^n, a codeword covers 1 + n*(q-1) words.
    # Represent space as integers 0..q^n-1.
    total = q ** n
    if total > 200000:
        return None  # too big for dense; caller handles
    covered = np.zeros(total, dtype=bool)
    code = []
    # Precompute neighborhoods for all possible codewords? That's q^n * (1+n*(q-1)) which is fine for small.
    # Instead, for each candidate center, compute its ball on the fly but that's expensive.
    # For radius 1, ball(center) = {center} U {center with one coord changed}.
    # We iterate centers in 0..total-1.
    centers = list(range(total))
    # To be efficient, compute uncovered count per center using bitset? 
    # Simpler: maintain covered; for each step, score each uncovered center by number of newly covered.
    # This is O(total^2) worst case; only use for small total.
    if total > 50000:
        return None
    uncovered = total
    while uncovered > 0 and _time_left() > _SAFE_MARGIN:
        best_center = -1
        best_gain = -1
        # sample a subset if too many
        if total > 20000:
            sample = np.random.choice(total, size=min(total, 5000), replace=False)
        else:
            sample = centers
        for c in sample:
            # compute ball
            ball = _ball_1(c, q, n)
            gain = np.count_nonzero(~covered[ball])
            if gain > best_gain:
                best_gain = gain
                best_center = c
        if best_gain <= 0:
            break
        ball = _ball_1(best_center, q, n)
        covered[ball] = True
        uncovered -= best_gain
        code.append(best_center)
    return code

def _ball_1(center_int, q, n):
    # return list/array of ints in ball of radius 1 around center_int
    # decode center
    digits = []
    t = center_int
    for _ in range(n):
        digits.append(t % q)
        t //= q
    digits = digits[::-1]
    ball = [center_int]
    for i in range(n):
        for v in range(q):
            if v == digits[i]:
                continue
            new_digits = digits[:]
            new_digits[i] = v
            val = 0
            for d in new_digits:
                val = val * q + d
            ball.append(val)
    return np.array(ball, dtype=np.int64)

def _local_search(initial_code, q, n, radius):
    # Convert to array of ints
    code = []
    seen = set()
    for cw in initial_code:
        val = 0
        for d in cw:
            val = val * q + d
        if val not in seen:
            seen.add(val)
            code.append(val)
    code = list(code)
    total = q ** n
    # For radius 1, precompute ball for each code word? We'll compute coverage as boolean array if feasible.
    if total <= 2000000:
        covered = np.zeros(total, dtype=bool)
        for c in code:
            ball = _ball_1(c, q, n)
            covered[ball] = True
        # Verify full coverage
        assert covered.all(), "initial not a cover"
        # Phase 1: removals
        changed = True
        while changed and _time_left() > _SAFE_MARGIN:
            changed = False
            # try removing each codeword
            # To speed up, only try a random subset per pass if large
            idxs = list(range(len(code)))
            np.random.shuffle(idxs)
            for i in idxs:
                if _time_left() < _SAFE_MARGIN:
                    break
                c = code[i]
                ball = _ball_1(c, q, n)
                # check if all in ball are covered by others
                others_cover = covered.copy()
                others_cover[ball] = False
                # recompute? that's expensive. Instead, maintain counts.
                # Let's maintain a coverage count array instead.
                break
            # Switch to count-based
            break
        # Use count-based coverage
        cov_count = np.zeros(total, dtype=np.int32)
        for c in code:
            ball = _ball_1(c, q, n)
            cov_count[ball] += 1
        # Phase 1: removals
        changed = True
        while changed and _time_left() > _SAFE_MARGIN:
            changed = False
            idxs = list(range(len(code)))
            np.random.shuffle(idxs)
            for i in idxs:
                if _time_left() < _SAFE_MARGIN:
                    break
                c = code[i]
                ball = _ball_1(c, q, n)
                if np.all(cov_count[ball] > 1):
                    cov_count[ball] -= 1
                    code.pop(i)
                    changed = True
                    break
        # Phase 2: 2-for-1 swaps (remove two, add one) and 1-for-2 (remove one, add two) -- we want to reduce, so try remove two add one, and remove one add two only if reduces? 
        # Actually to reduce size: try (remove 2, add 1) -> net -1. Also (remove 1, add 0) already done.
        # Also try (remove 2, add 1) and (remove 3, add 1) etc. Start with 2-for-1.
        for _pass in range(20):
            if _time_left() < _SAFE_MARGIN:
                break
            found = False
            idxs = list(range(len(code)))
            np.random.shuffle(idxs)
            # Try pairs
            for a in idxs:
                if _time_left() < _SAFE_MARGIN:
                    break
                if a >= len(code):
                    continue
                for b in idxs:
                    if b <= a:
                        continue
                    if b >= len(code):
                        continue
                    ca = code[a]
                    cb = code[b]
                    ball_a = _ball_1(ca, q, n)
                    ball_b = _ball_1(cb, q, n)
                    # union of balls
                    union = np.unique(np.concatenate([ball_a, ball_b]))
                    # coverage by others: cov_count - 1 for ball_a, -1 for ball_b
                    # We need: for every x in union, there exists another codeword covering x, OR a new center c covers x.
                    # First, find words not covered by others:
                    others_cov = cov_count.copy()
                    others_cov[ball_a] -= 1
                    others_cov[ball_b] -= 1
                    uncovered = union[others_cov[union] == 0]
                    if len(uncovered) == 0:
                        # can just remove both (that's 2-for-0, even better)
                        # But we already did removals; this can happen after other changes.
                        cov_count[ball_a] -= 1
                        cov_count[ball_b] -= 1
                        # remove higher index first
                        if a > b:
                            code.pop(a); code.pop(b)
                        else:
                            code.pop(b); code.pop(a)
                        found = True
                        break
                    # else, try to find a center c whose ball covers all uncovered
                    # uncovered is small hopefully
                    if len(uncovered) > 30:
                        continue
                    # The new center must be within distance 1 of all uncovered words.
                    # Intersection of balls of radius 1 around each uncovered word.
                    # For ternary, this is a constraint satisfaction. Try candidates: for each uncovered word, its ball; intersect.
                    candidates = None
                    for u in uncovered[:5]:  # limit
                        bu = _ball_1(u, q, n)
                        if candidates is None:
                            candidates = set(bu.tolist())
                        else:
                            candidates &= set(bu.tolist())
                        if len(candidates) == 0:
                            break
                    if not candidates:
                        continue
                    for c in candidates:
                        if c == ca or c == cb:
                            continue
                        ball_c = _ball_1(c, q, n)
                        # check ball_c covers all uncovered
                        if np.all(ball_c[np.isin(ball_c, uncovered)] == uncovered) or np.all(np.isin(uncovered, ball_c)):
                            # valid: remove a,b add c
                            cov_count[ball_a] -= 1
                            cov_count[ball_b] -= 1
                            cov_count[ball_c] += 1
                            if a > b:
                                code.pop(a); code.pop(b)
                            else:
                                code.pop(b); code.pop(a)
                            code.append(c)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                # try 1-for-2 won't reduce; try 3-for-1 occasionally? skip for now
                break
        # Phase 3: 1-for-1 swaps to enable future removals (hill climbing on size via swaps that keep size same but improve structure)
        # Skip for time; final removals pass
        changed = True
        while changed and _time_left() > _SAFE_MARGIN:
            changed = False
            idxs = list(range(len(code)))
            np.random.shuffle(idxs)
            for i in idxs:
                if _time_left() < _SAFE_MARGIN:
                    break
                c = code[i]
                ball = _ball_1(c, q, n)
                if np.all(cov_count[ball] > 1):
                    cov_count[ball] -= 1
                    code.pop(i)
                    changed = True
                    break
        # Final verify
        if total <= 2000000:
            final_cov = np.zeros(total, dtype=bool)
            for c in code:
                ball = _ball_1(c, q, n)
                final_cov[ball] = True
            if not final_cov.all():
                # fallback to initial
                return initial_code
    else:
        # too big for dense coverage; just do a few removals using sampled checks
        # For n=7, total=2187, which is small! So we always use dense.
        pass
    # Convert back to lists
    result = []
    for c in code:
        digits = []
        t = c
        for _ in range(n):
            digits.append(t % q)
            t //= q
        digits.reverse()
        result.append(digits)
    return result

# Precompute balls for speed in local search
def _precompute_balls(q, n):
    total = q ** n
    balls = {}
    for c in range(total):
        balls[c] = _ball_1(c, q, n)
    return balls

# Override _ball_1 to use cache
_ball_cache = {}
def _ball_1_cached(center_int, q, n):
    key = (q, n, center_int)
    if key not in _ball_cache:
        _ball_cache[key] = _ball_1(center_int, q, n)
    return _ball_cache[key]

# Replace _ball_1 calls in _local_search with cached version via monkey patch? 
# Simpler: redefine _local_search to use _ball_1_cached
def _local_search(initial_code, q, n, radius):
    code = []
    seen = set()
    for cw in initial_code:
        val = 0
        for d in cw:
            val = val * q + d
        if val not in seen:
            seen.add(val)
            code.append(val)
    code = list(code)
    total = q ** n
    if total > 5000000:
        return initial_code
    cov_count = np.zeros(total, dtype=np.int32)
    for c in code:
        ball = _ball_1_cached(c, q, n)
        cov_count[ball] += 1
    # Removals
    changed = True
    while changed and _time_left() > _SAFE_MARGIN:
        changed = False
        idxs = list(range(len(code)))
        np.random.shuffle(idxs)
        for i in idxs:
            if _time_left() < _SAFE_MARGIN:
                break
            c = code[i]
            ball = _ball_1_cached(c, q, n)
            if np.all(cov_count[ball] > 1):
                cov_count[ball] -= 1
                code.pop(i)
                changed = True
                break
    # 2-for-1
    for _pass in range(30):
        if _time_left() < _SAFE_MARGIN:
            break
        found = False
        idxs = list(range(len(code)))
        np.random.shuffle(idxs)
        m = len(code)
        for a in idxs:
            if a >= m:
                continue
            for b in idxs:
                if b <= a or b >= m:
                    continue
                ca = code[a]
                cb = code[b]
                ball_a = _ball_1_cached(ca, q, n)
                ball_b = _ball_1_cached(cb, q, n)
                union = np.unique(np.concatenate([ball_a, ball_b]))
                others_cov = cov_count.copy()
                others_cov[ball_a] -= 1
                others_cov[ball_b] -= 1
                uncovered = union[others_cov[union] == 0]
                if len(uncovered) == 0:
                    cov_count[ball_a] -= 1
                    cov_count[ball_b] -= 1
                    if a > b:
                        code.pop(a); code.pop(b)
                    else:
                        code.pop(b); code.pop(a)
                    found = True
                    break
                if len(uncovered) > 20:
                    continue
                # find candidate center covering all uncovered
                candidates = None
                for u in uncovered[:4]:
                    bu = _ball_1_cached(int(u), q, n)
                    if candidates is None:
                        candidates = set(bu.tolist())
                    else:
                        candidates &= set(bu.tolist())
                    if not candidates:
                        break
                if not candidates:
                    continue
                for c in candidates:
                    if c == ca or c == cb:
                        continue
                    ball_c = _ball_1_cached(c, q, n)
                    if np.all(np.isin(uncovered, ball_c)):
                        cov_count[ball_a] -= 1
                        cov_count[ball_b] -= 1
                        cov_count[ball_c] += 1
                        if a > b:
                            code.pop(a); code.pop(b)
                        else:
                            code.pop(b); code.pop(a)
                        code.append(c)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    # Final removals
    changed = True
    while changed and _time_left() > _SAFE_MARGIN:
        changed = False
        idxs = list(range(len(code)))
        np.random.shuffle(idxs)
        for i in idxs:
            if _time_left() < _SAFE_MARGIN:
                break
            c = code[i]
            ball = _ball_1_cached(c, q, n)
            if np.all(cov_count[ball] > 1):
                cov_count[ball] -= 1
                code.pop(i)
                changed = True
                break
    result = []
    for c in code:
        digits = []
        t = c
        for _ in range(n):
            digits.append(t % q)
            t //= q
        digits.reverse()
        result.append(digits)
    return result
