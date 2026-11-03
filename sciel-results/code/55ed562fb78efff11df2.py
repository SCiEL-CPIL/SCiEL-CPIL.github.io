import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    if not (q == 3 and n == 8 and radius == 1):
        return _generic_construct(q, n, radius, deadline)

    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute balls
    digits = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        v = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = v % q
            v //= q

    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total)
    idx = 1
    for pos in range(n):
        for delta in range(1, q):
            new_digits = digits.copy()
            new_digits[:, pos] = (digits[:, pos] + delta) % q
            vals = np.zeros(total, dtype=np.int32)
            for p in range(n):
                vals = vals * q + new_digits[:, p]
            balls[:, idx] = vals
            idx += 1

    # Tetracode: 27 codewords of length 4 covering all 81 words with radius 1
    # The tetracode is a [4,2,2] code (actually a covering code).
    # Let's construct it explicitly.
    # A simple (81, 27, 1) covering code for q=3, n=4.
    # One construction: take all words (a,b,c,d) such that a+b+c+d = 0 mod 3 and a+b = c+d = 0 mod 3? No.
    # Let's use the standard tetracode:
    # The tetracode is the set of 27 words in {0,1,2}^4.
    # Actually, K_3(4,1) = 27.
    # A simple construction:
    # For x in 0..2, y in 0..2:
    #   w1 = (x, y, x+y, x+2y)
    #   w2 = (x, y, x+2y, x+y)
    #   w3 = (x, y, 2x+y, 2x+2y)
    #   ... wait, that's 27 words? 3*3*3 = 27.
    # Let's verify this is a covering code.
    # Actually, the tetracode is the dual of the [4,2,2] Hamming code? No.
    # Let's just use a known list or a simple greedy for n=4 to get 27 words, then use them.
    # But we need a fast deterministic construction.
    # Let's use the following:
    # The tetracode can be defined as the set of all (a, b, c, d) such that:
    # a + b + c + d = 0 (mod 3)
    # a + 2b + 3c + 4d = 0 (mod 3) -> a + 2b + c + d = 0 (mod 3)
    # This is a [4,2,2] linear code. Size 3^2 = 9. Not 27.
    # Wait, K_3(4,1) = 27. The Hamming bound is 81/13 = 6.23. So 27 is much larger.
    # Ah, the "tetracode" is a (81, 27, 1) covering code.
    # Let's generate it by greedy for n=4, q=3, r=1. It's small, so it's fast.
    def _construct_small(q, n, r):
        total_small = q ** n
        bs = 1 + n * (q - 1)
        digs = np.zeros((total_small, n), dtype=np.int32)
        for i in range(total_small):
            v = i
            for pos in range(n - 1, -1, -1):
                digs[i, pos] = v % q
                v //= q
        b = np.zeros((total_small, bs), dtype=np.int32)
        b[:, 0] = np.arange(total_small)
        idx = 1
        for pos in range(n):
            for delta in range(1, q):
                nd = digs.copy()
                nd[:, pos] = (digs[:, pos] + delta) % q
                vals = np.zeros(total_small, dtype=np.int32)
                for p in range(n):
                    vals = vals * q + nd[:, p]
                b[:, idx] = vals
                idx += 1
        covered = np.zeros(total_small, dtype=np.bool_)
        cc = 0
        cws = []
        while cc < total_small:
            gains = bs - covered[b].sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            cws.append(best)
            newly = ~covered[b[best]]
            covered[b[best]] = True
            cc += int(newly.sum())
        return cws

    tetra = _construct_small(3, 4, 1)
    tetra_set = set(tetra)

    # Construct 729 codewords: (t1, t2) where t1, t2 in tetra
    # t1 is first 4 digits, t2 is last 4 digits
    code = []
    for t1 in tetra:
        for t2 in tetra:
            # t1 is int, convert to 4 digits
            d1 = [t1 % 3, (t1 // 3) % 3, (t1 // 9) % 3, (t1 // 27) % 3]
            d2 = [t2 % 3, (t2 // 3) % 3, (t2 // 9) % 3, (t2 // 27) % 3]
            word = d1 + d2
            code.append(to_int(word))

    # Now optimize
    code_set = set(code)
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in code:
        cov_count[balls[cw]] += 1

    # 1-in/1-out local search
    # For each codeword, try to remove it. If it exposes points, find a new word that covers them.
    # To speed up, we can precompute for each word, the set of words that cover it? No, that's the ball.
    # We will iterate.

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle order? No, deterministic is fine for now, but maybe random helps escape local minima.
        # Let's stick to deterministic first.
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            cw = code[i]
            # Check if removable
            exposed = balls[cw]
            if (cov_count[exposed] > 1).all():
                cov_count[exposed] -= 1
                code.pop(i)
                code_set.discard(cw)
                improved = True
                break

            # Try swap
            # Points that will become uncovered if cw is removed
            newly_exposed = exposed[cov_count[exposed] == 1]
            if len(newly_exposed) == 0:
                continue

            # Find candidate replacement
            # A replacement w must cover all newly_exposed points.
            # So w must be in the intersection of balls[x] for x in newly_exposed.
            # If len(newly_exposed) is large, intersection is likely empty.
            if len(newly_exposed) > 3:
                continue

            # Compute intersection
            # Start with neighbors of first exposed point
            candidates = set(balls[newly_exposed[0]].tolist())
            # Remove current code words (we want a NEW word, or we could allow same word? No, that's removal)
            # Actually, if we swap, the new word must be different from the old one?
            # If new word == old word, it's removal.
            # If new word is in code, it's a duplicate, which is invalid.
            # So new word must NOT be in code_set.
            candidates -= code_set

            for x in newly_exposed[1:]:
                if not candidates:
                    break
                candidates &= set(balls[x].tolist())

            if not candidates:
                continue

            # Pick best candidate
            # Gain = number of points in ball[w] that are currently covered ONLY by cw?
            # No, Gain = number of points in ball[w] that are currently uncovered (0) or covered by cw (1) AND not covered by others?
            # Actually, we just want to maximize the number of points in ball[w] that are currently covered by cw (cov_count==1)
            # because those are the ones we are "saving" from being uncovered.
            # Points already covered by others (cov_count > 1) don't benefit from the swap.
            # Points uncovered (cov_count == 0) are bad, we don't want to add a word that covers uncovered points?
            # Wait, if we add w, it covers ball[w].
            # The points in newly_exposed MUST be covered by w.
            # Other points in ball[w]:
            #   If cov_count > 1: no change in coverage status (still covered).
            #   If cov_count == 1 (by cw): becomes covered by w (still covered).
            #   If cov_count == 0: becomes covered by w (newly covered).
            # So we want to maximize the number of points in ball[w] with cov_count == 0?
            # Or minimize the "cost"?
            # Actually, the size of the code stays the same (1 in, 1 out).
            # So we just want to ensure the code remains a covering.
            # Any candidate that covers newly_exposed works.
            # To improve future removals, we might want a word that is "redundant" or covers many already covered points?
            # No, we want to reduce the code size.
            # A good swap is one that makes the new word easily removable later?
            # Or one that covers many points with high cov_count?
            # Let's just pick the one that maximizes the number of points in ball[w] with cov_count == 1 (by cw).
            # This "consolidates" coverage.
            best_w = -1
            best_score = -1
            for w in candidates:
                w = int(w)
                # Score: how many points in ball[w] are covered ONLY by cw?
                # These are the points that would be exposed if we removed cw, but are covered by w.
                # We already know w covers newly_exposed.
                # Let's just pick the first one? Or the one with max overlap with cw's ball?
                # Overlap = intersection(balls[cw], balls[w])
                # If overlap is large, w is "close" to cw.
                # Let's use a simple heuristic: max number of points in ball[w] with cov_count == 1.
                # This requires checking cov_count[balls[w]].
                # cov_count[balls[w]] == 1 means covered only by one word.
                # But we don't know WHICH word.
                # We know that if we remove cw, points in newly_exposed become 0.
                # If we add w, they become 1 (by w).
                # What about other points in ball[w]?
                # If cov_count[x] == 1 and x != cw, and we add w, cov_count[x] becomes 2.
                # This is good for redundancy.
                # Let's just pick the candidate that maximizes the sum of cov_count[balls[w]].
                # Higher sum means more overlap with existing code, which might allow removals?
                # Actually, we want to MINIMIZE the new "exposed" risk?
                # Let's just pick the first candidate. It's fast.
                best_w = w
                break

            if best_w != -1:
                # Perform swap
                cov_count[balls[cw]] -= 1
                cov_count[balls[best_w]] += 1
                code[i] = best_w
                code_set.add(best_w)
                code_set.discard(cw)
                improved = True
                break

    # Final redundancy removal
    cov_count = np.zeros(total, dtype=np.int32)
    for cw in code:
        cov_count[balls[cw]] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            cw = code[i]
            if (cov_count[balls[cw]] > 1).all():
                cov_count[balls[cw]] -= 1
                code.pop(i)
                changed = True
                break

    result = [to_word(w) for w in code]
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result

def _generic_construct(q, n, radius, deadline):
    start_time = time.time()
    total = q ** n

    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    ball_size = 1
    for pos in range(n):
        ball_size += (q - 1)

    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = np.arange(total)
    idx = 1
    digits = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        v = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = v % q
            v //= q

    for pos in range(n):
        for delta in range(1, q):
            new_digits = digits.copy()
            new_digits[:, pos] = (digits[:, pos] + delta) % q
            vals = np.zeros(total, dtype=np.int32)
            for p in range(n):
                vals = vals * q + new_digits[:, p]
            balls[:, idx] = vals
            idx += 1

    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []

    while covered_count < total:
        if time.time() > deadline - 5:
            break
        covered_in_balls = covered[balls]
        gains = ball_size - covered_in_balls.sum(axis=1)
        best_word = int(np.argmax(gains))
        best_new = int(gains[best_word])
        if best_new == 0:
            break
        codewords.append(best_word)
        ball_points = balls[best_word]
        newly_covered = ~covered[ball_points]
        if newly_covered.any():
            covered[ball_points] = True
            covered_count += int(newly_covered.sum())

    cov_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        cov_count[balls[cw]] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            cw = codewords[i]
            if (cov_count[balls[cw]] > 1).all():
                cov_count[balls[cw]] -= 1
                codewords.pop(i)
                changed = True
                break

    result = [to_word(w) for w in codewords]
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    return unique_result
