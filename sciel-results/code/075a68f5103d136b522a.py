import numpy as np
import time
import itertools

def construct(q, n, radius):
    if not (q == 3 and n == 8 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    time_limit = 120.0
    safety_margin = 10.0
    deadline = start_time + time_limit - safety_margin

    N = q ** n  # 6561

    # Precompute all words as a numpy array: shape (N, n)
    # Word index i -> digits
    all_words = np.zeros((N, n), dtype=np.int32)
    for i in range(N):
        val = i
        for k in range(n - 1, -1, -1):
            all_words[i, k] = val % q
            val //= q

    # Precompute balls: for each word, the set of words within Hamming distance 1
    # ball[i] is a list of indices j such that dist(all_words[i], all_words[j]) <= 1
    ball_size = 1 + n * (q - 1)  # 1 + 8*2 = 17
    balls = np.zeros((N, ball_size), dtype=np.int32)
    for i in range(N):
        w = all_words[i]
        balls[i, 0] = i
        idx = 1
        for k in range(n):
            orig = w[k]
            for v in range(q):
                if v == orig:
                    continue
                # Compute index of word with digit v at position k
                # all_words is stored MSB first, so position k has weight q^(n-1-k)
                delta = (v - orig) * (q ** (n - 1 - k))
                j = i + delta
                balls[i, idx] = j
                idx += 1

    # Phase 1: Greedy Set Cover
    # uncovered[i] = 1 if word i is not yet covered
    uncovered = np.ones(N, dtype=np.int32)
    covered_count = N
    codewords = []

    while covered_count > 0:
        if time.time() > deadline:
            break

        # For each candidate c, count how many uncovered words it covers
        # score[c] = sum of uncovered[j] for j in balls[c]
        scores = np.zeros(N, dtype=np.int32)
        for c in range(N):
            # balls[c] has ball_size entries
            scores[c] = np.sum(uncovered[balls[c]])

        best = np.argmax(scores)
        if scores[best] == 0:
            break

        codewords.append(best)
        new_uncovered = uncovered[balls[best]]
        newly_covered = np.sum(new_uncovered)
        uncovered[balls[best]] = 0
        covered_count -= newly_covered

    # Phase 2: Greedy Removal (multiple passes)
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1

    for _ in range(10):
        if time.time() > deadline:
            break
        new_codewords = []
        changed = False
        for c in codewords:
            counts[balls[c]] -= 1
            if np.min(counts) >= 1:
                changed = True
            else:
                counts[balls[c]] += 1
                new_codewords.append(c)
        codewords = new_codewords
        if not changed:
            break

    # Phase 3: Local Search - try to replace codewords with better ones
    # For each codeword, try removing it and see if we can replace it
    # with a codeword that covers the now-exposed words
    for iteration in range(50):
        if time.time() > deadline:
            break

        improved = False
        new_codewords = []

        for c in codewords:
            counts[balls[c]] -= 1
            if np.min(counts) >= 1:
                # Can remove without replacement
                improved = True
                continue

            # Find words that would become uncovered
            exposed = np.where(counts == 0)[0]
            exposed_set = set(exposed.tolist())

            # Find the best replacement: a word that covers the most exposed words
            # and also covers some already-covered words (for future flexibility)
            best_score = 0
            best_repl = -1

            # Only check candidates that cover at least one exposed word
            for cand in range(N):
                if cand == c:
                    continue
                # Check if cand covers any exposed words
                cand_ball = balls[cand]
                covers_exposed = 0
                for j in cand_ball:
                    if j in exposed_set:
                        covers_exposed += 1
                if covers_exposed == 0:
                    continue

                # Score: prioritize covering all exposed words, then minimize overlap with existing
                if covers_exposed >= len(exposed):
                    # This candidate covers all exposed words
                    # Score by how few additional words it covers (less overlap is better)
                    overlap = np.sum(counts[cand_ball] > 0)
                    score = covers_exposed * 1000 - overlap
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_repl = cand

            if best_repl >= 0:
                counts[balls[best_repl]] += 1
                new_codewords.append(best_repl)
                improved = True
            else:
                counts[balls[c]] += 1
                new_codewords.append(c)

        codewords = new_codewords

        # Recount
        counts = np.zeros(N, dtype=np.int32)
        for c in codewords:
            counts[balls[c]] += 1

        if not improved:
            break

    # Phase 4: Final removal pass
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1

    for _ in range(20):
        if time.time() > deadline:
            break
        new_codewords = []
        changed = False
        for c in codewords:
            counts[balls[c]] -= 1
            if np.min(counts) >= 1:
                changed = True
            else:
                counts[balls[c]] += 1
                new_codewords.append(c)
        codewords = new_codewords
        if not changed:
            break

    # Convert indices to digit lists
    result = []
    for c in codewords:
        result.append(all_words[c].tolist())

    return result
