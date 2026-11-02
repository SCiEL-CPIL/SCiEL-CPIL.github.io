"""Greedy set-cover for K_q(n, radius) with a time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _ball_indices(word, q, n):
    """Return the set of indices (as ints in 0..q^n-1) within Hamming distance 1 of word."""
    base = 0
    for i in range(n):
        base = base * q + word[i]
    ball = [base]
    for i in range(n):
        pw = q ** (n - 1 - i)
        w_i = word[i]
        for v in range(q):
            if v != w_i:
                ball.append(base + (v - w_i) * pw)
    return ball


def _ball_indices_fast(word, q, n, pows):
    """Fast ball: word as int index + offsets per position."""
    base = 0
    for i in range(n):
        base = base * q + word[i]
    ball = [base]
    for i in range(n):
        pw = pows[i]
        w_i = word[i]
        for v in range(q):
            if v != w_i:
                ball.append(base + (v - w_i) * pw)
    return ball


def _int_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (kill at 150s)

    total = q ** n
    pows = [q ** (n - 1 - i) for i in range(n)]

    # --- Strong initial seed: direct sum of ternary tetracode (for q=3) ---
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        seed = [h + list(tail)
                for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        seed = []

    # --- Convert to index-based representation ---
    def to_idx(word):
        b = 0
        for x in word:
            b = b * q + x
        return b

    # covered: bitset as python int for speed (total up to 729 fits easily)
    # For general n, total may be large; use set of ints.
    covered = set()
    code_indices = []
    for w in seed:
        idx = to_idx(w)
        code_indices.append(idx)
        for b in _ball_indices(w, q, n):
            covered.add(b)

    # Precompute all balls for candidate words? Too many. Instead, generate candidates on the fly.
    # Greedy: repeatedly pick the candidate (any word) that covers the most currently-uncovered words.
    # But the space is huge. Restrict candidates to words that are themselves uncovered (they need covering,
    # and placing a codeword there covers itself + neighbors). Actually a good heuristic: pick the uncovered
    # word with the largest "new coverage" among its ball.

    # To keep it fast, we precompute the ball for every word in the space (total = q^n, small for n=6,q=3).
    all_balls = {}
    for idx in range(total):
        w = _int_to_word(idx, q, n)
        all_balls[idx] = _ball_indices_fast(w, q, n, pows)

    # uncovered set
    uncovered = set(range(total)) - covered

    # We'll iterate: for each uncovered word, compute how many of its ball members are still uncovered
    # (new coverage). Pick the max, add it as a codeword, update.
    # To speed up, maintain for each word its current "gain" = number of uncovered in its ball.
    # When we add a codeword c, for each word u whose ball contains c, gain[u] decreases by 1 if c was uncovered.
    # This is a standard greedy set cover with lazy updates.

    # Initialize gain
    gain = {}
    for idx in range(total):
        ball = all_balls[idx]
        g = 0
        for b in ball:
            if b in uncovered:
                g += 1
        gain[idx] = g

    # Use a heap for max-gain selection (lazy).
    import heapq
    # heap of (-gain, idx)
    heap = [(-gain[idx], idx) for idx in range(total)]
    heapq.heapify(heap)

    best_code = list(code_indices)
    best_count = len(code_indices)

    while uncovered:
        if time.time() > deadline:
            break
        # pop until we find a valid (current) max
        while heap:
            neg_g, idx = heap[0]
            g = -neg_g
            if g == gain[idx] and gain[idx] > 0:
                break
            heapq.heappop(heap)
        if not heap:
            break
        neg_g, idx = heapq.heappop(heap)
        g = -neg_g
        if g <= 0:
            break

        # Add idx as a codeword
        code_indices.append(idx)
        ball = all_balls[idx]
        # Update uncovered and gains
        for b in ball:
            if b in uncovered:
                uncovered.discard(b)
                # b was uncovered; for every word u whose ball contains b, gain[u] -= 1
                # We don't have reverse mapping. Instead, lazily: just recompute gain when popped.
                # But we need to update gains for correctness of heap.
                # Simpler: mark and recompute lazily. We'll recompute gain[idx] when popped if stale.
                pass

        # Lazy recompute: we'll just recompute gain for all affected on pop.
        # To keep heap consistent, we push updated gains for words whose ball we changed.
        # Words affected: those u where idx in all_balls[u]. That's the "reverse ball".
        # Precompute reverse: rev[b] = list of u such that b in all_balls[u]. Since ball is symmetric
        # (distance <=1 is symmetric), rev[b] = all_balls[b].
        for b in ball:
            for u in all_balls[b]:
                # recompute gain[u] lazily? Just decrement if b was newly covered.
                # We know b was uncovered before this step (if it was in uncovered).
                if b in all_balls[idx] and True:
                    # We can't easily know if b was uncovered. Use a flag.
                    pass

        # This lazy approach is getting complex. Given small total (729), just recompute gains fully each step.
        # 729 * (ball size ~ 1+6*2=13) = ~9500 ops per step. ~100 steps = 1e6 ops. Fast.
        # Recompute all gains from scratch.
        for u in range(total):
            ball_u = all_balls[u]
            g2 = 0
            for b in ball_u:
                if b in uncovered:
                    g2 += 1
            gain[u] = g2
        # Rebuild heap
        heap = [(-gain[idx], idx) for idx in range(total)]
        heapq.heapify(heap)

        if len(code_indices) < best_count:
            best_code = list(code_indices)
            best_count = len(code_indices)

    # Convert back to words
    result = [_int_to_word(idx, q, n) for idx in best_code]
    return result
# EVOLVE-BLOCK-END
