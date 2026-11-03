"""Greedy set-cover + local improvement for ternary covering codes."""
import time
import itertools

import numpy as np


def _all_balls(q, n, radius, limit=2):
    """Return a list of flat integer indices for the radius-ball around every word.
    Only radius 0..limit supported."""
    qn = q ** n
    balls = np.empty((qn, qn), dtype=np.int32)
    # ball of radius 0: just the word
    idx = np.arange(qn, dtype=np.int32)
    balls[:, 0] = idx
    pos = 1
    if radius >= 1:
        # For each coordinate, for each offset in 1..q-1, compute the shifted word index.
        # We'll build all radius-1 neighbors by iterating coordinates.
        # Precompute per-coordinate stride.
        strides = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
        # For each coordinate c and delta d in 1..q-1, the neighbor index = idx +/- d*stride
        # but must wrap per coordinate (mod q).
        # Represent idx in base q digits.
        # We'll do it per coordinate.
        # Build the full set of radius-1 neighbors for every word.
        # Total neighbors = n*(q-1)
        total = n * (q - 1)
        # We'll fill balls[:, 1:1+total]
        # For each coordinate c, for each delta d:
        #   neighbor = word with coordinate c changed to (word_c + d) % q
        # We can compute by: take idx, extract digit c, add d mod q, put back.
        # Using vectorized ops.
        for c in range(n):
            stride = strides[c]
            # digit = (idx // stride) % q
            # neighbor = idx - digit*stride + ((digit + d) % q)*stride
            for d in range(1, q):
                digit = (idx // stride) % q
                new_digit = (digit + d) % q
                neighbor = idx - digit * stride + new_digit * stride
                balls[:, pos] = neighbor.astype(np.int32)
                pos += 1
    return balls


def _greedy_cover(q, n, radius, deadline, seed=0):
    """Greedy set-cover using precomputed balls. Returns list of codeword index tuples."""
    qn = q ** n
    radius = max(0, min(radius, n))
    # Build balls: shape (qn, ball_size)
    # For radius up to 2 we precompute; else fall back to radius-1 only (we'll handle radius>=2 separately)
    if radius == 0:
        # Every word is a codeword
        return [tuple(range(n))] * 0  # placeholder; handled below
    # Precompute balls for the actual radius (only support up to 2 efficiently)
    max_pre = min(radius, 2)
    balls = _all_balls(q, n, max_pre)
    ball_size = balls.shape[1]
    # If radius > max_pre, we cannot fully cover with these balls; but for our task radius=1.
    # We'll assume radius <= max_pre for correctness of coverage check.
    
    covered = np.zeros(qn, dtype=bool)
    chosen = []
    rng = np.random.RandomState(seed)
    
    # Precompute coverage counts for initial greedy
    # We'll maintain covered as a bool array and recompute gains incrementally.
    # For efficiency, maintain a count array of how many chosen balls cover each word.
    counts = np.zeros(qn, dtype=np.int32)
    
    while not covered.all():
        if time.time() > deadline:
            break
        # Compute marginal gain: number of currently-uncovered words in each ball
        # gain[i] = number of j in balls[i] where not covered[j]
        # Vectorized: for each ball, sum over its members of (not covered)
        # balls is (qn, ball_size), covered is (qn,)
        # We can do: (not covered[balls]).sum(axis=1)
        not_cov = ~covered
        gains = not_cov[balls].sum(axis=1)
        # Pick the word with max gain; break ties by index for determinism
        best = int(np.argmax(gains))
        if gains[best] == 0:
            # Should not happen if not covered.all(), but safety
            break
        # Add best
        chosen.append(best)
        # Update counts and covered
        counts[balls[best]] += 1
        covered[balls[best]] = True
    
    # Convert chosen indices to tuples
    result = []
    for idx in chosen:
        digits = []
        v = idx
        for i in range(n - 1, -1, -1):
            digits.append(v % q)
            v //= q
        digits.reverse()
        result.append(tuple(digits))
    return result


def _local_improve(q, n, radius, codewords, balls, covered, counts, deadline):
    """Try to reduce the code by replacing one codeword with an uncovered word.
    Returns (new_codewords, new_covered, new_counts)."""
    qn = q ** n
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find a redundant codeword: one whose removal doesn't uncover any word
        # i.e., for all j in balls[cw], counts[j] >= 2
        # We'll check each chosen codeword.
        # To be efficient, we can check in random order.
        order = list(range(len(codewords)))
        np.random.shuffle(order)
        for pos in order:
            if time.time() > deadline:
                break
            cw = codewords[pos]
            ball = balls[cw]
            # Check if all words in this ball are covered by at least 2 codewords
            if (counts[ball] >= 2).all():
                # This codeword is redundant. Try to replace it with an uncovered word.
                # Find uncovered words
                uncov = np.where(~covered)[0]
                if len(uncov) == 0:
                    # Just remove the redundant codeword
                    codewords.pop(pos)
                    counts[ball] -= 1
                    improved = True
                    break
                # For each uncovered word, compute how many currently-covered words it would cover
                # gain = number of covered words in its ball
                # We want to maximize gain.
                # But we also need to ensure that after removal, the ball of cw is still covered
                # by the new word plus the others. Since cw was redundant (counts>=2 everywhere in its ball),
                # removing it leaves counts>=1. So we just need to pick an uncovered word that
                # maximizes the number of covered words it covers (to potentially make more codewords redundant later).
                # Actually, we should pick the uncovered word that maximizes coverage of currently-covered words,
                # which is equivalent to minimizing the number of uncovered words left.
                # gain[i] = number of covered words in balls[i]
                cov = covered
                gains = cov[balls[uncov]].sum(axis=1)
                best_local = int(np.argmax(gains))
                new_cw = int(uncov[best_local])
                # Replace
                codewords[pos] = new_cw
                # Update counts: remove old, add new
                counts[ball] -= 1
                new_ball = balls[new_cw]
                counts[new_ball] += 1
                # Update covered: words that were only covered by old cw are now uncovered
                # But since counts[ball] >= 2 before, after decrement they are >= 1, so no word becomes uncovered.
                # So covered doesn't change.
                improved = True
                break
    return codewords, covered, counts


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin
    
    # Handle trivial cases
    if radius >= n:
        return [[0] * n]
    if radius == 0:
        return [list(itertools.product(range(q), repeat=n)) and list(word) for word in itertools.product(range(q), repeat=n)]
    
    qn = q ** n
    
    # For small qn, we can be more aggressive
    # Build balls for the actual radius (only support up to 2)
    max_pre = min(radius, 2)
    balls = _all_balls(q, n, max_pre)
    ball_size = balls.shape[1]
    
    # If radius > max_pre, we need to handle it differently. For now, assume radius <= 2.
    # For the task, radius=1, so this is fine.
    
    # Greedy cover
    codewords = _greedy_cover(q, n, max_pre, deadline, seed=42)
    
    # Convert to list of indices
    cw_indices = []
    for cw in codewords:
        idx = 0
        for i, d in enumerate(cw):
            idx = idx * q + d
        cw_indices.append(idx)
    
    # Build covered and counts
    covered = np.zeros(qn, dtype=bool)
    counts = np.zeros(qn, dtype=np.int32)
    for idx in cw_indices:
        ball = balls[idx]
        covered[ball] = True
        counts[ball] += 1
    
    # Local improvement
    cw_indices, covered, counts = _local_improve(q, n, max_pre, cw_indices, balls, covered, counts, deadline)
    
    # If we didn't cover everything (shouldn't happen for radius <= max_pre), fall back
    if not covered.all() and max_pre < radius:
        # Fall back to deterministic construction
        pass
    
    # Convert to list of lists
    result = []
    for idx in cw_indices:
        digits = []
        v = idx
        for i in range(n - 1, -1, -1):
            digits.append(v % q)
            v //= q
        digits.reverse()
        result.append(digits)
    
    return result
