"""Fresh design for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Fixed deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE_SECONDS = 135.0


def _tetracode_words():
    # The ternary tetracode: a perfect 1-error-correcting code of length 4, size 9.
    return [
        [a, b, (a + b) % 3, (a + 2 * b) % 3]
        for a in range(3)
        for b in range(3)
    ]


def _all_words(q, n):
    # Return a numpy array of shape (q**n, n) containing every word over {0..q-1}^n.
    base = np.arange(q)
    if n == 0:
        return np.empty((1, 0), dtype=np.int8)
    # Build via outer products / broadcasting
    words = base.reshape((1,) * (n - 1) + (q,))
    for i in range(n - 1):
        words = np.repeat(words, q, axis=0)
        words = np.concatenate([words] * q, axis=-1)
    # The above is a bit tricky; let's use a cleaner approach with np.indices or unravel
    total = q ** n
    words = np.empty((total, n), dtype=np.int8)
    for i in range(total):
        val = i
        for pos in range(n):
            words[i, pos] = val % q
            val //= q
    return words


def _ball_indices(q, n, radius, center):
    # Return a numpy array of indices (into the flattened space of size q**n)
    # for all words within Hamming distance <= radius of `center`.
    # For efficiency, we precompute the ball offsets.
    # Actually, for covering radius 1, the ball has 1 + n*(q-1) words.
    # We'll compute the index of each word in the ball.
    # Index of a word w = sum_{i=0}^{n-1} w[i] * q**(n-1-i)
    # Let's precompute powers.
    powers = [q ** (n - 1 - i) for i in range(n)]
    center_idx = sum(c * p for c, p in zip(center, powers))
    ball = [center_idx]
    for pos in range(n):
        for val in range(q):
            if val == center[pos]:
                continue
            new_center = list(center)
            new_center[pos] = val
            idx = sum(c * p for c, p in zip(new_center, powers))
            ball.append(idx)
    return np.array(ball, dtype=np.int64)


def _ball_indices_vectorized(q, n, radius, centers):
    # centers: array of shape (M, n)
    # Returns a set of indices (flattened) covered by the balls.
    # For radius 1, each center covers 1 + n*(q-1) indices.
    M = len(centers)
    if M == 0:
        return set()
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    # Compute center indices
    center_indices = centers @ powers  # shape (M,)
    
    all_covered = set(center_indices.tolist())
    
    # For each position and each alternative value, compute the new index
    # This is M * n * (q-1) operations, which can be large.
    # Let's vectorize per position.
    for pos in range(n):
        # For each center, for each val != center[pos]:
        # new_idx = center_idx - center[pos]*powers[pos] + val*powers[pos]
        center_vals = centers[:, pos]  # shape (M,)
        for val in range(q):
            mask = center_vals != val
            if not np.any(mask):
                continue
            affected_centers = centers[mask]
            affected_idx = center_indices[mask]
            new_vals = np.full(len(affected_idx), val, dtype=np.int64)
            new_indices = affected_idx - affected_centers[:, pos] * powers[pos] + new_vals * powers[pos]
            all_covered.update(new_indices.tolist())
    return all_covered


def _greedy_cover(q, n, radius, time_limit):
    # Greedy set cover: repeatedly pick the word that covers the most uncovered words.
    # Space size is q**n = 2187 for q=3, n=7.
    # Ball size for radius 1: 1 + 7*2 = 15.
    
    start_time = time.time()
    total_words = q ** n
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Precompute all words as indices 0..total_words-1
    # We need to map index -> word and word -> index
    # index = sum(word[i] * powers[i])
    
    # Precompute ball for each possible center? That's 2187 * 15 = 32805 entries.
    # Let's precompute a list of lists: ball[center_idx] = list of covered indices.
    balls = [None] * total_words
    for idx in range(total_words):
        # Decode idx to word
        word = [0] * n
        val = idx
        for pos in range(n):
            word[pos] = val % q
            val //= q
        # Compute ball
        ball = [idx]
        for pos in range(n):
            for v in range(q):
                if v == word[pos]:
                    continue
                new_word = word[:]
                new_word[pos] = v
                new_idx = sum(c * p for c, p in zip(new_word, powers))
                ball.append(new_idx)
        balls[idx] = ball
    
    uncovered = set(range(total_words))
    codewords = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        
        best_center = -1
        best_count = 0
        best_ball = []
        
        for center_idx in range(total_words):
            # Count how many uncovered words are in this ball
            count = 0
            for idx in balls[center_idx]:
                if idx in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_center = center_idx
                best_ball = balls[center_idx]
                if count == len(uncovered):
                    break
        
        if best_center == -1:
            break
        
        # Decode best_center to word
        word = [0] * n
        val = best_center
        for pos in range(n):
            word[pos] = val % q
            val //= q
        codewords.append(word)
        
        for idx in best_ball:
            uncovered.discard(idx)
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    time_limit = _DEADLINE_SECONDS - (time.time() - start_time)
    if time_limit < 1:
        time_limit = 1
    
    # For q=3, n=7, radius=1, we can use a greedy approach.
    # The initial seed gives 9 * 3**(7-4) = 9 * 27 = 243 words.
    # We want to do better.
    
    if q == 3 and n == 7 and radius == 1:
        # Use greedy set cover
        codewords = _greedy_cover(q, n, radius, time_limit)
        return codewords
    else:
        # Fallback: direct product with tetracode
        tetracode = _tetracode_words()
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
# EVOLVE-BLOCK-END
