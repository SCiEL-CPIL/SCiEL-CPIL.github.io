"""Rewritten program: structural change from fixed tetracode construction to a
randomized greedy set-cover with local-improvement (redundancy removal) and
multiple randomized restarts, all under an explicit wall-clock deadline."""
import time
import random
import numpy as np

# Precompute nothing heavy at import; do work inside construct.

def _all_words(q, n):
    """Return all q^n words as an array of shape (q^n, n)."""
    base = np.arange(q, dtype=np.int32)
    words = np.empty((q ** n, n), dtype=np.int32)
    for i in range(n):
        words[:, i] = (base ** (n - 1 - i)) % q
        # Actually simpler: use base-q expansion
    # Redo with a clean base-q expansion
    words = np.empty((q ** n, n), dtype=np.int32)
    for i in range(q ** n):
        x = i
        for j in range(n - 1, -1, -1):
            words[i, j] = x % q
            x //= q
    return words


def _ball_indices(word, q, n):
    """Return indices of all words within Hamming distance 1 of `word`."""
    # The word itself plus all single-position changes
    idx = int(np.sum(word * (np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64))))
    result = [idx]
    powers = [q ** (n - 1 - i) for i in range(n)]
    for pos in range(n):
        old = word[pos]
        for new_val in range(q):
            if new_val == old:
                continue
            diff = (new_val - old) * powers[pos]
            result.append(idx + diff)
    return result


def _validate(code, q, n, radius=1):
    """Check that every word is within distance `radius` of some codeword."""
    if radius != 1:
        # Fallback: brute force (shouldn't be called for this task)
        all_w = _all_words(q, n)
        covered = np.zeros(q ** n, dtype=bool)
        for cw in code:
            for i in range(q ** n):
                if np.sum(np.abs(all_w[i] - np.array(cw)) <= 1) >= n - radius:
                    covered[i] = True
        return covered.all()
    # radius == 1: mark covered
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    powers = [q ** (n - 1 - i) for i in range(n)]
    for cw in code:
        idx = int(sum(cw[i] * powers[i] for i in range(n)))
        covered[idx] = True
        for pos in range(n):
            old = cw[pos]
            for new_val in range(q):
                if new_val == old:
                    continue
                covered[idx + (new_val - old) * powers[pos]] = True
    return covered.all()


def _greedy_cover(q, n, radius, rng, deadline):
    """Greedy set cover: repeatedly pick the codeword covering the most uncovered words."""
    total = q ** n
    # Precompute ball membership: for each word index, which codeword candidates cover it?
    # Instead: maintain uncovered set; for each candidate codeword, count newly covered.
    # To speed up, precompute for each index its ball (list of indices it covers).
    # But that's total * (1 + n*(q-1)) entries. For 729 * 19 = ~13841, fine.
    
    # Build ball for every word index
    powers = [q ** (n - 1 - i) for i in range(n)]
    balls = []
    for idx in range(total):
        word = []
        x = idx
        for j in range(n - 1, -1, -1):
            word.append(x % q)
            x //= q
        ball = [idx]
        for pos in range(n):
            old = word[pos]
            for new_val in range(q):
                if new_val == old:
                    continue
                ball.append(idx + (new_val - old) * powers[pos])
        balls.append(ball)
    
    uncovered = np.ones(total, dtype=bool)
    code = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = -1
        # Sample a subset of candidates if too many, but 729 is small
        for idx in range(total):
            if not uncovered[idx]:
                # We can still pick idx as codeword; it covers its own ball
                pass
            gain = 0
            for b in balls[idx]:
                if uncovered[b]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        code.append(best_idx)
        for b in balls[best_idx]:
            uncovered[b] = False
    
    # Convert indices to word lists
    result = []
    for idx in code:
        word = []
        x = idx
        for j in range(n - 1, -1, -1):
            word.append(x % q)
            x //= q
        result.append(word)
    return result


def _remove_redundant(code, q, n, radius, deadline):
    """Try removing each codeword; keep removal if still valid."""
    if not code:
        return code
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_code = list(code)
        for i in range(len(new_code)):
            if time.time() > deadline:
                break
            trial = new_code[:i] + new_code[i+1:]
            if _validate(trial, q, n, radius):
                new_code = trial
                changed = True
                break
        code = new_code
    return code


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    # Fallback: deterministic tetracode-based construction
    import itertools
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: all words (valid but large)
        fallback = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    best = fallback
    best_len = len(fallback)
    
    if q == 3 and n == 6 and radius == 1:
        rng = random.Random(12345)
        # Multiple randomized greedy restarts with different orderings
        total = 3 ** 6  # 729
        powers = [3 ** (5 - i) for i in range(6)]
        
        # Precompute balls once
        balls = []
        for idx in range(total):
            word = []
            x = idx
            for j in range(5, -1, -1):
                word.append(x % 3)
                x //= 3
            ball = [idx]
            for pos in range(6):
                old = word[pos]
                for new_val in range(3):
                    if new_val == old:
                        continue
                    ball.append(idx + (new_val - old) * powers[pos])
            balls.append(ball)
        
        # Greedy with random tie-breaking, multiple restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            uncovered = np.ones(total, dtype=bool)
            code_indices = []
            
            while uncovered.any():
                if time.time() > deadline:
                    break
                # Find best gain with random tie-breaking
                best_gain = -1
                candidates = []
                for idx in range(total):
                    gain = 0
                    for b in balls[idx]:
                        if uncovered[b]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [idx]
                    elif gain == best_gain:
                        candidates.append(idx)
                if best_gain <= 0:
                    break
                # Random tie-break
                chosen = candidates[rng.randrange(len(candidates))]
                code_indices.append(chosen)
                for b in balls[chosen]:
                    uncovered[b] = False
            
            # Convert to words
            code = []
            for idx in code_indices:
                word = []
                x = idx
                for j in range(5, -1, -1):
                    word.append(x % 3)
                    x //= 3
                code.append(word)
            
            # Local improvement: remove redundant codewords
            code = _remove_redundant(code, q, n, radius, deadline)
            
            if len(code) < best_len:
                best_len = len(code)
                best = code
                if best_len <= 73:  # can't do better than known best easily
                    # Try more aggressive removal
                    best = _remove_redundant(best, q, n, radius, deadline)
                    best_len = len(best)
    
    # Final validation
    if not _validate(best, q, n, radius):
        # Return fallback if somehow invalid
        best = fallback
    
    return best
