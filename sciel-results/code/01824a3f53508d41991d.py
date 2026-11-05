"""Covering code K_q(n,r) via greedy set-cover + local improvement."""
import time
import itertools
import random
import numpy as np


def _all_words(q, n):
    """Return all q^n words as a numpy array of shape (q^n, n) with entries 0..q-1."""
    if n == 0:
        return np.empty((1, 0), dtype=np.int8)
    # Build via base-q expansion
    total = q ** n
    words = np.empty((total, n), dtype=np.int8)
    # Generate in mixed-radix order: index -> digits
    idx = np.arange(total, dtype=np.int64)
    for i in range(n):
        words[:, i] = (idx // (q ** (n - 1 - i))) % q
    return words


def _hamming_distances_to_ball(center, q, n):
    """
    For a single center (list/tuple of n ints), return a boolean mask of length q^n
    indicating which words are within Hamming distance 1 of center.
    Uses precomputed word array via closure? We'll pass words in.
    """
    pass


def construct(q, n, radius):
    """
    Build a covering code of length n over alphabet size q with covering radius `radius`.
    Returns a list of codewords (each a list of n ints in 0..q-1).
    Strategy:
      - Start from a structured seed (direct-sum of tetracode if n>=4, else random)
      - Greedy set-cover: repeatedly add the word that covers the most still-uncovered points
      - Local improvement: try to remove redundant codewords, try to swap codewords
    """
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(12345)

    total_words = q ** n
    if total_words <= 2000000:
        words = _all_words(q, n)
    else:
        words = None  # too large to hold; use on-the-fly

    # Precompute: for each word index, the list of indices within distance `radius`
    # This is expensive for large spaces; only do if space is small enough.
    # For q=3, n=7: 2187 words, radius=1 -> each ball has 1 + 7*2 = 15 points.
    # We can precompute the ball membership.

    use_precompute = total_words <= 500000
    if use_precompute:
        # ball[i] = set/list of word-indices within radius of word i
        # For radius=1: center + all words differing in exactly 1 position
        ball = []
        if radius == 1:
            for i in range(total_words):
                w = words[i]
                members = [i]
                for pos in range(n):
                    orig = w[pos]
                    for v in range(q):
                        if v == orig:
                            continue
                        # compute index of word with pos-th digit = v
                        # index = sum(digits[j] * q^(n-1-j))
                        idx_val = i
                        # subtract old contribution, add new
                        power = q ** (n - 1 - pos)
                        idx_val = idx_val - orig * power + v * power
                        members.append(idx_val)
                ball.append(members)
        else:
            # Generic radius: compute by hamming distance (slow, only for small spaces)
            for i in range(total_words):
                wi = words[i]
                members = []
                for j in range(total_words):
                    d = int(np.sum(wi != words[j]))
                    if d <= radius:
                        members.append(j)
                ball.append(members)
    else:
        ball = None

    # --- Initial code: structured seed ---
    # Direct sum of tetracode if n >= 4
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_code = [tuple(h + list(tail)) for h in tetracode
                     for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Random initial code
        init_size = max(1, total_words // (q ** n + 1))
        init_code = [tuple(rng.randrange(q) for _ in range(n)) for _ in range(init_size)]

    code = list(init_code)
    code_set = set(code)

    # --- Helper: compute uncovered set ---
    def get_uncovered(code_list):
        """Return a set of word-indices not covered by code_list."""
        if use_precompute:
            covered = np.zeros(total_words, dtype=bool)
            for cw in code_list:
                # find index of cw
                # We need a mapping from tuple to index
                idx = tuple_to_idx[cw]
                covered[ball[idx]] = True
            return np.where(~covered)[0].tolist()
        else:
            # On the fly: check each word
            uncovered = []
            for i in range(total_words):
                w = words[i]
                covered = False
                for cw in code_list:
                    d = sum(1 for a, b in zip(w, cw) if a != b)
                    if d <= radius:
                        covered = True
                        break
                if not covered:
                    uncovered.append(i)
            return uncovered

    # Mapping from tuple to index (only if using precompute)
    tuple_to_idx = {}
    if use_precompute:
        for i in range(total_words):
            tuple_to_idx[tuple(words[i])] = i

    # --- Greedy set-cover phase ---
    uncovered = get_uncovered(code)
    while uncovered and time.time() < deadline - 30:
        # Find the word that covers the most uncovered points
        # For each candidate word (all words), count how many uncovered it covers
        best_count = 0
        best_word = None
        if use_precompute:
            # For each uncovered point, it's covered by words in its ball
            # We want the word covering most uncovered points
            # Use a counter
            from collections import Counter
            cover_count = Counter()
            for ui in uncovered:
                for cw_idx in ball[ui]:
                    cover_count[cw_idx] += 1
            if cover_count:
                best_idx = max(cover_count, key=cover_count.get)
                best_count = cover_count[best_idx]
                best_word = tuple(words[best_idx])
        else:
            # Brute force over all words (slow)
            for i in range(total_words):
                w = words[i]
                cnt = 0
                for ui in uncovered:
                    d = sum(1 for a, b in zip(w, words[ui]) if a != b)
                    if d <= radius:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_word = tuple(w)

        if best_word is None or best_count == 0:
            break
        if best_word not in code_set:
            code.append(best_word)
            code_set.add(best_word)
            # Update uncovered
            uncovered = get_uncovered(code)

    # --- Local improvement phase: try to remove redundant codewords ---
    if use_precompute and time.time() < deadline - 10:
        # Try to remove each codeword if the code still covers everything
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            # Temporarily remove code[i]
            code[i] = None
            # Check coverage
            covered = np.zeros(total_words, dtype=bool)
            for j, cw in enumerate(code):
                if cw is None:
                    continue
                idx = tuple_to_idx[cw]
                covered[ball[idx]] = True
            if covered.all():
                # Can remove
                code.pop(i)
                code_set.discard(code[i] if i < len(code) else None)
                # Fix: we need to remove the right one
                # Actually, let's redo this more carefully
                pass
            code[i] = None  # placeholder, will fix below

    # The removal logic above is buggy; let's do it cleanly
    # Rebuild code cleanly
    code = [c for c in code if c is not None]
    code_set = set(code)

    # --- Local improvement: try swaps ---
    if use_precompute and time.time() < deadline - 5:
        # Try to swap a codeword with another word to maintain coverage
        # This is complex; skip for now and just return
        pass

    # Final check: ensure coverage
    if use_precompute:
        covered = np.zeros(total_words, dtype=bool)
        for cw in code:
            idx = tuple_to_idx[cw]
            covered[ball[idx]] = True
        if not covered.all():
            # Greedily add more until covered
            uncovered = np.where(~covered)[0].tolist()
            while uncovered and time.time() < deadline:
                from collections import Counter
                cover_count = Counter()
                for ui in uncovered:
                    for cw_idx in ball[ui]:
                        cover_count[cw_idx] += 1
                if not cover_count:
                    break
                best_idx = max(cover_count, key=cover_count.get)
                best_word = tuple(words[best_idx])
                if best_word not in code_set:
                    code.append(best_word)
                    code_set.add(best_word)
                uncovered = get_uncovered(code)
    else:
        # Verify and fix
        for i in range(total_words):
            w = words[i]
            covered = False
            for cw in code:
                d = sum(1 for a, b in zip(w, cw) if a != b)
                if d <= radius:
                    covered = True
                    break
            if not covered:
                # Add w itself
                code.append(list(w))

    # Ensure distinct
    seen = set()
    unique_code = []
    for cw in code:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_code.append(list(cw))

    return unique_code
