"""Superpermutation construction with improved greedy overlap and suffix-priority heuristics."""
import time
import random
from itertools import permutations


# EVOLVE-BLOCK-START

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    if n <= 1:
        return "1"
    if n <= 3:
        return _standard_superperm(n)
    if n == 4:
        return _standard_superperm(n)

    # Start with the classic construction as baseline
    word = _standard_superperm(n)
    best_word = word
    best_len = len(word)

    # For n=7, the classic construction gives 5913.
    # We try to improve via randomized greedy with suffix-priority.
    # The key insight: when building the superpermutation level by level,
    # the order in which we append blocks affects the total length.
    # Suffix-priority: prefer transitions that maximize reuse of existing suffix structures.

    if n >= 5:
        # Use multiple strategies and pick the best
        max_restarts = 30 if n <= 6 else 15

        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                break

            rng = random.Random(restart * 7919 + 13)
            result = _greedy_construct(n, rng, strategy="suffix_priority")
            if result is not None and len(result) < best_len:
                best_word = result
                best_len = len(result)

            # Also try standard greedy
            result2 = _greedy_construct(n, rng, strategy="standard")
            if result2 is not None and len(result2) < best_len:
                best_word = result2
                best_len = len(result2)

    return best_word


def _standard_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = _max_overlap(new, block)
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _max_overlap(existing, block):
    """Find maximum suffix of existing that matches prefix of block."""
    max_ov = min(len(existing), len(block))
    for o in range(max_ov, -1, -1):
        if o == 0:
            return 0
        if existing[len(existing) - o:] == block[:o]:
            return o
    return 0


def _greedy_construct(n, rng=None, strategy="standard"):
    """
    Greedy construction with randomized ordering.
    strategy: "standard" or "suffix_priority"
    """
    if rng is None:
        rng = random.Random()

    word = [1]
    for m in range(2, n + 1):
        # Get all (m-1)-permutations that appear in the current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        # Build the new word by ordering these perms
        remaining = list(perms)
        rng.shuffle(remaining)

        if not remaining:
            continue

        first = remaining.pop(0)
        new = list(first) + [m] + list(first)

        if strategy == "suffix_priority":
            new = _build_suffix_priority(new, remaining, m, rng)
        else:
            new = _build_standard_greedy(new, remaining, m, rng)

        word = new

    result = "".join(map(str, word))
    if _verify(result, n):
        return result
    return None


def _build_standard_greedy(new, remaining_perms, m, rng):
    """Standard greedy: pick next perm that gives maximum overlap."""
    while remaining_perms:
        best_idx = -1
        best_overlap = -1
        for i, p in enumerate(remaining_perms):
            block = list(p) + [m] + list(p)
            ov = _max_overlap(new, block)
            if ov > best_overlap:
                best_overlap = ov
                best_idx = i
        if best_idx == -1:
            best_idx = 0
            best_overlap = 0
        p = remaining_perms.pop(best_idx)
        block = list(p) + [m] + list(p)
        new += block[best_overlap:]
    return new


def _build_suffix_priority(new, remaining_perms, m, rng):
    """
    Suffix-priority greedy: prioritize transitions that maximize reuse
    of existing suffix structures. We score each candidate by:
    1. Overlap with current suffix (primary)
    2. How many of the block's internal windows are already "covered" by
       the existing structure (secondary)
    3. Random tie-breaking
    """
    while remaining_perms:
        best_idx = -1
        best_score = -1
        best_overlap = 0
        for i, p in enumerate(remaining_perms):
            block = list(p) + [m] + list(p)
            ov = _max_overlap(new, block)

            # Score: primarily overlap, secondarily how much of block
            # shares structure with existing word
            # The block has length 2*(m-1)+1 = 2m-1
            # Internal windows of length m-1 in block:
            # positions 0..m-2 and m..2m-2 (total 2*(m-1) windows, but
            # some may be duplicates)
            # Count how many distinct (m-1)-windows in block already appear
            # in the existing word
            block_windows = set()
            for j in range(len(block) - (m - 1) + 1):
                w = tuple(block[j:j + m - 1])
                if len(set(w)) == m - 1:
                    block_windows.add(w)

            # Count how many of these appear in existing word
            existing_windows = set()
            for j in range(len(new) - (m - 1) + 1):
                w = tuple(new[j:j + m - 1])
                if len(set(w)) == m - 1:
                    existing_windows.add(w)

            reuse = len(block_windows & existing_windows)
            # Score: overlap is dominant, reuse is secondary
            score = ov * 100 + reuse + rng.random() * 0.5
            if score > best_score:
                best_score = score
                best_idx = i
                best_overlap = ov

        if best_idx == -1:
            best_idx = 0
            best_overlap = 0
        p = remaining_perms.pop(best_idx)
        block = list(p) + [m] + list(p)
        new += block[best_overlap:]
    return new


def _verify(word_str, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word_str) < n:
        return False
    seen = set()
    for i in range(len(word_str) - n + 1):
        w = word_str[i:i + n]
        if len(set(w)) == n:
            seen.add(w)
    return len(seen) == _factorial(n)


def _factorial(k):
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result
# EVOLVE-BLOCK-END
