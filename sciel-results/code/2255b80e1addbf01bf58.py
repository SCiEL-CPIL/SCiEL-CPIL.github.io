"""Superpermutation constructor: classic recursive overlap construction with
deadline guard, a fast verifier, and safe deterministic variants to shave length.
Always returns a verified-valid superpermutation (fallback = classic 5913 for n=7)."""
import time
import itertools

# EVOLVE-BLOCK-START

_DEADLINE = 150.0          # wall-clock limit (evaluator kills at 150s)
_MARGIN = 5.0              # safety margin in seconds

def _elapsed_ok(start, margin=_MARGIN):
    return (time.time() - start) < (_DEADLINE - margin)

def _verify(word, n):
    """Return True iff every permutation of 1..n appears as n consecutive symbols."""
    if len(word) < n:
        return False
    target = set(itertools.permutations(range(1, n + 1)))
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
        if len(seen) == len(target):
            return True
    return seen == target

def _classic(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                           if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _build_from_order(order, m, prev_word):
    """Given an ordering of (m-1)-windows, build level-m word by inserting m."""
    new = []
    for p in order:
        block = list(p) + [m] + list(p)
        overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                       if new[len(new) - o:] == block[:o])
        new += block[overlap:]
    return new

def construct(n):
    start = time.time()

    # Guaranteed-valid baseline: classic construction (fast, <1s for n=7).
    base = _classic(n)
    best = base
    if not _verify(base, n):
        # Should never happen, but guard against it.
        return "".join(map(str, base))

    # Build the (n-1)-windows in classic order of first appearance for the
    # final level, so we can try safe deterministic re-orderings of that level.
    if n >= 2:
        prev = _classic(n - 1)
        order, seen = [], set()
        for i in range(len(prev) - n + 1):
            w = tuple(prev[i:i + n - 1])
            if len(set(w)) == n - 1 and w not in seen:
                seen.add(w)
                order.append(w)

        # Candidate orderings: identity (== classic top level), reversed, and
        # a few rotations. Only accept a candidate if it is verified valid and
        # strictly shorter. This keeps the result always valid.
        candidates = [order, list(reversed(order))]
        for k in (1, 2, 3, n - 1):
            candidates.append(order[k:] + order[:k])

        for cand in candidates:
            if not _elapsed_ok(start):
                break
            if len(cand) != len(order):
                continue
            cand_word = _build_from_order(cand, n, prev)
            if len(cand_word) < len(best) and _verify(cand_word, n):
                best = cand_word

    return "".join(map(str, best))

# EVOLVE-BLOCK-END
