"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
from collections import Counter

def _classic_superperm(n):
    """Classic recursive construction: length 1!+2!+...+n!."""
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _greedy_construct(n, mode, deadline, start_time):
    """Suffix-priority constructive pass.

    Build the word symbol-by-symbol. Maintain the last (n-1) symbols as the
    sliding suffix. At each step, among symbols c in 1..n that (a) are not
    already in the suffix and (b) form a new (unseen) n-permutation when
    appended, pick the one that maximizes reuse of existing suffix structure.
    'mode' selects the priority among ties/choices:
      - 'overlap': prefer the symbol whose resulting permutation shares the
        longest suffix with previously seen permutations (max reuse).
      - 'lex': lexicographically smallest new permutation.
      - 'reuse': prefer transitions whose last symbol equals the most common
        next-symbol continuation (maximize reuse of existing transitions).
    """
    all_perms = set(itertools.permutations(range(1, n + 1)))
    total = n ** n  # n * n! is the trivial upper bound; use n**n as safe cap
    # Trivial upper bound is n * n!
    upper = n * math_factorial(n)

    # Start with the first permutation
    start_perm = tuple(range(1, n + 1))
    word = list(start_perm)
    seen_perms = {start_perm}
    remaining = set(all_perms) - seen_perms

    # Track last (n-1) symbols as suffix
    suffix = list(start_perm[1:])  # length n-1

    # Transition reuse counts: for each symbol s, count how often it has been
    # appended (i.e., how many times s appeared as a new symbol).
    trans_count = Counter()

    step_budget = 4 * upper  # we cannot append more than this in a sane build
    steps = 0
    while remaining and steps < step_budget:
        if time.time() > deadline:
            break
        steps += 1

        suffix_set = set(suffix)
        # Candidate symbols: not in suffix, and (suffix + [c]) is a new perm
        candidates = []
        for c in range(1, n + 1):
            if c in suffix_set:
                continue
            new_perm = tuple(suffix) + (c,)
            if new_perm not in remaining:
                continue
            candidates.append((c, new_perm))

        if not candidates:
            # Dead end: we cannot extend without repeating. This should not
            # happen for a proper superpermutation, but guard anyway.
            break

        # Choose the best candidate by mode
        best = None
        if mode == 'lex':
            best = min(candidates, key=lambda t: t[1])
        elif mode == 'reuse':
            # Prefer candidate whose appended symbol c has the highest
            # transition count (most-reused continuation), tie-break lex.
            best = min(candidates,
                       key=lambda t: (-trans_count[t[0]], t[1]))
        else:  # 'overlap'
            # Prefer candidate whose new permutation shares the longest
            # suffix with an already-seen permutation (max reuse of structure).
            # We approximate by checking, for each seen perm, the length of
            # common suffix with the candidate new_perm.
            # To keep it cheap, sample a small set of seen perms.
            seen_list = list(seen_perms)
            if len(seen_list) > 20000:
                # Too many; fall back to lex to save time
                best = min(candidates, key=lambda t: t[1])
            else:
                best_key = None
                best_score = -1
                for c, new_perm in candidates:
                    # compute max suffix overlap with any seen perm
                    max_ov = 0
                    # Only need to check suffixes of new_perm against seen perms
                    # that end the same way. Use a map from suffix to perm.
                    # Cheap approach: check overlap for a few seen perms.
                    # We'll check up to 50 random-ish seen perms for speed.
                    sample = seen_list[:50] if len(seen_list) > 50 else seen_list
                    for sp in sample:
                        ov = _suffix_overlap(sp, new_perm)
                        if ov > max_ov:
                            max_ov = ov
                    # score: higher overlap better; tie-break lex
                    score = max_ov
                    if best_key is None or score > best_score or \
                            (score == best_score and new_perm < best_key[1]):
                        best_score = score
                        best_key = (c, new_perm)
                best = best_key

        c, new_perm = best
        word.append(c)
        seen_perms.add(new_perm)
        remaining.remove(new_perm)
        trans_count[c] += 1
        # update suffix: drop first, append c
        suffix.pop(0)
        suffix.append(c)

    return word

def _suffix_overlap(a, b):
    """Length of the longest suffix of a that equals a prefix of b? No:
    we want the longest k such that the last k of a == last k of b
    (shared tail), which is the reuse of suffix structure."""
    m = min(len(a), len(b))
    k = 0
    for i in range(1, m + 1):
        if a[-i] == b[-i]:
            k = i
        else:
            break
    return k

def _deletion_polish(word, n, all_perms, deadline, start_time):
    """Greedy deletion pass with incremental count updates."""
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        perm_counts[tuple(word[i:i + n])] += 1

    changed = True
    while changed and (time.time() - start_time) < deadline - 3:
        changed = False
        for i in range(len(word)):
            if time.time() - start_time > deadline - 3:
                break
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            safe = True
            for s in range(s_min, s_max + 1):
                w = tuple(word[s:s + n])
                if w in all_perms and perm_counts[w] == 1:
                    safe = False
                    break
            if safe:
                word.pop(i)
                changed = True
                # recompute counts locally: recompute all for simplicity
                perm_counts = Counter()
                for s in range(len(word) - n + 1):
                    perm_counts[tuple(word[s:s + n])] += 1
                break
    return word

def math_factorial(x):
    r = 1
    for i in range(2, x + 1):
        r *= i
    return r

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        # Classic construction is optimal for n<=4
        word = _classic_superperm(n)
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    all_perms = set(itertools.permutations(range(1, n + 1)))

    # Generate candidate words from different heuristic modes within the
    # same constructive family, plus the classic construction as a baseline.
    candidates = []

    # Baseline: classic construction
    classic = _classic_superperm(n)
    candidates.append(classic)

    # Greedy constructive passes with different priority modes
    for mode in ('overlap', 'reuse', 'lex'):
        if time.time() > deadline - 10:
            break
        w = _greedy_construct(n, mode, deadline, start_time)
        if w:
            candidates.append(w)

    # Pick the shortest valid candidate
    best_word = None
    best_len = None
    for w in candidates:
        if not w:
            continue
        # validate
        if len(w) < n:
            continue
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        if not all_perms.issubset(seen):
            continue
        if best_len is None or len(w) < best_len:
            best_len = len(w)
            best_word = w

    if best_word is None:
        # fallback: classic is always valid
        best_word = classic
        best_len = len(classic)

    # Deletion polish on the best candidate if we have time
    if (time.time() - start_time) < deadline - 8 and best_len > n:
        polished = _deletion_polish(list(best_word), n, all_perms,
                                    deadline, start_time)
        # validate polished
        if len(polished) >= n:
            seen = set()
            for i in range(len(polished) - n + 1):
                seen.add(tuple(polished[i:i + n]))
            if all_perms.issubset(seen) and len(polished) < best_len:
                best_word = polished
                best_len = len(polished)

    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
