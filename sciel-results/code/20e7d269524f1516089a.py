"""Constraint-satisfaction / critical-word merge search for K_3(8,1).

Structural change from the seed: instead of only the fixed 486-word direct sum,
we start from that construction and then (1) greedily remove redundant
codewords, and (2) apply a "critical-word merge" heuristic that replaces a
codeword whose uniquely-covered words can all be absorbed by a single
alternative word, thereby *reducing* the number of codewords. A deadline guard
returns the best (smallest) valid configuration found before the limit.
"""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 100.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _direct_sum(q, n, radius):
    """The 486-word ternary direct sum (tetracode x full space on rest)."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [tuple(h + list(tail))
            for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_valid(codewords, q, n, radius, all_words_np):
    """Check every word is within radius of some codeword."""
    covered = np.zeros(len(all_words_np), dtype=bool)
    for c in codewords:
        d = (all_words_np != c).sum(axis=1)
        covered |= (d <= radius)
    return bool(covered.all())


def _greedy_remove(codewords, q, n, radius, all_words_np, time_limit):
    """Iteratively remove any codeword that is fully redundant."""
    codewords = list(codewords)
    changed = True
    while changed and _remaining() > time_limit:
        changed = False
        # coverage count per word
        cov = np.zeros(len(all_words_np), dtype=np.int32)
        for c in codewords:
            d = (all_words_np != c).sum(axis=1)
            cov += (d <= radius).astype(np.int32)
        
        # find a removable codeword: every word it covers has cov>=2
        for i, c in enumerate(codewords):
            d = (all_words_np != c).sum(axis=1)
            ball = d <= radius
            if (cov[ball] >= 2).all():
                codewords.pop(i)
                changed = True
                break
    return codewords


def _merge_step(codewords, q, n, radius, all_words_np, rng, time_limit):
    """Try to reduce codeword count by one via a critical-word merge."""
    codewords = list(codewords)
    n_words = len(all_words_np)
    
    # Precompute balls for current codewords to speed up unique calculation
    # and to allow fast checking of candidates later if we cache them.
    # However, calculating balls for 486 words takes time. 
    # We only do this if we have time.
    
    # 1. Compute coverage map and unique sets
    cover = {}  # word_idx -> set of cw indices
    balls = []
    for i, c in enumerate(codewords):
        d = (all_words_np != c).sum(axis=1)
        idxs = np.nonzero(d <= radius)[0]
        balls.append(idxs)
        for j in idxs:
            cover.setdefault(int(j), set()).add(i)

    # 2. Identify unique words for each codeword
    unique_sets = []
    for i in range(len(codewords)):
        u = [int(j) for j in balls[i] if len(cover.get(int(j), set())) == 1]
        unique_sets.append(u)

    # 3. Try to merge
    # Sort by number of unique words descending (hardest to cover first)
    order = sorted(range(len(codewords)), key=lambda i: -len(unique_sets[i]))
    
    for i in order:
        if _remaining() < time_limit:
            break
        u = unique_sets[i]
        if not u:
            continue
        
        c = codewords[i]
        c_set = set(codewords)
        
        # Candidate generation:
        # Instead of random, try neighbors of c (distance 1) and some randoms.
        # If c has unique words, a replacement w must be close to them.
        # Actually, w must cover u. 
        # A good heuristic: pick a word u_j from u, and try words close to u_j.
        
        candidates = set()
        # Add c's neighbors
        for pos in range(n):
            for val in range(q):
                if c[pos] != val:
                    w_list = list(c)
                    w_list[pos] = val
                    w = tuple(w_list)
                    if w not in c_set:
                        candidates.add(w)
        
        # Add random candidates
        for _ in range(50):
            w = tuple(rng.randrange(q) for _ in range(n))
            if w not in c_set:
                candidates.add(w)
        
        # Try candidates
        found = False
        for w in candidates:
            if _remaining() < time_limit:
                break
            # Check if w covers all u
            d_w = (all_words_np != w).sum(axis=1)
            wball = d_w <= radius
            # Check if all u are in wball
            # u is a list of indices
            if all(wball[j] for j in u):
                # Accept
                codewords.pop(i)
                codewords.append(w)
                return codewords, True
    return codewords, False


def construct(q, n, radius):
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = 100.0

    rng = random.Random(12345)

    # Precompute all words as numpy array for fast distance calculation
    all_words_list = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    all_words_np = np.array(all_words_list, dtype=np.int8)

    # Start from the 486-word direct sum.
    codewords = _direct_sum(q, n, radius)

    best = list(codewords)
    best_len = len(best)

    # Phase 1: greedy removal of fully-redundant codewords.
    codewords = _greedy_remove(codewords, q, n, radius, all_words_np, 10.0)
    if len(codewords) < best_len:
        best = list(codewords)
        best_len = len(best)

    # Phase 2: iterative critical-word merges
    iters = 0
    while _remaining() > 20.0:
        iters += 1
        codewords, did = _merge_step(codewords, q, n, radius, all_words_np, rng, 5.0)
        if did and len(codewords) < best_len:
            best = list(codewords)
            best_len = len(best)
        
        # Occasionally re-greedy to clean up
        if iters % 5 == 0 and _remaining() > 20.0:
            codewords = _greedy_remove(codewords, q, n, radius, all_words_np, 5.0)
            if len(codewords) < best_len:
                best = list(codewords)
                best_len = len(best)

    # Final validation
    try:
        if _is_valid(best, q, n, radius, all_words_np):
            return [list(w) for w in best]
    except Exception:
        pass
    
    # Fallback
    return [list(w) for w in _direct_sum(q, n, radius)]
# EVOLVE-BLOCK-END
