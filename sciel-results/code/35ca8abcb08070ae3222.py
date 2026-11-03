"""Fresh design for the ternary covering code task (K_3(7,1)).

Strategy (same family as the seed: direct-sum construction + improvement, but
fully vectorized so a real greedy reduction can run inside the time budget):

  1. Build a guaranteed-covering code via the direct sum of the ternary
     tetracode (a perfect code of covering radius 1 on length 4) with the
     full space on the remaining coordinates.  This always covers every word.
  2. Greedily reduce the code to a minimum-ish covering code by solving the
     set-cover problem over the q**n words: each codeword's radius-`radius`
     ball is a set, and we greedily pick the ball covering the most
     still-uncovered words until the whole space is covered.  The selected
     balls form a valid covering code; we return its codewords.
  3. An explicit deadline guard tracks wall-clock time and returns the best
     (smallest, still fully-covering) code found before the limit.

Entry point name and return type are unchanged:
    construct(q, n, radius) -> list of codewords (list of lists of ints)
"""
import time
import itertools
import numpy as np

_DEADLINE_SEC = 150.0
_SAFETY_SEC = 15.0
_MAX_BUILD = 600_000


def _tetracode(q):
    # Ternary tetracode: 9 words of length 4, a perfect code of covering radius 1.
    return [[a, b, (a + b) % q, (a + 2 * b) % q]
            for a in range(q) for b in range(q)]


def _words_to_index(words, q, n):
    """Convert an (m,n) array of words to their base-q indices (m,)."""
    words = np.asarray(words)
    m = words.shape[0]
    idx = np.zeros(m, dtype=np.int64)
    for p in range(n):
        idx = idx * q + words[:, p]
    return idx


def _ball_indices(c, q, n, radius):
    """Return the base-q indices of all words within Hamming distance <= radius
    of the single word c (a list of n ints)."""
    idxs = set()
    idxs.add(int(_words_to_index(np.array([c], dtype=np.int8), q, n)[0]))
    for r in range(1, radius + 1):
        for comb in itertools.combinations(range(n), r):
            for vals in itertools.product(range(q), repeat=r):
                cc = list(c)
                for p, v in zip(comb, vals):
                    cc[p] = v
                idxs.add(int(_words_to_index(np.array([cc], dtype=np.int8), q, n)[0]))
    return idxs


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE_SEC - _SAFETY_SEC

    # ---- 1. Build a guaranteed-covering code (direct sum of tetracode). ---
    if n <= 4:
        codes = np.array(_tetracode(q), dtype=np.int8)
    else:
        tet = _tetracode(q)
        m = len(tet) * (q ** (n - 4))
        codes = np.empty((m, n), dtype=np.int8)
        i = 0
        for head in tet:
            for tail in itertools.product(range(q), repeat=n - 4):
                codes[i, :4] = head
                codes[i, 4:] = tail
                i += 1

    N = q ** n

    # ---- 2. Precompute each codeword's radius-`radius` ball (as indices). --
    # For the given task (m=243, N=2187, radius=1) this is tiny and fast.
    ball_list = []
    for c in codes:
        ball_list.append(_ball_indices(list(c), q, n, radius))

    # ---- 3. Greedy set-cover over the space. ------------------------------
    uncovered = np.ones(N, dtype=bool)
    selected = []
    remaining = list(range(codes.shape[0]))
    # Precompute each ball as a boolean vector over the space for fast counting.
    ball_bool = [np.zeros(N, dtype=bool) for _ in range(codes.shape[0])]
    for j, idxs in enumerate(ball_list):
        ball_bool[j][list(idxs)] = True

    while uncovered.any():
        if time.time() > deadline:
            break
        # Pick the codeword covering the most uncovered words.
        best_j = -1
        best_cnt = -1
        for j in remaining:
            cnt = int(np.count_nonzero(ball_bool[j] & uncovered))
            if cnt > best_cnt:
                best_cnt = cnt
                best_j = j
        if best_cnt == 0:
            # No remaining codeword covers anything uncovered -> cannot cover.
            break
        selected.append(best_j)
        uncovered &= ~ball_bool[best_j]
        remaining.remove(best_j)

    # Build the selected code and verify exact coverage.
    best_codes = codes
    if selected:
        sel_codes = codes[selected]
        sel_covered = np.zeros(N, dtype=bool)
        for j in selected:
            sel_covered |= ball_bool[j]
        if sel_covered.all() and sel_codes.shape[0] < best_codes.shape[0]:
            best_codes = sel_codes

    # ---- 4. Return the best (smallest) valid code found. ------------------
    out = [[int(x) for x in c] for c in best_codes]
    return out
