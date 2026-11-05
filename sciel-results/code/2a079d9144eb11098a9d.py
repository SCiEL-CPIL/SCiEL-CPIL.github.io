"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _build_superperm(n):
    """Standard recursive superpermutation of length 1!+2!+...+n!."""
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
            # overlap: longest suffix of new == prefix of block
            overlap = 0
            min_len = min(len(new), len(block))
            for o in range(min_len, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _perm_count(word, n):
    """Return set of all n-perms appearing in word."""
    s = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            s.add(w)
    return s

def _deletion_pass(word, n, budget):
    """
    Greedily delete symbols from word. A symbol at position i can be deleted
    if no n-window that contains only position i as a 'unique' occurrence would be lost.
    Fast check: maintain the set of present n-perms; a deletion is safe iff
    every n-perm that used to appear still appears.
    We use a sliding approach: for position i, the affected windows are those
    starting in [i-n+1, i]. For each such window, check if it's a perm and whether
    it survives after deletion.
    """
    t0 = time.time()
    word = list(word)
    n_set = _perm_count(word, n)
    required = set(n_set)  # we must keep all these
    changed = True
    while changed and time.time() < budget:
        changed = False
        # scan left to right
        i = 0
        while i < len(word):
            if time.time() > budget:
                break
            # windows affected by removing position i: start in [max(0,i-n+1), min(i, len(word)-n)]
            lo = max(0, i - n + 1)
            hi = min(i, len(word) - n)
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s + n])
                if len(set(w)) == n and w in required:
                    # after deletion, does w still appear?
                    # Reconstruct the word with i removed and check window at s
                    # (if s < i) or s (if s >= i, but s <= i so s < i or s == i)
                    # Actually windows starting at s: if s < i, the window shifts;
                    # the window that was at s now becomes word[s:s+i] + word[i+1:s+i+1]
                    # This is complex; simpler: build deleted word and recheck.
                    # Optimization: only need to check if w appears elsewhere.
                    # Find all occurrences of w in current word.
                    found = False
                    for j in range(len(word) - n + 1):
                        if j == s:
                            continue
                        if tuple(word[j:j + n]) == w:
                            found = True
                            break
                    if not found:
                        safe = False
                        break
            if safe:
                del word[i]
                # update required: recompute is expensive; instead update incrementally
                # Remove perms that no longer appear
                # For efficiency, recompute affected windows
                changed = True
                # don't increment i (next element shifted into position i)
            else:
                i += 1
    return word

def _fast_deletion(word, n, budget):
    """
    Faster deletion using the insight: a symbol can be removed if it's not
    the 'only' occurrence of any permutation in its local neighborhood.
    We track, for each position, whether it's critical for any perm.
    """
    t0 = time.time()
    word = list(word)
    L = len(word)
    # Map: perm -> list of starting positions
    perm_pos = {}
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            perm_pos.setdefault(w, []).append(i)
    required = set(perm_pos.keys())

    changed = True
    while changed and time.time() < budget:
        changed = False
        i = 0
        while i < L:
            if time.time() > budget:
                break
            # A position i is in windows starting at [i-n+1, i]
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s + n])
                if len(set(w)) == n:
                    # how many times does w appear?
                    if s in perm_pos.get(w, []):
                        if len(perm_pos[w]) == 1:
                            safe = False
                            break
            if safe:
                del word[i]
                L -= 1
                changed = True
                # rebuild perm_pos locally is expensive; do a full rebuild occasionally
                # For correctness, rebuild perm_pos after each deletion batch
                # Optimization: rebuild every few deletions
            else:
                i += 1
        # Rebuild perm_pos after a full pass
        perm_pos = {}
        for i in range(L - n + 1):
            w = tuple(word[i:i + n])
            if len(set(w)) == n:
                perm_pos.setdefault(w, []).append(i)
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    if n <= 6:
        word = _build_superperm(n)
        return "".join(map(str, word))

    if n == 7:
        # Build the standard S_7 superpermutation (length 5913)
        base = _build_superperm(7)
        best = base
        best_len = len(base)

        # Try multiple deletion strategies with random restarts
        random.seed(12345)

        # Strategy 1: left-to-right greedy deletion
        budget1 = deadline - 60
        w1 = _fast_deletion(base, 7, budget1)
        if len(w1) < best_len:
            best = w1
            best_len = len(w1)

        # Strategy 2: random-order deletion
        if time.time() < deadline - 40:
            budget2 = deadline - 40
            w2 = list(base)
            random.shuffle(w2)  # no, we can't shuffle the word itself
            # Instead: do deletion in random position order
            positions = list(range(len(base)))
            random.shuffle(positions)
            # Apply deletions in random order
            word2 = list(base)
            L = len(word2)
            perm_pos = {}
            for i in range(L - 6):
                w = tuple(word2[i:i + 7])
                if len(set(w)) == 7:
                    perm_pos.setdefault(w, []).append(i)
            for pos in positions:
                if time.time() > budget2:
                    break
                if pos >= L:
                    continue
                lo = max(0, pos - 6)
                hi = min(pos, L - 7)
                safe = True
                for s in range(lo, hi + 1):
                    w = tuple(word2[s:s + 7])
                    if len(set(w)) == 7:
                        if s in perm_pos.get(w, []):
                            if len(perm_pos[w]) == 1:
                                safe = False
                                break
                if safe:
                    del word2[pos]
                    L -= 1
                    # rebuild perm_pos
                    perm_pos = {}
                    for i in range(L - 6):
                        w = tuple(word2[i:i + 7])
                        if len(set(w)) == 7:
                            perm_pos.setdefault(w, []).append(i)
            if len(word2) < best_len:
                best = word2
                best_len = len(word2)

        # Strategy 3: multi-pass left-to-right (repeat deletion)
        if time.time() < deadline - 20:
            budget3 = deadline - 20
            w3 = list(best)
            # Multiple passes
            for _ in range(3):
                w3 = _fast_deletion(w3, 7, min(budget3, time.time() + 5))
                if time.time() > budget3:
                    break
            if len(w3) < best_len:
                best = w3
                best_len = len(w3)

        return "".join(map(str, best))

    # Fallback for n > 7
    word = _build_superperm(n)
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
