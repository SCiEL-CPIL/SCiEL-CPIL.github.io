import time
import random
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    # --- 1) Classic recursive superpermutation (length 1!+2!+...+n!) ---
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            w = tuple(word[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                perms.append(w)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            lo = min(len(new), len(block))
            o = 0
            for o in range(lo, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    break
            new += block[o:]
        word = new
    # Classic construction already contains every permutation, but it is not
    # short. We now try to shrink it.
    if n <= 2:
        return "".join(map(str, word))

    # --- 2) Greedy missing-permutation insertion search ---
    # Track which permutations appear as windows of length n.
    idx = 0
    perm_set = set()
    for i in range(len(word) - n + 1):
        t = tuple(word[i:i + n])
        perm_set.add(t)
        idx = max(idx, i)
    for i in range(idx + 1, len(word) - n + 1):
        perm_set.add(tuple(word[i:i + n]))

    total = 1
    for k in range(2, n + 1):
        total *= k
    target = set(permutations(range(1, n + 1)))
    missing = target - perm_set

    # Candidate insertion points: after every distinct (n-1)-window that appears.
    # Inserting a symbol at such a boundary can only "break" windows that end at
    # that boundary, which are exactly the windows that were just removed.
    pos_set = set()
    for i in range(len(word) - (n - 1) + 1):
        if i + 1 < len(word) and i + n <= len(word):
            pos_set.add(i + 1)  # position after index i
    positions = list(pos_set)

    best_word = word
    best_missing = missing
    best_len = len(word)

    start = time.time()
    deadline = start + 110.0  # safety margin well under the 120s limit

    rng = random.Random(20260827)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(missing) == 0:
            break
        # Try each missing permutation, each position (shuffled), each symbol.
        missing_list = list(missing)
        rng.shuffle(missing_list)
        for p in missing_list:
            if time.time() > deadline:
                break
            # Windows that would be removed by an insertion at boundary j.
            # Build the set of windows that end at each boundary we care about.
            # We'll compute lazily per boundary.
            for j in positions:
                if time.time() > deadline:
                    break
                # Remove windows ending at boundary j-1 and j.
                # window ending at index j-1: word[j-n:j]  (valid if j-n>=0)
                removed = []
                if j - n >= 0:
                    removed.append(tuple(word[j - n:j]))
                # window ending at index j: word[j+1-j-1... ] i.e. word[j+1 - n +1?] 
                # window of length n ending at index (j+1-1)=j is word[j+1-n:j+1]? 
                # Wait: after inserting one symbol, the window that previously
                # started at index j (word[j:j+n]) is now broken (its last n-1
                # shift). That window is word[j:j+n].
                if j + n <= len(word):
                    removed.append(tuple(word[j:j + n]))
                # Try each symbol s; check that every removed window is either in
                # perm_set or in missing (i.e. we don't newly lose a kept one that
                # isn't already accounted). We insert s then compute new windows.
                for s in range(1, n + 1):
                    # new windows created by inserting s at position j:
                    #  window ending at new symbol s: word[j-1-n+1..j] + s -> word[j-n:j] + [s]
                    new1 = tuple(word[j - n:j]) + (s,) if j - n >= 0 else None
                    #  window starting at s: [s] + word[j:j+1-n+1] -> [s] + word[j:j+1]
                    new2 = (s,) + tuple(word[j:j + 1]) if j + 1 <= len(word) else None
                    # check no removed window is uniquely kept
                    ok = True
                    for r in removed:
                        if r in perm_set and r not in missing:
                            # losing a window that is currently present and not
                            # in missing means we'd drop a valid perm we have.
                            # But we only lose it if it isn't recreated. It isn't
                            # recreated (new1/new2 are the only new windows).
                            if r != new1 and r != new2:
                                ok = False
                                break
                    if not ok:
                        continue
                    # apply
                    for r in removed:
                        perm_set.discard(r)
                        if r in missing:
                            missing.discard(r)
                    if new1 is not None:
                        if new1 in missing:
                            missing.discard(new1)
                        perm_set.add(new1)
                    if new2 is not None:
                        if new2 in missing:
                            missing.discard(new2)
                        perm_set.add(new2)
                    # update word
                    word.insert(j, s)
                    # shift positions >= j by +1
                    positions = [j + 1 if p >= j else p for p in positions]
                    # add the two new boundaries around the inserted symbol
                    if j not in positions:
                        positions.append(j)
                    if j + 1 not in positions:
                        positions.append(j + 1)
                    # record improvement
                    if len(missing) < best_missing is not None and len(missing) < len(best_missing):
                        pass
                    if len(word) < best_len or len(missing) < len(best_missing):
                        best_word = list(word)
                        best_missing = set(missing)
                        best_len = len(word)
                        improved = True
                    break  # accept first valid symbol for this (p, j)
                # re-check deadline
        # (loop continues only if improved)

    # Final deadline guard: return the best result found.
    result = best_word if best_word is not None else word
    return "".join(map(str, result))
