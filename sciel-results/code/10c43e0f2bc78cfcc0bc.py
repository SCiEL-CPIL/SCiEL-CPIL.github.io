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
    perm_set = set()
    for i in range(len(word) - n + 1):
        perm_set.add(tuple(word[i:i + n]))

    total = 1
    for k in range(2, n + 1):
        total *= k
    target = set(permutations(range(1, n + 1)))
    missing = target - perm_set

    # Candidate insertion points: positions 1..len(word)-1 (between symbols)
    # We'll use a set for O(1) membership and a list for iteration
    positions = list(range(1, len(word)))

    best_word = list(word)
    best_missing = set(missing)
    best_len = len(word)

    start = time.time()
    deadline = start + 140.0  # safety margin under the 150s limit

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
            # For each candidate position, try to insert a symbol that creates
            # a new window matching some missing permutation
            pos_sample = list(positions)
            rng.shuffle(pos_sample)
            for j in pos_sample:
                if time.time() > deadline:
                    break
                if j < 1 or j >= len(word):
                    continue
                # When we insert symbol s at position j (0-indexed, before word[j]):
                # The ONLY window that is destroyed is the one that straddles
                # the insertion point: word[j-n+1 : j+1] if valid
                # New windows created:
                #   W1 = word[j-n+1:j] + [s]   (if j-n+1 >= 0)
                #   W2 = [s] + word[j:j+n-1]   (if j+n-1 <= len(word))
                
                # Check if inserting any symbol creates a window in missing
                for s in range(1, n + 1):
                    # Compute the destroyed window (if any)
                    destroyed = None
                    if j - n + 1 >= 0 and j + 1 <= len(word):
                        destroyed = tuple(word[j - n + 1:j + 1])
                    
                    # Compute new windows
                    new1 = None
                    if j - n + 1 >= 0:
                        new1 = tuple(word[j - n + 1:j]) + (s,)
                    new2 = None
                    if j + n - 1 <= len(word):
                        new2 = (s,) + tuple(word[j:j + n - 1])
                    
                    # Check: destroyed window must not be a unique kept permutation
                    ok = True
                    if destroyed is not None:
                        if destroyed in perm_set and destroyed not in missing:
                            # We'd lose this permutation unless it's recreated
                            if destroyed != new1 and destroyed != new2:
                                ok = False
                    
                    if not ok:
                        continue
                    
                    # Apply the insertion
                    if destroyed is not None:
                        perm_set.discard(destroyed)
                        if destroyed in missing:
                            missing.discard(destroyed)
                    
                    if new1 is not None:
                        if new1 in missing:
                            missing.discard(new1)
                        perm_set.add(new1)
                    if new2 is not None:
                        if new2 in missing:
                            missing.discard(new2)
                        perm_set.add(new2)
                    
                    # Update word
                    word.insert(j, s)
                    
                    # Update positions: all positions >= j shift by +1
                    # Add new positions j and j+1
                    new_positions = []
                    for pos in positions:
                        if pos >= j:
                            new_positions.append(pos + 1)
                        else:
                            new_positions.append(pos)
                    if j not in new_positions:
                        new_positions.append(j)
                    if j + 1 not in new_positions and j + 1 < len(word):
                        new_positions.append(j + 1)
                    positions = new_positions
                    
                    # Record improvement
                    if len(missing) < len(best_missing) or (len(missing) == len(best_missing) and len(word) < best_len):
                        best_word = list(word)
                        best_missing = set(missing)
                        best_len = len(word)
                        improved = True
                    break  # accept first valid symbol for this position
                # re-check deadline
        # (loop continues only if improved)

    # Final deadline guard: return the best result found.
    result = best_word if best_word is not None else word
    return "".join(map(str, result))
