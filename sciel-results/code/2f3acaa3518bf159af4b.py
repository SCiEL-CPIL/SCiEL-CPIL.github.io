"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    # Deadline guard: the evaluator kills after 150s wall-clock; keep a safety margin.
    deadline = time.time() + 140.0

    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum of ternary tetracode with full product on remaining coords.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # For n < 4, just use all words (trivially valid, large).
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute all 2187 words as tuples for fast hashing.
    all_words = list(itertools.product(range(3), repeat=7))
    word_set = set(all_words)

    # Build the "ball" (codeword + all words at Hamming distance 1) for each possible word.
    # ball[w] = frozenset of words covered by choosing w as a codeword.
    # A word covers itself + 7*2=14 neighbors = 15 words.
    def neighbors(w):
        wlist = list(w)
        nbrs = [w]
        for i in range(7):
            orig = wlist[i]
            for v in range(3):
                if v == orig:
                    continue
                wlist[i] = v
                nbrs.append(tuple(wlist))
            wlist[i] = orig
        return nbrs

    # Precompute balls
    balls = {}
    for w in all_words:
        balls[w] = frozenset(neighbors(w))

    # Greedy set cover: start with all words, repeatedly remove the one whose removal
    # leaves the fewest uncovered words (or equivalently, start empty and add the one
    # that covers the most uncovered words).
    # Strategy: Greedy add -- start with empty set, add the word covering most uncovered.
    uncovered = set(all_words)
    code = []
    # For speed, we'll track for each word how many code words cover it.
    # Actually, let's do the standard greedy: pick word maximizing |ball[w] ∩ uncovered|.
    # To speed up, we can precompute ball sizes (all 15) and use a priority approach.
    # But 2187 words * 15 = ~33k operations per greedy step, and we need ~186 steps.
    # That's ~6M ops, fine.

    # Use a list of (word, ball) for iteration.
    word_list = list(all_words)
    # We'll use a simple greedy: at each step, find word with max coverage of uncovered.
    # Optimization: since all balls have size 15, we just count how many of those 15 are in uncovered.
    while uncovered:
        best_word = None
        best_count = -1
        for w in word_list:
            b = balls[w]
            # Count how many elements of b are in uncovered
            cnt = 0
            for x in b:
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                if best_count == 15:  # Can't do better
                    break
        if best_word is None:
            break
        code.append(best_word)
        # Remove covered words from uncovered
        for x in balls[best_word]:
            uncovered.discard(x)
        if time.time() > deadline - 30:
            break

    # Now local search: try to remove redundant codewords.
    # A codeword c is redundant if every word in ball[c] is covered by some other codeword.
    def is_redundant(idx, code_list):
        c = code_list[idx]
        for x in balls[c]:
            covered = False
            for j, cj in enumerate(code_list):
                if j == idx:
                    continue
                if x in balls[cj]:
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Try removals repeatedly
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for i in range(len(code)):
            if is_redundant(i, code):
                code.pop(i)
                changed = True
                break  # restart scan after removal

    # Local search: try swapping a codeword for a better one (one that covers more uncovered-ish words)
    # Actually, after greedy + removal, try: for each codeword, see if replacing it with
    # a nearby word reduces total coverage overlap.
    # Simpler: try adding a word that covers the most currently-uncovered-by-unique-coverage words.
    # Let's do a "repair" pass: check if still valid, if not, add greedy.
    # Verify validity:
    def is_valid(code_list):
        if not code_list:
            return False
        # Check every word is covered
        covered = set()
        for c in code_list:
            covered.update(balls[c])
        return covered == word_set

    # If not valid (shouldn't happen), add greedy
    if not is_valid(code) and time.time() < deadline - 10:
        # Rebuild uncovered
        covered = set()
        for c in code:
            covered.update(balls[c])
        uncovered = word_set - covered
        while uncovered and time.time() < deadline - 5:
            best_word = None
            best_count = -1
            for w in word_list:
                b = balls[w]
                cnt = 0
                for x in b:
                    if x in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
                    if best_count == 15:
                        break
            if best_word is None:
                break
            code.append(best_word)
            for x in balls[best_word]:
                uncovered.discard(x)

    # Final: ensure distinct
    seen = set()
    unique_code = []
    for c in code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique_code.append(list(c))

    # If somehow invalid, fall back to full product (shouldn't happen)
    if not is_valid(unique_code):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]

    return unique_code
# EVOLVE-BLOCK-END
