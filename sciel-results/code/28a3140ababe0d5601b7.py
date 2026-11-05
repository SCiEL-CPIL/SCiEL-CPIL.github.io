"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def construct(q, n, radius):
    # Deadline guard: the evaluator kills at 150s wall-clock.
    # We use a safety margin to ensure we return before then.
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # The task is specifically for q=3, n=6, radius=1 (ternary covering code K_3(6,1)).
    # The best known code has 73 words, lower bound is 71.
    # The seed program uses direct sum of tetracode: 9 * 3^(6-4) = 81 words.
    # We need to improve upon 81.

    # Strategy: Start with a good initial code (tetracode direct sum = 81 words),
    # then use local search to try to reduce the number of codewords while maintaining coverage.

    # First, generate all words in the space for coverage checking
    # For q=3, n=6: 3^6 = 729 words. This is manageable.

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: use the tetracode construction if possible
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, just use all words
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute all words as tuples for fast lookup
    all_words = list(itertools.product(range(3), repeat=6))
    word_set = set(all_words)

    # Precompute neighbors: for each word, the set of words within Hamming distance 1
    # This includes the word itself and all words that differ in exactly one position
    neighbors = {}
    for w in all_words:
        w_list = list(w)
        neigh = set()
        neigh.add(w)  # itself
        for pos in range(6):
            orig = w_list[pos]
            for val in range(3):
                if val != orig:
                    w_list[pos] = val
                    neigh.add(tuple(w_list))
            w_list[pos] = orig
        neighbors[w] = neigh

    # Build reverse mapping: for each word, which words cover it (i.e., are within distance 1)
    # Actually, we need: for a set of codewords, check if every word is covered.
    # A word is covered if it's within distance 1 of some codeword.

    # Helper: check if a set of codewords covers all words
    def is_valid(codewords):
        covered = set()
        for cw in codewords:
            cw_tuple = tuple(cw)
            covered.update(neighbors[cw_tuple])
        return len(covered) == len(all_words)

    # Initial code: tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    # 9 * 9 = 81 words

    # Verify initial code is valid
    if not is_valid(initial_code):
        # Shouldn't happen, but fallback
        return initial_code

    best_code = list(initial_code)
    best_size = len(best_code)

    # Local search: try to remove codewords one at a time if coverage is maintained
    # Then try to add/remove pairs, or swap

    # Phase 1: Greedy removal
    # Try to remove each codeword; if coverage is maintained, keep it removed.
    code = list(initial_code)
    improved = True
    while improved:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            # Try removing code[i]
            candidate = code[:i] + code[i+1:]
            if is_valid(candidate):
                code = candidate
                improved = True
                best_code = list(code)
                best_size = len(code)
                break  # Restart from beginning since indices shifted
        if time.time() > deadline - 5:
            break

    # Phase 2: Try swapping a codeword with another word that might cover more
    # For each codeword, try replacing it with a different word
    # This is expensive, so we limit the search

    # Better approach: use simulated annealing or greedy with swaps
    # Try: for each codeword, remove it, then try to find a word that covers all the words
    # that were only covered by it.

    # Let's use a more systematic approach:
    # For each codeword, compute which words are ONLY covered by it (critical words).
    # If we remove it, we need to cover those critical words with other codewords.

    def get_critical_words(code, idx):
        """Find words that are only covered by code[idx]."""
        # Words covered by code[idx]
        cw_tuple = tuple(code[idx])
        covered_by_idx = neighbors[cw_tuple]

        # Words covered by all other codewords
        others_covered = set()
        for j in range(len(code)):
            if j != idx:
                others_covered.update(neighbors[tuple(code[j])])

        # Critical words: covered by idx but not by others
        critical = covered_by_idx - others_covered
        return critical

    # Phase 2: Try removing codewords that have few critical words,
    # and see if we can cover those critical words by modifying other codewords.
    # This is complex, so let's try a simpler heuristic:
    # Randomly perturb the code and see if we can reduce size.

    # Phase 3: Local search with swaps
    # For each pair of codewords, try swapping one with a random word
    # and see if coverage improves or size decreases.

    # Actually, let's try a different approach:
    # Start with the 81-word code, and try to find a smaller valid code
    # by iteratively removing words and checking coverage.

    # Let's also try to use the fact that K_3(6,1) = 73 is known.
    # We can try to construct a 73-word code directly.

    # For now, let's do a more aggressive local search:
    # 1. Try all single removals (done in Phase 1)
    # 2. Try all single additions followed by removals
    # 3. Try swaps

    # Phase 2: Try replacing each codeword with a random word
    # If the new code is valid and smaller, keep it.
    # This is a form of local search.

    # Let's use a time-limited search:
    # While time remains, try random perturbations.

    random.seed(42)  # For reproducibility

    # Phase 2: Random perturbation search
    while time.time() < deadline - 10:
        # Try a random perturbation
        # Option 1: Remove a random codeword
        # Option 2: Add a random word, then try to remove others
        # Option 3: Swap a codeword with a random word

        # Let's try: remove a random codeword, then add a random word if needed
        idx = random.randint(0, len(code) - 1)
        candidate = list(code)
        candidate.pop(idx)

        # Check if valid
        if is_valid(candidate):
            code = candidate
            best_code = list(code)
            best_size = len(code)
            continue

        # Not valid, so try to fix by adding a word
        # Find uncovered words
        covered = set()
        for cw in candidate:
            covered.update(neighbors[tuple(cw)])
        uncovered = set(all_words) - covered

        if uncovered:
            # Try to cover uncovered words by adding one word
            # Find a word that covers the most uncovered words
            best_word = None
            best_count = 0
            for w in uncovered:
                # How many uncovered words does w cover?
                count = len(neighbors[tuple(w)] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = w
            if best_word is not None:
                candidate.append(list(best_word))
                if is_valid(candidate):
                    # Same size, so no improvement, but might help future removals
                    # Don't update best, but keep candidate for further search
                    code = candidate
                    continue

        # Option: swap a codeword with a random word
        idx = random.randint(0, len(code) - 1)
        candidate = list(code)
        new_word = list(random.choice(all_words))
        candidate[idx] = new_word
        if is_valid(candidate):
            code = candidate
            # Same size, but might help future removals
            continue

        # Option: try to remove two codewords and add one
        if len(code) >= 2:
            i = random.randint(0, len(code) - 1)
            j = random.randint(0, len(code) - 1)
            if i == j:
                continue
            candidate = list(code)
            # Remove the higher index first to keep indices valid
            if i > j:
                i, j = j, i
            candidate.pop(j + 1)  # Remove j+1 (which was originally j+1, but after popping j, it's j)
            # Wait, this is getting complicated. Let's just remove two and check.
            candidate = list(code)
            candidate.pop(max(i, j))
            candidate.pop(min(i, j))
            # Now try to add one word to cover
            covered = set()
            for cw in candidate:
                covered.update(neighbors[tuple(cw)])
            uncovered = set(all_words) - covered
            if uncovered:
                best_word = None
                best_count = 0
                for w in uncovered:
                    count = len(neighbors[tuple(w)] & uncovered)
                    if count > best_count:
                        best_count = count
                        best_word = w
                if best_word is not None:
                    candidate.append(list(best_word))
                    if is_valid(candidate):
                        code = candidate
                        best_code = list(code)
                        best_size = len(code)
                        continue

    # Final check: ensure best_code is valid
    if not is_valid(best_code):
        # Fallback to initial code
        return initial_code

    return best_code
# EVOLVE-BLOCK-END
