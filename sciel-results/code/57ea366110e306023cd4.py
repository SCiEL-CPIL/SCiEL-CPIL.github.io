"""Improved ternary covering code K_3(7,1) via randomized greedy set cover with time guard."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius.

    For the specific task q=3, n=7, radius=1, uses a randomized greedy set cover
    approach with multiple restarts and local improvement, guarded by a deadline.
    """
    start_time = time.time()
    deadline = start_time + 135  # 135s safety margin within 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct product construction for other parameters
        # For general case, use a simple greedy approach
        return _general_greedy(q, n, radius, deadline)

    # Specialized for q=3, n=7, radius=1
    total_words = 3 ** n  # 2187

    # Precompute: for each codeword, which words it covers (itself + neighbors at dist 1)
    # A word at position i can be changed to 2 other values, so each codeword covers 1 + 7*2 = 15 words

    # Convert word to index: base-3 encoding
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute coverage sets for all possible codewords
    # coverage[cw_idx] = set of word indices covered by codeword cw_idx
    # We'll represent coverage as a bitset or a list of indices
    # Since 2187 words, we can use a boolean array or a frozenset

    # For speed, precompute the neighbor indices for each position
    # For a word w, its neighbors at distance 1 are obtained by changing one coordinate
    # to one of the other 2 values

    # Let's precompute for each index and each position, the two neighbor indices
    # neighbors[idx][pos] = (neighbor_idx_1, neighbor_idx_2)
    # Actually, let's precompute the full coverage list for each possible codeword

    # There are 2187 possible codewords, each covers 15 words
    # Total coverage entries: 2187 * 15 = 32805, manageable

    # Precompute coverage for all 2187 possible words
    all_words = [idx_to_word(i) for i in range(total_words)]

    # For each possible codeword c, compute the set of words it covers
    # coverage[c] = sorted list of word indices covered
    coverage = []
    for c in range(total_words):
        cw = all_words[c]
        covered = set()
        covered.add(c)
        for pos in range(n):
            for new_val in range(q):
                if new_val == cw[pos]:
                    continue
                neighbor = cw[:]
                neighbor[pos] = new_val
                covered.add(word_to_idx(neighbor))
        coverage.append(frozenset(covered))

    # Now perform greedy set cover with randomization
    # We want to select as few codewords as possible to cover all 2187 words

    def greedy_cover(seed):
        """Greedy set cover with randomized tie-breaking."""
        rng = random.Random(seed)
        covered = [False] * total_words
        selected = []
        remaining = total_words

        while remaining > 0:
            # Find the codeword that covers the most uncovered words
            # With randomization among ties
            best_count = -1
            best_candidates = []
            for c in range(total_words):
                # Count uncovered words covered by c
                cnt = 0
                for w in coverage[c]:
                    if not covered[w]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)

            if best_count == 0:
                break  # Should not happen if we have a valid cover

            # Randomly pick from best candidates
            chosen = rng.choice(best_candidates)
            selected.append(chosen)

            # Mark covered
            for w in coverage[chosen]:
                if not covered[w]:
                    covered[w] = True
                    remaining -= 1

        return selected

    def try_remove(selected):
        """Try to remove redundant codewords from the selected set."""
        selected_set = set(selected)
        changed = True
        while changed:
            changed = False
            for i, cw in enumerate(selected):
                # Check if removing cw still covers all words
                # Only need to check words that were covered by cw
                # and see if they're covered by other selected codewords
                can_remove = True
                for w in coverage[cw]:
                    # Check if w is covered by some other selected codeword
                    covered_by_other = False
                    for j, other_cw in enumerate(selected):
                        if j == i:
                            continue
                        if w in coverage[other_cw]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        can_remove = False
                        break
                if can_remove:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    def local_search(selected, seed):
        """Try to swap codewords to reduce the count."""
        rng = random.Random(seed)
        selected = list(selected)
        improved = True
        while improved:
            improved = False
            # Try removing one codeword and adding a different one
            for i in range(len(selected)):
                if time.time() > deadline - 5:
                    return selected
                # Temporarily remove selected[i]
                removed = selected.pop(i)
                # Check what's now uncovered
                uncovered = []
                for w in coverage[removed]:
                    is_covered = False
                    for cw in selected:
                        if w in coverage[cw]:
                            is_covered = True
                            break
                    if not is_covered:
                        uncovered.append(w)
                if not uncovered:
                    # Can remove without adding
                    improved = True
                    continue
                # Try to add a codeword that covers all uncovered
                found = False
                for c in range(total_words):
                    if c in selected:
                        continue
                    if all(w in coverage[c] for w in uncovered):
                        selected.append(c)
                        found = True
                        improved = True
                        break
                if not found:
                    selected.insert(i, removed)

        # Try removals again
        selected = try_remove(selected)
        return selected

    # Run multiple restarts
    best_solution = None
    best_count = float('inf')

    # First, get a deterministic greedy solution as baseline
    baseline = greedy_cover(0)
    baseline = try_remove(baseline)
    best_solution = baseline
    best_count = len(baseline)

    # Now try randomized restarts
    restarts = 0
    while time.time() < deadline - 10:
        seed = 1000 + restarts
        solution = greedy_cover(seed)
        solution = try_remove(solution)
        if len(solution) < best_count:
            best_solution = solution
            best_count = len(solution)

        # Try local search on the best solution
        if time.time() < deadline - 10:
            improved = local_search(best_solution, seed)
            if len(improved) < best_count:
                best_solution = improved
                best_count = len(improved)

        restarts += 1

    # Convert selected indices to word lists
    return [all_words[c] for c in best_solution]


def _general_greedy(q, n, radius, deadline):
    """General greedy covering code construction for arbitrary q, n, radius."""
    total_words = q ** n
    if total_words > 1000000:
        # Too large for exhaustive approach, use a simple construction
        # Direct product or random sampling
        import itertools
        # Use a simple random sampling approach
        rng = random.Random(42)
        selected = []
        covered = set()
        all_words_list = list(itertools.product(range(q), repeat=n))
        rng.shuffle(all_words_list)

        for word in all_words_list:
            if time.time() > deadline:
                break
            # Check if this word adds new coverage
            new_coverage = 0
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    neighbor = list(word)
                    neighbor[pos] = val
                    if tuple(neighbor) not in covered:
                        new_coverage += 1
            if new_coverage > 0:
                selected.append(list(word))
                # Add all words within radius to covered
                _add_covered(word, q, n, radius, covered)

        return selected

    # For smaller spaces, use greedy
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def _add_covered(word, q, n, radius, covered):
        import itertools
        for dist in range(radius + 1):
            for pos in itertools.combinations(range(n), dist):
                for vals in itertools.product(range(q), repeat=dist):
                    if all(v == word[p] for p, v in zip(pos, vals)):
                        continue
                    neighbor = list(word)
                    for p, v in zip(pos, vals):
                        neighbor[p] = v
                    covered.add(tuple(neighbor))

    all_words = [idx_to_word(i) for i in range(total_words)]
    coverage = []
    for c in range(total_words):
        cw = all_words[c]
        covered = set()
        _add_covered(cw, q, n, radius, covered)
        coverage.append(frozenset(covered))

    # Greedy
    covered = [False] * total_words
    selected = []
    remaining = total_words

    while remaining > 0 and time.time() < deadline:
        best_count = -1
        best_c = -1
        for c in range(total_words):
            cnt = sum(1 for w in coverage[c] if not covered[w])
            if cnt > best_count:
                best_count = cnt
                best_c = c
        if best_count == 0:
            break
        selected.append(best_c)
        for w in coverage[best_c]:
            if not covered[w]:
                covered[w] = True
                remaining -= 1

    return [all_words[c] for c in selected]
# EVOLVE-BLOCK-END
