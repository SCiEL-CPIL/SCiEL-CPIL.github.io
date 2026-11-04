"""Greedy set-cover construction for K_3(6,1)."""
import random
import itertools


def construct(q, n, radius):
    # The task contract fixes q=3, n=6, radius=1 for the football pool problem.
    # We implement a general greedy set-cover with random restarts for any q,n,radius.
    if radius < 0:
        radius = 0

    # Generate all words as integer IDs 0..q^n-1
    total = q ** n
    # Precompute neighbors: all words within Hamming distance <= radius
    # For radius=1 this is efficient: self + 1 position change per digit.
    # For general radius we use itertools.product over positions and symbols.

    def word_to_digits(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def digits_to_word(digits):
        w = 0
        for i in range(n):
            w = w * q + digits[i]
        return w

    # Precompute neighbors for each word
    # neighbors[x] = list of word IDs within distance <= radius
    # For efficiency, we compute all neighbors by iterating over all possible
    # combinations of position changes.

    if radius == 1:
        # Fast path for radius=1
        neighbors = []
        for x in range(total):
            digits = word_to_digits(x)
            nb = [x]
            for i in range(n):
                for sym in range(q):
                    if sym != digits[i]:
                        nd = digits[:]
                        nd[i] = sym
                        nb.append(digits_to_word(nd))
            neighbors.append(nb)
    else:
        # General case: enumerate all words within distance <= radius
        # This can be expensive but n is small (6)
        neighbors = []
        # Precompute all position subsets of size <= radius
        pos_subsets = []
        for r in range(radius + 1):
            for subset in itertools.combinations(range(n), r):
                pos_subsets.append(subset)
        for x in range(total):
            digits = word_to_digits(x)
            nb_set = set()
            for subset in pos_subsets:
                # For each position in subset, change to any other symbol
                # Use product to enumerate all combinations
                if len(subset) == 0:
                    nb_set.add(x)
                else:
                    for combo in itertools.product(range(q), repeat=len(subset)):
                        nd = digits[:]
                        for j, pos in enumerate(subset):
                            nd[pos] = combo[j]
                        nb_set.add(digits_to_word(nd))
            neighbors.append(list(nb_set))

    # Greedy set cover with random restarts
    best_code = None
    best_len = total  # upper bound: all words

    n_restarts = 20
    rng = random.Random(42)

    for restart in range(n_restarts):
        covered = [False] * total
        code = []
        uncovered_count = total

        while uncovered_count > 0:
            best_gain = 0
            best_candidates = []
            for c in range(total):
                gain = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(c)

            if not best_candidates or best_gain == 0:
                # Should not happen if uncovered_count > 0, but safety check
                break

            # Random pick from best candidates for diversity
            c_star = rng.choice(best_candidates)
            code.append(c_star)
            for w in neighbors[c_star]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1

        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]

    # Convert best_code integers to lists of digits
    result = [word_to_digits(x) for x in best_code]
    return result
