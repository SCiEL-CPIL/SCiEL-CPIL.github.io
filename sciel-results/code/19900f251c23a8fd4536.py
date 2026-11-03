"""Ternary covering code K_3(8,1) via greedy maximum-coverage + lazy-deletion local search."""
import time
import random
import itertools
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    if q != 3 or n != 8 or radius != 1:
        total = q ** n
        if total <= 200000:
            return [list(x) for x in itertools.product(range(q), repeat=n)]
        else:
            random.seed(42)
            codewords = []
            seen = set()
            while len(codewords) < 2000:
                w = tuple(random.randint(0, q - 1) for _ in range(n))
                if w not in seen:
                    seen.add(w)
                    codewords.append(list(w))
            return codewords

    # Specialized for q=3, n=8, radius=1
    # Total words: 3^8 = 6561
    # Each codeword covers 1 + 8*2 = 17 words
    # Lower bound on codewords: ceil(6561/17) = 387

    # Precompute neighbor indices for all words
    def word_to_idx(w):
        idx = 0
        for i, v in enumerate(w):
            idx += v * (q ** (n - 1 - i))
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    # Precompute neighbors for each word index
    neighbor_indices = np.zeros((q ** n, 17), dtype=np.int32)
    for w in itertools.product(range(q), repeat=n):
        idx = word_to_idx(w)
        neighbor_indices[idx, 0] = idx
        pos = 0
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    neighbor_indices[idx, pos + 1] = word_to_idx(tuple(new_w))

    # Coverage count: how many codewords cover each word
    covered_count = np.zeros(q ** n, dtype=np.int32)

    def add_codeword(cw):
        idx = word_to_idx(cw)
        covered_count[neighbor_indices[idx]] += 1

    def remove_codeword(cw):
        idx = word_to_idx(cw)
        covered_count[neighbor_indices[idx]] -= 1

    def is_removable(cw):
        idx = word_to_idx(cw)
        return np.all(covered_count[neighbor_indices[idx]] >= 2)

    # Step 1: Greedy construction - repeatedly add the codeword that covers the most uncovered words
    codewords = []
    cw_set = set()

    # Use a priority queue with lazy deletion for greedy
    # Each entry: (-gain, codeword)
    # We need to compute gain for each candidate

    # For efficiency, we'll compute gains periodically
    # Initial: all words are uncovered

    def compute_gain(cw):
        idx = word_to_idx(cw)
        return int(np.sum(covered_count[neighbor_indices[idx]] == 0))

    # Greedy: add codeword with max gain
    # We'll do this in batches to keep it fast
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]

    # Use a simple greedy with periodic re-evaluation
    # First pass: add codewords in a structured order, then refine
    # Start with tetracode-based structure for good initial coverage
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))

    # Add tetracode product as initial set
    initial = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial.append(h + tail)
    initial = list(set(initial))

    for cw in initial:
        add_codeword(cw)
        cw_set.add(cw)
    codewords = list(initial)

    # Step 2: Remove redundant codewords (iterative)
    removed = True
    while removed:
        if time.time() > deadline - 40:
            break
        removed = False
        for i in range(len(codewords) - 1, -1, -1):
            if is_removable(codewords[i]):
                remove_codeword(codewords[i])
                cw_set.discard(codewords[i])
                codewords.pop(i)
                removed = True

    best_codewords = list(codewords)
    best_size = len(codewords)

    # Step 3: Local search with add-best-remove-worst using lazy deletion
    # Maintain a max-heap of (gain, codeword) for candidate additions
    # and a min-heap of (cost, codeword) for removable codewords

    # For the local search, we'll do:
    # 1. Try to remove any removable codeword
    # 2. If stuck, try swapping: remove one, add one that covers the gaps

    random.seed(42)

    for iteration in range(50000):
        if time.time() > deadline - 10:
            break

        # First, try to remove redundant codewords
        found_removal = False
        for i in range(len(codewords) - 1, -1, -1):
            if is_removable(codewords[i]):
                remove_codeword(codewords[i])
                cw_set.discard(codewords[i])
                codewords.pop(i)
                found_removal = True
                break

        if found_removal:
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
            continue

        # Try swap: pick a random codeword, see if we can find a better replacement
        if len(codewords) == 0:
            break

        idx = random.randint(0, len(codewords) - 1)
        cw = codewords[idx]

        # Find words that would become uncovered if we remove cw
        cw_idx = word_to_idx(cw)
        critical_words = neighbor_indices[cw_idx][covered_count[neighbor_indices[cw_idx]] == 1]

        if len(critical_words) == 0:
            # Shouldn't happen since we already checked removability
            continue

        # Find a candidate that covers all critical_words and isn't already in cw_set
        # Sample candidates
        best_new = None
        best_extra = -1

        for _ in range(50):
            new_cw = tuple(random.randint(0, q - 1) for _ in range(n))
            if new_cw in cw_set:
                continue

            new_idx = word_to_idx(new_cw)
            new_neighbors = neighbor_indices[new_idx]

            # Check if new_cw covers all critical_words
            # critical_words are indices, so check if each is in new_neighbors
            covers_all = True
            for cw_word in critical_words:
                if cw_word not in new_neighbors:
                    covers_all = False
                    break

            if not covers_all:
                continue

            # Compute extra coverage: how many words does new_cw cover that aren't covered by others?
            extra = int(np.sum(covered_count[new_neighbors] == 1))  # words that would only be covered by new_cw after removal

            if extra > best_extra:
                best_extra = extra
                best_new = new_cw

        if best_new is not None:
            # Perform swap
            remove_codeword(cw)
            cw_set.discard(cw)
            codewords.pop(idx)
            add_codeword(best_new)
            cw_set.add(best_new)
            codewords.append(best_new)

            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)

    # Final verification
    final_covered = np.zeros(q ** n, dtype=bool)
    for cw in best_codewords:
        idx = word_to_idx(cw)
        final_covered[neighbor_indices[idx]] = True

    if not np.all(final_covered):
        # Fallback: use the initial construction
        return [list(w) for w in initial]

    return [list(w) for w in best_codewords]
