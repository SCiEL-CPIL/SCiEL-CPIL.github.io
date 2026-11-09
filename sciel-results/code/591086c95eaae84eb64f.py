"""Football pool K_3(7,1): bitset local search with greedy construction."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    total_words = q ** n  # 2187
    all_words = [list(w) for w in product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

    # Precompute neighbors: for each word, all words within Hamming distance <= radius
    # For radius=1, q=3, n=7: each word has 1 + 7*2 = 15 neighbors
    neighbors = []
    for idx, w in enumerate(all_words):
        nbrs = set()
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = w[:]
                    nw[pos] = val
                    nbrs.add(tuple(nw))
        nbrs.add(tuple(w))
        neighbors.append(nbrs)

    # For efficient bitset operations, map each word to a bit position
    # Use Python integers as bitsets (2187 bits fits easily)
    # Precompute neighbor bitsets
    neighbor_bitsets = []
    for idx in range(total_words):
        bs = 0
        for nb in neighbors[idx]:
            nb_idx = word_to_idx[nb]
            bs |= (1 << nb_idx)
        neighbor_bitsets.append(bs)

    def greedy_construct():
        """Greedy: repeatedly pick the word covering the most uncovered words."""
        uncovered = (1 << total_words) - 1
        code = []
        while uncovered:
            best_word = -1
            best_count = 0
            # Sample or scan all words
            for idx in range(total_words):
                if not (uncovered & (1 << idx)):
                    continue
                covered = neighbor_bitsets[idx] & uncovered
                count = covered.bit_count()
                if count > best_count:
                    best_count = count
                    best_word = idx
            if best_word == -1:
                break
            code.append(best_word)
            uncovered &= ~neighbor_bitsets[best_word]
        return code

    def local_search(code, time_limit):
        """Try to remove codewords or swap to reduce size while maintaining coverage."""
        code_set = set(code)
        covered = 0
        for c in code:
            covered |= neighbor_bitsets[c]

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removal
            for i in range(len(code)):
                c = code[i]
                new_covered = covered & ~neighbor_bitsets[c]
                if new_covered == ((1 << total_words) - 1):
                    code.pop(i)
                    covered = new_covered
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove one, add another
            for i in range(len(code)):
                c_remove = code[i]
                base_covered = covered & ~neighbor_bitsets[c_remove]
                # Find a word that covers the missing part
                missing = ((1 << total_words) - 1) & ~base_covered
                if missing == 0:
                    # Can just remove
                    code.pop(i)
                    covered = base_covered
                    improved = True
                    break
                # Try adding each uncovered word
                for j in range(total_words):
                    if j in code_set:
                        continue
                    if not (neighbor_bitsets[j] & missing):
                        continue
                    # Check if adding j covers all missing
                    if (neighbor_bitsets[j] & missing) == missing:
                        code[i] = j
                        code_set.discard(c_remove)
                        code_set.add(j)
                        covered = base_covered | neighbor_bitsets[j]
                        improved = True
                        break
                if improved:
                    break
        return code

    # Phase 1: Greedy construction
    code = greedy_construct()
    code = local_search(code, time.time() + 5.0)
    best_code = code[:]
    best_len = len(code)

    # Phase 2: Multiple restarts with randomization
    while time.time() < DEADLINE:
        # Randomized greedy
        uncovered = (1 << total_words) - 1
        code = []
        while uncovered:
            best_word = -1
            best_count = 0
            # Add some randomness: consider top candidates
            candidates = []
            for idx in range(total_words):
                if not (uncovered & (1 << idx)):
                    continue
                covered = neighbor_bitsets[idx] & uncovered
                count = covered.bit_count()
                if count > best_count:
                    candidates = [(count, idx)]
                    best_count = count
                elif count == best_count:
                    candidates.append((count, idx))
            if best_word == -1 and candidates:
                best_word = random.choice(candidates)[1]
            elif candidates:
                best_word = random.choice(candidates)[1]
            else:
                break
            code.append(best_word)
            uncovered &= ~neighbor_bitsets[best_word]

        code = local_search(code, time.time() + 3.0)
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]

        # Check time
        if time.time() > DEADLINE:
            break

    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
