"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _neighbors_of(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    yield tuple(word)
    if radius >= 1:
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                yield tuple(new_word)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin within 150s limit

    total_words = q ** n

    # Precompute all words and neighbor indices
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor index lists for each word
    neighbor_indices = []
    for w in all_words:
        nbs = list(_neighbors_of(w, q, n, radius))
        neighbor_indices.append([word_to_idx[nei] for nei in nbs])

    # Also precompute neighbor sets for O(1) lookup
    neighbor_sets = [set(ni) for ni in neighbor_indices]

    # Coverage count array
    coverage = np.zeros(total_words, dtype=np.int32)

    # --- Step 1: Start with tetracode product construction ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    if n >= 4:
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                initial_codewords.append(tuple(h + list(tail)))
    else:
        initial_codewords = []

    initial_indices = [word_to_idx[w] for w in initial_codewords if w in word_to_idx]

    # Build coverage from initial code
    for idx in initial_indices:
        for ni in neighbor_indices[idx]:
            coverage[ni] += 1

    codeword_indices = list(initial_indices)
    in_code = set(codeword_indices)

    # --- Step 2: Local search to reduce code size ---
    def try_remove(i):
        """Try to remove codeword at position i. Returns True if removed."""
        if time.time() > deadline - 5.0:
            return False
        idx = codeword_indices[i]
        for ni in neighbor_indices[idx]:
            if coverage[ni] <= 1:
                return False
        for ni in neighbor_indices[idx]:
            coverage[ni] -= 1
        codeword_indices.pop(i)
        in_code.discard(idx)
        return True

    def try_swap(i, new_idx):
        """Try to swap codeword at position i with new_idx. Returns True if swapped."""
        if time.time() > deadline - 5.0:
            return False
        old_idx = codeword_indices[i]
        if new_idx == old_idx or new_idx in in_code:
            return False

        would_uncover = []
        for ni in neighbor_indices[old_idx]:
            if coverage[ni] <= 1:
                would_uncover.append(ni)

        new_neighbors = neighbor_sets[new_idx]
        for w in would_uncover:
            if w not in new_neighbors:
                return False

        for ni in neighbor_indices[old_idx]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx]:
            coverage[ni] += 1
        codeword_indices[i] = new_idx
        in_code.discard(old_idx)
        in_code.add(new_idx)
        return True

    def find_candidates_for_exclusive(exclusive_words, in_code_set, max_candidates=200):
        """Find candidate codewords that cover all exclusive words.
        
        Uses a targeted search: for each exclusive word, find candidates that
        are within radius of it, then intersect the sets.
        """
        if not exclusive_words:
            return []
        
        # Start with candidates that cover the first exclusive word
        first_excl = exclusive_words[0]
        # Find all words within distance radius of first_excl
        # These are potential candidates
        first_excl_tuple = all_words[first_excl]
        candidates = set()
        
        # Generate all words within distance radius of first_excl
        for cand_word in _neighbors_of(first_excl_tuple, q, n, radius):
            cand_idx = word_to_idx.get(cand_word)
            if cand_idx is not None and cand_idx not in in_code_set:
                candidates.add(cand_idx)
        
        # Intersect with candidates covering other exclusive words
        for excl in exclusive_words[1:]:
            excl_tuple = all_words[excl]
            next_candidates = set()
            for cand_word in _neighbors_of(excl_tuple, q, n, radius):
                cand_idx = word_to_idx.get(cand_word)
                if cand_idx is not None and cand_idx not in in_code_set:
                    next_candidates.add(cand_idx)
            candidates &= next_candidates
            if not candidates:
                break
        
        # Return a limited list
        result = list(candidates)
        if len(result) > max_candidates:
            random.shuffle(result)
            result = result[:max_candidates]
        return result

    def try_multi_swap(i1, i2, new_idx1, new_idx2):
        """Try to swap two codewords simultaneously.
        
        Remove codewords at i1 and i2, add new_idx1 and new_idx2.
        Returns True if successful.
        """
        if time.time() > deadline - 5.0:
            return False
        old_idx1 = codeword_indices[i1]
        old_idx2 = codeword_indices[i2]
        if old_idx1 == old_idx2:
            return False
        if new_idx1 in in_code or new_idx2 in in_code:
            return False
        if new_idx1 == old_idx1 and new_idx2 == old_idx2:
            return False
        
        # Check coverage after hypothetical swap
        # Words only covered by old1 or old2 need to be covered by new1 or new2
        for ni in neighbor_indices[old_idx1]:
            if coverage[ni] <= 1:
                # Must be covered by new1 or new2
                if ni not in neighbor_sets[new_idx1] and ni not in neighbor_sets[new_idx2]:
                    return False
        
        for ni in neighbor_indices[old_idx2]:
            if coverage[ni] <= 1:
                # Must be covered by new1 or new2
                if ni not in neighbor_sets[new_idx1] and ni not in neighbor_sets[new_idx2]:
                    return False
        
        # Perform swap
        for ni in neighbor_indices[old_idx1]:
            coverage[ni] -= 1
        for ni in neighbor_indices[old_idx2]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx1]:
            coverage[ni] += 1
        for ni in neighbor_indices[new_idx2]:
            coverage[ni] += 1
        
        # Update codeword_indices (handle index shift if i1 > i2)
        codeword_indices[i1] = new_idx1
        codeword_indices[i2] = new_idx2
        in_code.discard(old_idx1)
        in_code.discard(old_idx2)
        in_code.add(new_idx1)
        in_code.add(new_idx2)
        return True

    # Iterative improvement: repeatedly try removals and swaps
    improved = True
    stall_count = 0
    while improved and time.time() < deadline - 10.0:
        improved = False

        # Phase 1: Try removals
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            if try_remove(i):
                improved = True

        if time.time() > deadline - 10.0:
            break

        # Phase 2: Try targeted swaps
        for i in range(len(codeword_indices)):
            if time.time() > deadline - 10.0:
                break

            old_idx = codeword_indices[i]

            exclusive_words = []
            for ni in neighbor_indices[old_idx]:
                if coverage[ni] == 1:
                    exclusive_words.append(ni)

            if not exclusive_words:
                continue

            # Use targeted candidate search
            candidates = find_candidates_for_exclusive(exclusive_words, in_code, max_candidates=100)
            
            for cand in candidates:
                if time.time() > deadline - 10.0:
                    break
                if try_swap(i, cand):
                    improved = True
                    break

        # Phase 3: Try multi-swaps to escape local optima
        if not improved and time.time() < deadline - 10.0:
            # Try random pairs of codewords for multi-swap
            num_multi_tries = 10
            for _ in range(num_multi_tries):
                if time.time() > deadline - 10.0:
                    break
                if len(codeword_indices) < 2:
                    break
                
                i1 = random.randint(0, len(codeword_indices) - 1)
                i2 = random.randint(0, len(codeword_indices) - 1)
                if i1 == i2:
                    continue
                
                old_idx1 = codeword_indices[i1]
                old_idx2 = codeword_indices[i2]
                
                # Find exclusive words for both
                excl1 = [ni for ni in neighbor_indices[old_idx1] if coverage[ni] == 1]
                excl2 = [ni for ni in neighbor_indices[old_idx2] if coverage[ni] == 1]
                
                if not excl1 and not excl2:
                    continue
                
                # Find candidates that cover exclusive words
                all_excl = list(set(excl1 + excl2))
                candidates = find_candidates_for_exclusive(all_excl, in_code, max_candidates=50)
                
                if len(candidates) >= 2:
                    # Pick two distinct candidates
                    c1, c2 = random.sample(candidates, 2)
                    if try_multi_swap(i1, i2, c1, c2):
                        improved = True
                        break
                elif len(candidates) == 1:
                    # Try swapping one with the candidate
                    cand = candidates[0]
                    if try_swap(i1, cand):
                        improved = True
                        break
                    if try_swap(i2, cand):
                        improved = True
                        break

    # --- Step 3: Final cleanup pass ---
    if time.time() < deadline - 5.0:
        for i in range(len(codeword_indices) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            try_remove(i)

    # Convert back to lists
    result = [list(all_words[idx]) for idx in codeword_indices]

    return result
# EVOLVE-BLOCK-END
