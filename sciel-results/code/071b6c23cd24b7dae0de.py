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
    deadline = start_time + 135.0  # Safety margin within 150s limit

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
    # Use multiple tetracode variants to break symmetry
    tetracodes = []
    # Original tetracode
    tetracode0 = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tetracodes.append(tetracode0)
    
    # Permutations of coordinates
    if n >= 4:
        from itertools import permutations
        perms = list(permutations(range(4)))
        random.shuffle(perms)
        for perm in perms[:6]:  # Use up to 6 different permutations
            tc = []
            for a in range(q):
                for b in range(q):
                    base = [a, b, (a + b) % q, (a + 2 * b) % q]
                    permuted = [base[perm[i]] for i in range(4)]
                    tc.append(permuted)
            tetracodes.append(tc)

    initial_codewords = []
    if n >= 4:
        tail_len = n - 4
        tails = list(itertools.product(range(q), repeat=tail_len))
        
        # Assign different tetracodes to different tails to break symmetry
        for tail_idx, tail in enumerate(tails):
            tc = tetracodes[tail_idx % len(tetracodes)]
            for h in tc:
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
        """Find candidate codewords that cover all exclusive words."""
        if not exclusive_words:
            return []
        
        first_excl = exclusive_words[0]
        first_excl_tuple = all_words[first_excl]
        candidates = set()
        
        for cand_word in _neighbors_of(first_excl_tuple, q, n, radius):
            cand_idx = word_to_idx.get(cand_word)
            if cand_idx is not None and cand_idx not in in_code_set:
                candidates.add(cand_idx)
        
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
        
        result = list(candidates)
        if len(result) > max_candidates:
            random.shuffle(result)
            result = result[:max_candidates]
        return result

    def try_multi_swap(i1, i2, new_idx1, new_idx2):
        """Try to swap two codewords simultaneously."""
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
        
        for ni in neighbor_indices[old_idx1]:
            if coverage[ni] <= 1:
                if ni not in neighbor_sets[new_idx1] and ni not in neighbor_sets[new_idx2]:
                    return False
        
        for ni in neighbor_indices[old_idx2]:
            if coverage[ni] <= 1:
                if ni not in neighbor_sets[new_idx1] and ni not in neighbor_sets[new_idx2]:
                    return False
        
        for ni in neighbor_indices[old_idx1]:
            coverage[ni] -= 1
        for ni in neighbor_indices[old_idx2]:
            coverage[ni] -= 1
        for ni in neighbor_indices[new_idx1]:
            coverage[ni] += 1
        for ni in neighbor_indices[new_idx2]:
            coverage[ni] += 1
        
        codeword_indices[i1] = new_idx1
        codeword_indices[i2] = new_idx2
        in_code.discard(old_idx1)
        in_code.discard(old_idx2)
        in_code.add(new_idx1)
        in_code.add(new_idx2)
        return True

    def try_triple_swap(i1, i2, i3, new_idx1, new_idx2, new_idx3):
        """Try to swap three codewords simultaneously."""
        if time.time() > deadline - 5.0:
            return False
        old_idx1 = codeword_indices[i1]
        old_idx2 = codeword_indices[i2]
        old_idx3 = codeword_indices[i3]
        
        if old_idx1 == old_idx2 or old_idx1 == old_idx3 or old_idx2 == old_idx3:
            return False
        if new_idx1 in in_code or new_idx2 in in_code or new_idx3 in in_code:
            return False
        if new_idx1 == old_idx1 and new_idx2 == old_idx2 and new_idx3 == old_idx3:
            return False
        
        # Check coverage
        new_sets = [neighbor_sets[new_idx1], neighbor_sets[new_idx2], neighbor_sets[new_idx3]]
        
        for old_idx in [old_idx1, old_idx2, old_idx3]:
            for ni in neighbor_indices[old_idx]:
                if coverage[ni] <= 1:
                    if ni not in new_sets[0] and ni not in new_sets[1] and ni not in new_sets[2]:
                        return False
        
        # Apply swap
        for old_idx in [old_idx1, old_idx2, old_idx3]:
            for ni in neighbor_indices[old_idx]:
                coverage[ni] -= 1
        for new_idx in [new_idx1, new_idx2, new_idx3]:
            for ni in neighbor_indices[new_idx]:
                coverage[ni] += 1
        
        codeword_indices[i1] = new_idx1
        codeword_indices[i2] = new_idx2
        codeword_indices[i3] = new_idx3
        in_code.discard(old_idx1)
        in_code.discard(old_idx2)
        in_code.discard(old_idx3)
        in_code.add(new_idx1)
        in_code.add(new_idx2)
        in_code.add(new_idx3)
        return True

    # Iterative improvement: repeatedly try removals and swaps
    improved = True
    stall_count = 0
    max_stall = 3
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
            num_multi_tries = 20
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
                
                excl1 = [ni for ni in neighbor_indices[old_idx1] if coverage[ni] == 1]
                excl2 = [ni for ni in neighbor_indices[old_idx2] if coverage[ni] == 1]
                
                if not excl1 and not excl2:
                    continue
                
                all_excl = list(set(excl1 + excl2))
                candidates = find_candidates_for_exclusive(all_excl, in_code, max_candidates=50)
                
                if len(candidates) >= 2:
                    c1, c2 = random.sample(candidates, 2)
                    if try_multi_swap(i1, i2, c1, c2):
                        improved = True
                        break
                elif len(candidates) == 1:
                    cand = candidates[0]
                    if try_swap(i1, cand):
                        improved = True
                        break
                    if try_swap(i2, cand):
                        improved = True
                        break

        # Phase 4: Try triple swaps for deeper local search
        if not improved and time.time() < deadline - 10.0 and len(codeword_indices) >= 3:
            num_triple_tries = 10
            for _ in range(num_triple_tries):
                if time.time() > deadline - 10.0:
                    break
                
                indices = random.sample(range(len(codeword_indices)), 3)
                i1, i2, i3 = indices
                
                old_idx1 = codeword_indices[i1]
                old_idx2 = codeword_indices[i2]
                old_idx3 = codeword_indices[i3]
                
                excl1 = [ni for ni in neighbor_indices[old_idx1] if coverage[ni] == 1]
                excl2 = [ni for ni in neighbor_indices[old_idx2] if coverage[ni] == 1]
                excl3 = [ni for ni in neighbor_indices[old_idx3] if coverage[ni] == 1]
                
                all_excl = list(set(excl1 + excl2 + excl3))
                if not all_excl:
                    continue
                
                candidates = find_candidates_for_exclusive(all_excl, in_code, max_candidates=30)
                
                if len(candidates) >= 3:
                    c1, c2, c3 = random.sample(candidates, 3)
                    if try_triple_swap(i1, i2, i3, c1, c2, c3):
                        improved = True
                        break
                elif len(candidates) >= 2:
                    c1, c2 = random.sample(candidates, 2)
                    if try_multi_swap(i1, i2, c1, c2):
                        improved = True
                        break
                elif len(candidates) == 1:
                    cand = candidates[0]
                    for i in [i1, i2, i3]:
                        if try_swap(i, cand):
                            improved = True
                            break

        # Track stalls
        if improved:
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= max_stall:
                # Perturb: randomly swap a few codewords to escape local optimum
                if time.time() < deadline - 15.0 and len(codeword_indices) >= 2:
                    for _ in range(3):
                        if time.time() > deadline - 15.0:
                            break
                        i = random.randint(0, len(codeword_indices) - 1)
                        old_idx = codeword_indices[i]
                        # Try to swap with a random candidate
                        for _ in range(20):
                            cand = random.randint(0, total_words - 1)
                            if cand in in_code:
                                continue
                            excl = [ni for ni in neighbor_indices[old_idx] if coverage[ni] == 1]
                            if not excl:
                                continue
                            if all(w in neighbor_sets[cand] for w in excl):
                                if try_swap(i, cand):
                                    break
                stall_count = 0

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
