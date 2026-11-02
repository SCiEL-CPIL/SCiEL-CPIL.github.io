"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation."""
import itertools
import random
import time

# Precompute for the small space 3^6 = 729.
_SPACE = [3] * 6  # length n placeholder

def _build_space(q, n, radius):
    # Return:
    #   words: list of tuples, length q**n, in lex order
    #   sphere: list of lists; sphere[i] = list of indices within Hamming distance radius of words[i]
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        # radius 1 neighbors: change one coordinate to a different value
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
    return words, sphere, total

def _verify(code_idx, sphere, total):
    # returns True iff every word index is covered by at least one codeword
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    for c in counts:
        if c == 0:
            return False
    return True

def _coverage_counts(code_idx, sphere, total):
    counts = [0] * total
    for ci in code_idx:
        for w in sphere[ci]:
            counts[w] += 1
    return counts

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under 120/150s limits

    words, sphere, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Seed: tetracode direct sum -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            seed_idx.append(word_to_idx[tuple(w)])
    best_code = sorted(set(seed_idx))
    best_size = len(best_code)

    def local_search(code_idx):
        # Repeatedly delete any redundant codeword until fixed point.
        counts = _coverage_counts(list(code_idx), sphere, total)
        code = sorted(set(code_idx))
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # check redundancy: after removing ci, every word in sphere[ci]
                # must still have count > 1 (i.e., covered by some other)
                redundant = True
                for w in sphere[ci]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    for w in sphere[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        # uncovered_set: set of word indices currently uncovered
        # returns list of centers that cover all of them (additions)
        chosen = []
        uncov = set(uncovered_set)
        while uncov:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break  # should not happen
            chosen.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        # add centers to cover uncovered, then local-search
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    def swap_local_search(code_idx, rng, max_swaps=10):
        # Try to swap codewords to enable further deletions
        code = list(code_idx)
        counts = _coverage_counts(code, sphere, total)
        
        improved = True
        swaps_made = 0
        while improved and swaps_made < max_swaps:
            improved = False
            if not code:
                break
            out_idx = rng.randrange(len(code))
            c_out = code[out_idx]
            
            # Find words covered ONLY by c_out
            critical_words = []
            for w in sphere[c_out]:
                if counts[w] == 1:
                    critical_words.append(w)
            
            if not critical_words:
                continue

            # Find a candidate c_in that covers the most critical_words
            best_c_in = -1
            for c_in in range(total):
                if c_in in code:
                    continue
                score = 0
                for w in critical_words:
                    if w in sphere[c_in]:
                        score += 1
                if score == len(critical_words):
                    best_c_in = c_in
                    break
            
            if best_c_in != -1:
                code[out_idx] = best_c_in
                counts = _coverage_counts(code, sphere, total)
                new_code = local_search(code)
                if len(new_code) < len(code):
                    code = new_code
                    improved = True
                    swaps_made += 1
        return code

    rng = random.Random(20240517)

    # Pass 1: local search on the 81-word seed
    cand = local_search(best_code)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand

    # Pass 2: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(60):
        if time.time() > DEADLINE - 30:
            break
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = set(range(total))
        code = []
        while uncov and len(code) < 90:
            best_c = -1
            best_cnt = -1
            for c in centers:
                if not uncov:
                    break
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
        if uncov:
            continue
        code2 = local_search(code)
        code2_swapped = swap_local_search(code2, rng)
        if len(code2_swapped) < best_size:
            best_size = len(code2_swapped)
            best_code = code2_swapped
        elif len(code2) < best_size:
            best_size = len(code2)
            best_code = code2

    # Pass 3: perturbation from best_code
    iterations = 0
    while time.time() < DEADLINE - 15:
        iterations += 1
        code = list(best_code)
        k = len(code)
        if k < 2:
            break
        
        if iterations % 2 == 0:
            # Drop and Repair
            drop_idx = rng.randrange(k)
            new_code = [c for j, c in enumerate(code) if j != drop_idx]
            counts = _coverage_counts(new_code, sphere, total)
            uncovered = set()
            for w in range(total):
                if counts[w] == 0:
                    uncovered.add(w)
            repaired = repair_and_local_search(new_code, uncovered)
            repaired_swapped = swap_local_search(repaired, rng)
            if len(repaired_swapped) < best_size:
                best_size = len(repaired_swapped)
                best_code = repaired_swapped
            elif len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired
        else:
            # Swap Perturbation
            out_idx = rng.randrange(k)
            c_out = code[out_idx]
            c_in = rng.randrange(total)
            while c_in in code:
                c_in = rng.randrange(total)
            
            new_code = list(code)
            new_code[out_idx] = c_in
            
            if _verify(new_code, sphere, total):
                reduced = local_search(new_code)
                reduced_swapped = swap_local_search(reduced, rng)
                if len(reduced_swapped) < best_size:
                    best_size = len(reduced_swapped)
                    best_code = reduced_swapped
                elif len(reduced) < best_size:
                    best_size = len(reduced)
                    best_code = reduced
            else:
                counts = _coverage_counts(new_code, sphere, total)
                uncovered = set()
                for w in range(total):
                    if counts[w] == 0:
                        uncovered.add(w)
                if uncovered:
                    repaired = repair_and_local_search(new_code, uncovered)
                    repaired_swapped = swap_local_search(repaired, rng)
                    if len(repaired_swapped) < best_size:
                        best_size = len(repaired_swapped)
                        best_code = repaired_swapped
                    elif len(repaired) < best_size:
                        best_size = len(repaired)
                        best_code = repaired

    if not _verify(best_code, sphere, total):
        best_code = sorted(set(seed_idx))

    return [list(words[i]) for i in sorted(set(best_code))]
