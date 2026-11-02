"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation."""
import itertools
import random
import time
import numpy as np

# Precompute for the small space 3^6 = 729.
_SPACE = [3] * 6  # length n placeholder

def _build_space(q, n, radius):
    # Return:
    #   words: list of tuples, length q**n, in lex order
    #   M: numpy array (total, total) where M[i,j]=1 if j is in sphere of i
    #   word_to_idx: dict
    #   total: int
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    M = np.zeros((total, total), dtype=np.uint8)
    
    for i, w in enumerate(words):
        M[i, i] = 1
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    j = word_to_idx[tuple(w2)]
                    M[i, j] = 1
                    # M[j, i] = 1 is not necessarily true for directed, but here Hamming is symmetric.
                    # However, we only need M[i] for coverage of i.
    
    return words, M, word_to_idx, total

def _verify_np(code_idx, M, total):
    if not code_idx:
        return total == 0
    union = np.zeros(total, dtype=np.uint8)
    for ci in code_idx:
        union |= M[ci]
    return np.all(union == 1)

def _coverage_counts_np(code_idx, M, total):
    if not code_idx:
        return np.zeros(total, dtype=np.int16)
    counts = np.zeros(total, dtype=np.int16)
    for ci in code_idx:
        counts += M[ci]
    return counts

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 110.0  # safety margin under 120/150s limits

    words, M, word_to_idx, total = _build_space(q, n, radius)

    # Precompute neighbor lists for faster greedy? 
    # M is already sparse. 
    # For greedy, we need to count intersections with a set.
    # Let's stick to numpy arrays for counts.

    # Seed: tetracode direct sum -> 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_idx = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            w = h + list(tail)
            seed_idx.append(word_to_idx[tuple(w)])
    
    best_code = sorted(set(seed_idx))
    best_size = len(best_code)

    def local_search(code_idx):
        # Repeatedly delete any redundant codeword until fixed point.
        code = sorted(set(code_idx))
        if not code:
            return code
        
        counts = _coverage_counts_np(code, M, total)
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # Redundant if no word is uniquely covered by ci
                # unique = (counts == 1) & M[ci]
                # if not any(unique): redundant
                if not np.any((counts == 1) & M[ci]):
                    counts -= M[ci]
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        # uncovered_set: set of word indices currently uncovered
        chosen = []
        uncov_mask = np.zeros(total, dtype=np.uint8)
        for w in uncovered_set:
            uncov_mask[w] = 1
        
        uncov_count = len(uncovered_set)
        
        while uncov_count > 0:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                # Count how many uncovered words are covered by c
                # cnt = np.sum(uncov_mask & M[c])
                cnt = np.dot(uncov_mask, M[c])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            
            if best_cnt <= 0:
                break  # should not happen
            
            chosen.append(best_c)
            # Update uncov_mask
            uncov_mask &= ~M[best_c]
            uncov_count = np.sum(uncov_mask)
            
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    def swap_local_search(code_idx, rng, max_swaps=50):
        code = list(code_idx)
        counts = _coverage_counts_np(code, M, total)
        
        improved = True
        swaps_made = 0
        while improved and swaps_made < max_swaps:
            improved = False
            if not code:
                break
            out_idx = rng.randrange(len(code))
            c_out = code[out_idx]
            
            # Find words covered ONLY by c_out
            unique_mask = (counts == 1) & M[c_out]
            unique_words = np.where(unique_mask)[0]
            
            if len(unique_words) == 0:
                # c_out is redundant (should have been removed, but maybe state changed)
                code.pop(out_idx)
                counts = _coverage_counts_np(code, M, total)
                new_code = local_search(code)
                if len(new_code) < len(code):
                    code = new_code
                    improved = True
                    swaps_made += 1
                continue

            best_c_in = -1
            best_score = -1
            for c_in in range(total):
                if c_in in code:
                    continue
                # Score: how many unique_words does c_in cover?
                score = np.dot(M[c_in], unique_mask)
                if score > best_score:
                    best_score = score
                    best_c_in = c_in
            
            if best_c_in != -1 and best_score == len(unique_words):
                # Perfect swap
                code[out_idx] = best_c_in
                counts = _coverage_counts_np(code, M, total)
                new_code = local_search(code)
                if len(new_code) < len(code):
                    code = new_code
                    improved = True
                    swaps_made += 1
                    counts = _coverage_counts_np(code, M, total)
                elif len(new_code) == len(code):
                    # Accept neutral swap to escape local minima (50% chance)
                    if rng.random() < 0.5:
                        code = new_code
                        counts = _coverage_counts_np(code, M, total)
                        improved = True
                        swaps_made += 1
            else:
                # Try a random swap to perturb
                c_in = rng.randrange(total)
                while c_in in code:
                    c_in = rng.randrange(total)
                
                code[out_idx] = c_in
                if _verify_np(code, M, total):
                    new_code = local_search(code)
                    if len(new_code) < len(code):
                        code = new_code
                        improved = True
                        swaps_made += 1
                        counts = _coverage_counts_np(code, M, total)
                    elif len(new_code) == len(code) and rng.random() < 0.2:
                        code = new_code
                        counts = _coverage_counts_np(code, M, total)
                        improved = True
                        swaps_made += 1
                    else:
                        code[out_idx] = c_out
                        counts = _coverage_counts_np(code, M, total)

        return code

    rng = random.Random(20240517)

    # Pass 1: local search on the 81-word seed
    cand = local_search(best_code)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand

    # Pass 2: full greedy from scratch (random tie-breaks) + local search, multiple times
    for trial in range(150): # Increased trials
        if time.time() > DEADLINE - 30:
            break
        
        # Random start for greedy
        centers = list(range(total))
        rng.shuffle(centers)
        
        uncov_mask = np.ones(total, dtype=np.uint8)
        uncov_count = total
        code = []
        
        while uncov_count > 0 and len(code) < 90:
            best_c = -1
            best_cnt = -1
            # Iterate through shuffled centers for random tie-breaking
            for c in centers:
                cnt = np.dot(uncov_mask, M[c])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            
            if best_c == -1 or best_cnt <= 0:
                break
            
            code.append(best_c)
            uncov_mask &= ~M[best_c]
            uncov_count = np.sum(uncov_mask)
            # Remove best_c from centers to avoid re-choosing? 
            # Greedy usually allows re-choosing, but it's redundant.
            # We can just filter centers or keep a set.
            # For speed, let's just keep the list and skip if already chosen?
            # Actually, standard greedy picks best from remaining.
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
            
        if uncov_count > 0:
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
            counts = _coverage_counts_np(new_code, M, total)
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) > 0:
                repaired = repair_and_local_search(new_code, set(uncovered))
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
            
            if _verify_np(new_code, M, total):
                reduced = local_search(new_code)
                reduced_swapped = swap_local_search(reduced, rng)
                if len(reduced_swapped) < best_size:
                    best_size = len(reduced_swapped)
                    best_code = reduced_swapped
                elif len(reduced) < best_size:
                    best_size = len(reduced)
                    best_code = reduced
            else:
                counts = _coverage_counts_np(new_code, M, total)
                uncovered = np.where(counts == 0)[0]
                if len(uncovered) > 0:
                    repaired = repair_and_local_search(new_code, set(uncovered))
                    repaired_swapped = swap_local_search(repaired, rng)
                    if len(repaired_swapped) < best_size:
                        best_size = len(repaired_swapped)
                        best_code = repaired_swapped
                    elif len(repaired) < best_size:
                        best_size = len(repaired)
                        best_code = repaired

    if not _verify_np(best_code, M, total):
        best_code = sorted(set(seed_idx))

    return [list(words[i]) for i in sorted(set(best_code))]
