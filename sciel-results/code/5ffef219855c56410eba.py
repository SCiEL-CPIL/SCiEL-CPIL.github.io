"""Portfolio SEARCH approach for K_3(6,1): greedy + local search + perturbation + flow-based refinement."""
import itertools
import random
import time
import sys
import numpy as np

def _build_space(q, n, radius):
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute sphere as numpy arrays for fast operations
    sphere_np = []
    for w in words:
        s = [word_to_idx[w]]
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    s.append(word_to_idx[tuple(w2)])
        sphere_np.append(np.array(s, dtype=np.int32))
    
    # Also build an array of all sphere indices for faster coverage counting
    # coverage_matrix: total x total, but that's 729x729 = 531k entries, manageable
    # Actually, let's use a different approach: precompute for each codeword index,
    # which words it covers as a boolean mask
    cover_masks = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for w in sphere_np[i]:
            cover_masks[i, w] = True
    
    return words, sphere_np, cover_masks, total

def _verify(code_idx, cover_masks, total):
    if not code_idx:
        return total == 0
    covered = np.any(cover_masks[list(code_idx)], axis=0)
    return bool(np.all(covered))

def _coverage_counts(code_idx, cover_masks, total):
    if not code_idx:
        return np.zeros(total, dtype=np.int32)
    counts = np.zeros(total, dtype=np.int32)
    for ci in code_idx:
        counts += cover_masks[ci]
    return counts

def _local_search_fast(code_idx, cover_masks, total):
    """Fast local search: repeatedly delete redundant codewords."""
    code = sorted(set(code_idx))
    if not code:
        return code
    
    counts = np.zeros(total, dtype=np.int32)
    for ci in code:
        counts += cover_masks[ci]
    
    changed = True
    while changed:
        changed = False
        new_code = []
        for ci in code:
            # Check if ci is redundant: all words it covers are covered by others
            if np.all(counts[cover_masks[ci]] > 1):
                counts -= cover_masks[ci]
                changed = True
            else:
                new_code.append(ci)
        code = new_code
    return code

def _greedy_cover(uncov_mask, cover_masks, total, rng, max_centers=100):
    """Greedy set cover on a boolean mask of uncovered words."""
    chosen = []
    uncov = uncov_mask.copy()
    for _ in range(max_centers):
        if not np.any(uncov):
            break
        # Find center covering most uncovered
        gains = cover_masks @ uncov
        best_c = int(np.argmax(gains))
        if gains[best_c] == 0:
            break
        chosen.append(best_c)
        uncov &= ~cover_masks[best_c]
    return chosen

def _swap_search(code_idx, cover_masks, total, rng, max_rounds=30):
    """Try swapping codewords to enable further deletions."""
    code = list(code_idx)
    if len(code) < 2:
        return code
    
    counts = np.zeros(total, dtype=np.int32)
    for ci in code:
        counts += cover_masks[ci]
    
    for _ in range(max_rounds):
        if time.time() > 140:
            break
        if not code:
            break
        
        # Pick a random codeword to try replacing
        out_idx = rng.randrange(len(code))
        c_out = code[out_idx]
        
        # Find words covered ONLY by c_out
        critical = cover_masks[c_out] & (counts == 1)
        if not np.any(critical):
            # Can remove directly
            code.pop(out_idx)
            counts -= cover_masks[c_out]
            continue
        
        # Find candidate c_in that covers all critical words
        best_c_in = -1
        best_score = -1
        # Sample candidates for speed
        candidates = list(range(total))
        rng.shuffle(candidates)
        for c_in in candidates[:200]:
            if c_in in code:
                continue
            score = int(np.sum(cover_masks[c_in] & critical))
            if score > best_score:
                best_score = score
                best_c_in = c_in
                if score == int(np.sum(critical)):
                    break
        
        if best_c_in != -1 and best_score == int(np.sum(critical)):
            code[out_idx] = best_c_in
            counts -= cover_masks[c_out]
            counts += cover_masks[best_c_in]
            # Try to delete some now-redundant codewords
            code = _local_search_fast(code, cover_masks, total)
            counts = np.zeros(total, dtype=np.int32)
            for ci in code:
                counts += cover_masks[ci]
    
    return _local_search_fast(code, cover_masks, total)

def _perturb_and_search(best_code, cover_masks, total, rng, deadline):
    """Perturbation + local search cycle."""
    code = list(best_code)
    k = len(code)
    if k < 2:
        return best_code
    
    strategy = rng.randint(0, 2)
    
    if strategy == 0:
        # Drop and repair
        drop_idx = rng.randrange(k)
        new_code = [c for j, c in enumerate(code) if j != drop_idx]
        counts = np.zeros(total, dtype=np.int32)
        for ci in new_code:
            counts += cover_masks[ci]
        uncovered = (counts == 0).astype(bool)
        if np.any(uncovered):
            added = _greedy_cover(uncovered, cover_masks, total, rng)
            new_code.extend(added)
        new_code = _local_search_fast(new_code, cover_masks, total)
        new_code = _swap_search(new_code, cover_masks, total, rng)
        return new_code
    
    elif strategy == 1:
        # Swap perturbation
        out_idx = rng.randrange(k)
        c_in = rng.randrange(total)
        while c_in in code:
            c_in = rng.randrange(total)
        new_code = list(code)
        new_code[out_idx] = c_in
        # Verify and repair
        counts = np.zeros(total, dtype=np.int32)
        for ci in new_code:
            counts += cover_masks[ci]
        uncovered = (counts == 0).astype(bool)
        if np.any(uncovered):
            added = _greedy_cover(uncovered, cover_masks, total, rng)
            new_code.extend(added)
        new_code = _local_search_fast(new_code, cover_masks, total)
        new_code = _swap_search(new_code, cover_masks, total, rng)
        return new_code
    
    else:
        # Multi-swap
        new_code = list(code)
        for _ in range(rng.randint(1, 3)):
            if len(new_code) < 2:
                break
            out_idx = rng.randrange(len(new_code))
            c_in = rng.randrange(total)
            while c_in in new_code:
                c_in = rng.randrange(total)
            new_code[out_idx] = c_in
        counts = np.zeros(total, dtype=np.int32)
        for ci in new_code:
            counts += cover_masks[ci]
        uncovered = (counts == 0).astype(bool)
        if np.any(uncovered):
            added = _greedy_cover(uncovered, cover_masks, total, rng)
            new_code.extend(added)
        new_code = _local_search_fast(new_code, cover_masks, total)
        new_code = _swap_search(new_code, cover_masks, total, rng)
        return new_code

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 145.0
    
    words, sphere_np, cover_masks, total = _build_space(q, n, radius)
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
    
    rng = random.Random(20240517)
    
    # Pass 1: local search on the 81-word seed
    cand = _local_search_fast(best_code, cover_masks, total)
    if len(cand) < best_size:
        best_size = len(cand)
        best_code = cand
    
    # Pass 2: swap search
    if time.time() < DEADLINE - 30:
        cand = _swap_search(best_code, cover_masks, total, rng)
        if len(cand) < best_size:
            best_size = len(cand)
            best_code = cand
    
    # Pass 3: fresh greedy starts
    for trial in range(20):
        if time.time() > DEADLINE - 40:
            break
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = np.ones(total, dtype=bool)
        code = []
        while np.any(uncov) and len(code) < 90:
            gains = cover_masks @ uncov
            best_c = int(np.argmax(gains))
            if gains[best_c] == 0:
                break
            code.append(best_c)
            uncov &= ~cover_masks[best_c]
            centers = [c for c in centers if c != best_c]
            if centers:
                rng.shuffle(centers)
        if not np.any(uncov):
            code2 = _local_search_fast(code, cover_masks, total)
            code2 = _swap_search(code2, cover_masks, total, rng)
            if len(code2) < best_size:
                best_size = len(code2)
                best_code = code2
    
    # Pass 4: iterative perturbation
    iterations = 0
    while time.time() < DEADLINE - 10:
        iterations += 1
        cand = _perturb_and_search(best_code, cover_masks, total, rng, DEADLINE)
        if len(cand) < best_size:
            best_size = len(cand)
            best_code = cand
    
    # Final verification
    if not _verify(best_code, cover_masks, total):
        best_code = sorted(set(seed_idx))
    
    return [list(words[i]) for i in sorted(set(best_code))]
