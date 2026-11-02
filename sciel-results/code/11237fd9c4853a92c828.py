import itertools
import time
import numpy as np

def construct(q, n, radius):
    # Fallback: Tetracode direct-sum construction
    def construct_tetracode(q, n, radius):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n <= 4:
            return [h[:n] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Direct-sum construction: 9 * 3^2 = 81 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]

    # Verify coverage
    all_space = list(itertools.product(range(3), repeat=6))
    
    # For efficiency, represent coverage using numpy arrays
    word_to_idx = {w: i for i, w in enumerate(all_space)}
    
    # Precompute sphere indices for each possible word (729 words, 13 neighbors each)
    # sphere_indices[idx] = list of indices of words within distance 1
    sphere_indices = np.zeros((729, 13), dtype=np.int32)
    for idx, w in enumerate(all_space):
        w_list = list(w)
        sphere_indices[idx, 0] = idx
        for i in range(6):
            for d in [-1, 1]:
                nw = w_list[:]
                nw[i] = (nw[i] + d) % 3
                sphere_indices[idx, 1 + i * 2 + (1 if d == 1 else 0)] = word_to_idx[tuple(nw)]
    
    def word_idx(c):
        return word_to_idx[tuple(c)]
    
    def get_sphere(c):
        return sphere_indices[word_idx(c)]
    
    cover_count = np.zeros(729, dtype=np.int32)
    for c in code:
        cover_count[get_sphere(c)] += 1
    
    improved_code = list(code)
    
    # Greedy removal: try to remove codewords while maintaining coverage
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 2.0:
            break
        for i in range(len(improved_code)):
            if time.time() > deadline - 1.0:
                break
            c = improved_code[i]
            # Check if removing c leaves all words covered
            if np.any(cover_count[get_sphere(c)] == 1):
                continue
            # Remove c and update cover counts
            cover_count[get_sphere(c)] -= 1
            improved_code.pop(i)
            changed = True
            break
    
    # Two-codeword removal: try removing pairs of codewords
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 3.0:
            break
        for i in range(len(improved_code)):
            if time.time() > deadline - 2.0:
                break
            ci = improved_code[i]
            si = get_sphere(ci)
            if np.any(cover_count[si] == 1):
                continue
            for j in range(i + 1, len(improved_code)):
                if time.time() > deadline - 1.0:
                    break
                cj = improved_code[j]
                sj = get_sphere(cj)
                # Check if removing both ci and cj leaves all words covered
                # A word w is uncovered if cover_count[w] == 1 and w in si, or cover_count[w] == 2 and w in si∩sj
                # More precisely: after removing ci, words in si have count-1; after removing cj, words in sj have count-1
                # A word w is uncovered iff (w in si and cover_count[w]==1) or (w in sj and cover_count[w]==1) or (w in si∩sj and cover_count[w]==2)
                # Simpler: simulate removal and check
                sim = cover_count.copy()
                sim[si] -= 1
                sim[sj] -= 1
                if np.any(sim < 0):
                    continue
                # Remove both
                cover_count[si] -= 1
                cover_count[sj] -= 1
                improved_code.pop(j)
                improved_code.pop(i)
                changed = True
                break
            if changed:
                break
    
    # Local search: try replacing codewords to enable further removals
    # Use vectorized operations for speed
    all_indices = np.arange(729)
    
    # Build a lookup: for each word index, which sphere_indices contain it
    # This allows us to quickly find which codewords cover a given word
    # Actually, let's just do iterative improvement
    
    def try_remove_single(cc):
        """Try to remove a single codeword from cc. Returns (new_cc, removed)"""
        if time.time() > deadline - 3.0:
            return cc, False
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
            for i in range(len(cc)):
                if time.time() > deadline - 1.0:
                    break
                c = cc[i]
                if np.any(cover_count[get_sphere(c)] == 1):
                    continue
                cover_count[get_sphere(c)] -= 1
                cc.pop(i)
                changed = True
                break
        return cc, changed or True  # True if we made any change in this call
    
    def try_remove_pairs(cc):
        """Try to remove pairs of codewords from cc"""
        if time.time() > deadline - 3.0:
            return cc, False
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
            for i in range(len(cc)):
                if time.time() > deadline - 1.0:
                    break
                ci = cc[i]
                si = get_sphere(ci)
                if np.any(cover_count[si] == 1):
                    continue
                for j in range(i + 1, len(cc)):
                    if time.time() > deadline - 1.0:
                        break
                    cj = cc[j]
                    sj = get_sphere(cj)
                    sim = cover_count.copy()
                    sim[si] -= 1
                    sim[sj] -= 1
                    if np.any(sim < 0):
                        continue
                    cover_count[si] -= 1
                    cover_count[sj] -= 1
                    cc.pop(j)
                    cc.pop(i)
                    changed = True
                    break
                if changed:
                    break
        return cc, changed
    
    # Iterative improvement: try replacements and removals
    for round_num in range(50):
        if time.time() > deadline - 5.0:
            break
        
        made_improvement = False
        
        # Try replacing each codeword with a better one
        for i in range(len(improved_code)):
            if time.time() > deadline - 3.0:
                break
            c = improved_code[i]
            ci = get_sphere(c)
            
            # Words that are only covered by c (critical words)
            critical = ci[cover_count[ci] == 1]
            
            if len(critical) == 0:
                # Can remove c directly
                cover_count[ci] -= 1
                improved_code.pop(i)
                made_improvement = True
                break
            
            # Try to find a replacement that covers all critical words
            # A replacement word w can cover a critical word w' if w' is in sphere(w)
            # We need a w such that all critical words are in sphere(w)
            # This means w must be within distance 1 of all critical words
            
            # For small sets of critical words, we can enumerate candidates
            if len(critical) <= 3:
                # Find all words that cover all critical words
                candidates = all_indices.copy()
                for cw in critical:
                    # cw is a word index; we need words w such that cw is in sphere(w)
                    # i.e., w is in sphere(cw) (by symmetry)
                    candidates = np.intersect1d(candidates, sphere_indices[cw])
                
                if len(candidates) > 0:
                    # Try each candidate
                    for cand in candidates:
                        if time.time() > deadline - 2.0:
                            break
                        if cand == word_idx(c):
                            continue
                        sj = sphere_indices[cand]
                        # Check if this replacement is valid
                        # After removing c, words in ci have count-1
                        # After adding new word, words in sj have count+1
                        # Net: words in ci\sj have -1, words in sj\ci have +1, words in ci∩sj have 0
                        # We need all counts to remain >= 1
                        sim = cover_count.copy()
                        sim[ci] -= 1
                        sim[sj] += 1
                        if np.any(sim < 1):
                            continue
                        # Do the replacement
                        cover_count[ci] -= 1
                        cover_count[sj] += 1
                        improved_code[i] = list(all_space[cand])
                        made_improvement = True
                        break
                    if made_improvement:
                        break
            else:
                # Too many critical words; try a random sample of replacements
                # Pick a critical word and try words in its sphere
                cw = critical[0]
                for cand in sphere_indices[cw]:
                    if time.time() > deadline - 2.0:
                        break
                    if cand == word_idx(c):
                        continue
                    sj = sphere_indices[cand]
                    # Check coverage of all critical words
                    # A critical word cw' is covered by new word if cw' is in sphere(cand)
                    # i.e., cand is in sphere(cw')
                    all_covered = True
                    for cw2 in critical:
                        if not np.any(sphere_indices[cw2] == cand):
                            all_covered = False
                            break
                    if not all_covered:
                        continue
                    sim = cover_count.copy()
                    sim[ci] -= 1
                    sim[sj] += 1
                    if np.any(sim < 1):
                        continue
                    cover_count[ci] -= 1
                    cover_count[sj] += 1
                    improved_code[i] = list(all_space[cand])
                    made_improvement = True
                    break
                if made_improvement:
                    break
        
        if made_improvement:
            continue
        
        # Try pair removals
        if time.time() > deadline - 3.0:
            break
        improved_code, pair_removed = try_remove_pairs(improved_code)
        if pair_removed:
            continue
        
        # Try single removals again
        if time.time() > deadline - 3.0:
            break
        improved_code, single_removed = try_remove_single(improved_code)
        if single_removed:
            continue
        
        # No improvement; break
        break
    
    # Final verification
    final_cover = np.zeros(729, dtype=np.int32)
    for c in improved_code:
        final_cover[get_sphere(c)] += 1
    if np.any(final_cover == 0):
        # Coverage broken; fall back to original
        return code
    
    return improved_code
