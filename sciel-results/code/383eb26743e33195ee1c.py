import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-(3,6,1) cases
    def construct_tetracode(q, n, radius):
        if q != 3:
            # Simple greedy for general case
            all_words = list(itertools.product(range(q), repeat=n))
            covered = [False] * (q ** n)
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            code = []
            for w in all_words:
                if not covered[word_to_idx[w]]:
                    code.append(list(w))
                    for i in range(n):
                        for d in range(1, q):
                            nw = list(w)
                            nw[i] = (nw[i] + d) % q
                            if radius >= 1:
                                covered[word_to_idx[tuple(nw)]] = True
                    covered[word_to_idx[w]] = True
            return code
        
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n <= 4:
            return [h[:n] for h in tetracode]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    if q != 3 or n != 6 or radius != 1:
        return construct_tetracode(q, n, radius)

    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all words and their indices
    all_space = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_space)}
    N = 729

    # Precompute sphere index arrays (numpy) for each word
    # sphere_indices[i] = array of indices of words within distance 1 of word i
    sphere_indices = np.zeros((N, 13), dtype=np.int32)
    for i, w in enumerate(all_space):
        w_list = list(w)
        sphere_indices[i, 0] = i
        idx = 1
        for pos in range(6):
            for d in [-1, 1]:
                nc = w_list[:]
                nc[pos] = (nc[pos] + d) % 3
                sphere_indices[i, idx] = word_to_idx[tuple(nc)]
                idx += 1

    # Initial code: direct-sum construction (81 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    code_indices = [word_to_idx[tuple(c)] for c in code]
    
    # Build coverage count array
    cover_count = np.zeros(N, dtype=np.int32)
    for ci in code_indices:
        cover_count[sphere_indices[ci]] += 1
    
    if not np.all(cover_count > 0):
        return code

    # Phase 1: Greedy removal - repeatedly remove any codeword that's redundant
    changed = True
    while changed:
        changed = False
        if time.time() > deadline - 15.0:
            break
        for i in range(len(code_indices)):
            if time.time() > deadline - 10.0:
                break
            ci = code_indices[i]
            # Check if all words in sphere have coverage > 1
            if np.all(cover_count[sphere_indices[ci]] > 1):
                cover_count[sphere_indices[ci]] -= 1
                code_indices.pop(i)
                changed = True
                break

    # Phase 2: Local search with exchange moves
    # Strategy: try to remove a codeword; if that creates uncovered words,
    # find a replacement codeword that covers those uncovered words.
    
    best_code_indices = list(code_indices)
    best_cover = cover_count.copy()
    best_size = len(code_indices)
    
    # Multiple restarts
    rng = np.random.RandomState(42)
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Start from best so far (or perturb it)
        if restart == 0:
            current_indices = list(code_indices)
            current_cover = cover_count.copy()
        else:
            # Perturb: remove 1-3 random codewords and repair
            current_indices = list(best_code_indices)
            current_cover = best_cover.copy()
            
            # Remove some random codewords
            num_remove = min(1 + rng.randint(0, 2), len(current_indices) - 5)
            for _ in range(num_remove):
                if len(current_indices) <= 5:
                    break
                ri = rng.randint(0, len(current_indices) - 1)
                ci = current_indices[ri]
                current_cover[sphere_indices[ci]] -= 1
                current_indices.pop(ri)
            
            # Repair: add codewords to cover uncovered words
            while True:
                uncovered = np.where(current_cover == 0)[0]
                if len(uncovered) == 0:
                    break
                # Find a word that covers the most uncovered words
                best_add = -1
                best_count = -1
                for u in uncovered[:50]:  # limit candidates for speed
                    # Find words within distance 1 of u
                    # Actually, find a codeword candidate that covers u and other uncovered words
                    # Try all words in the space
                    pass
                # Simpler: pick a random uncovered word and add a random word near it
                u = uncovered[rng.randint(0, len(uncovered) - 1)]
                # Add the word u itself or a neighbor
                add_candidates = sphere_indices[u]
                add_idx = add_candidates[rng.randint(0, len(add_candidates) - 1)]
                if add_idx not in current_indices:
                    current_cover[sphere_indices[add_idx]] += 1
                    current_indices.append(add_idx)
                else:
                    # Already have it, try another
                    for ac in add_candidates:
                        if ac not in current_indices:
                            current_cover[sphere_indices[ac]] += 1
                            current_indices.append(ac)
                            break
                    else:
                        # All neighbors already in code, pick random
                        add_idx = rng.randint(0, N)
                        if add_idx not in current_indices:
                            current_cover[sphere_indices[add_idx]] += 1
                            current_indices.append(add_idx)
                        else:
                            break
        
        current_size = len(current_indices)
        
        # Local search iterations
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if current_size <= best_size:
                # Try to remove
                # Find a codeword whose removal doesn't break coverage
                removable = []
                for ci in current_indices:
                    if np.all(current_cover[sphere_indices[ci]] > 1):
                        removable.append(ci)
                
                if removable:
                    ci = removable[rng.randint(0, len(removable) - 1)]
                    pos = current_indices.index(ci)
                    current_cover[sphere_indices[ci]] -= 1
                    current_indices.pop(pos)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code_indices = list(current_indices)
                        best_cover = current_cover.copy()
                    continue
                
                # Try exchange: remove one, add another
                # Pick a random codeword to remove
                ri = rng.randint(0, len(current_indices) - 1)
                old_ci = current_indices[ri]
                
                # What words would become uncovered?
                temp_cover = current_cover - 1  # after removing old_ci
                temp_cover[sphere_indices[old_ci]] -= 1  # wait, this is wrong
                # Actually: current_cover represents coverage with all current codewords
                # After removing old_ci: new_cover = current_cover - indicator(sphere of old_ci)
                new_cover = current_cover.copy()
                new_cover[sphere_indices[old_ci]] -= 1
                uncovered_after = np.where(new_cover == 0)[0]
                
                if len(uncovered_after) == 0:
                    # Can just remove
                    current_cover = new_cover
                    current_indices.pop(ri)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code_indices = list(current_indices)
                        best_cover = current_cover.copy()
                    continue
                
                # Need to find a replacement that covers all uncovered_after
                # A replacement word w covers word u if dist(w,u) <= 1
                # For each uncovered word, the replacement must be in its sphere
                # Find intersection of spheres of all uncovered words
                # Since there are at most ~12 uncovered words, this is feasible
                
                if len(uncovered_after) > 6:
                    # Too many uncovered, skip this exchange
                    continue
                
                # Find candidate replacements
                # A word w covers u if w is in sphere_indices[u]
                # We need w in intersection of all spheres
                # Start with sphere of first uncovered word
                candidates = set(sphere_indices[uncovered_after[0]].tolist())
                for u in uncovered_after[1:]:
                    candidates &= set(sphere_indices[u].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    # No single word can cover all uncovered words
                    # Try a 2-exchange: remove 2, add 1 or 2
                    # For now, skip
                    continue
                
                # Pick a random candidate that's not already in code
                candidates = [c for c in candidates if c not in current_indices]
                if not candidates:
                    # All candidates already in code, just remove old
                    current_cover = new_cover
                    current_indices.pop(ri)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code_indices = list(current_indices)
                        best_cover = current_cover.copy()
                    continue
                
                new_ci = candidates[rng.randint(0, len(candidates) - 1)]
                
                # Apply exchange
                current_cover = new_cover
                current_cover[sphere_indices[new_ci]] += 1
                current_indices[ri] = new_ci
                
                # After exchange, try to remove more
                for _ in range(3):  # try a few removals
                    removable2 = []
                    for ci in current_indices:
                        if np.all(current_cover[sphere_indices[ci]] > 1):
                            removable2.append(ci)
                    if not removable2:
                        break
                    ci = removable2[rng.randint(0, len(removable2) - 1)]
                    pos = current_indices.index(ci)
                    current_cover[sphere_indices[ci]] -= 1
                    current_indices.pop(pos)
                    current_size -= 1
                
                if current_size < best_size:
                    best_size = current_size
                    best_code_indices = list(current_indices)
                    best_cover = current_cover.copy()
        
        # End of restart

    # Phase 3: Final greedy removal on best
    if time.time() < deadline - 2.0:
        final_cover = best_cover.copy()
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 1.0:
                break
            for i in range(len(best_code_indices)):
                if time.time() > deadline - 0.5:
                    break
                ci = best_code_indices[i]
                if np.all(final_cover[sphere_indices[ci]] > 1):
                    final_cover[sphere_indices[ci]] -= 1
                    best_code_indices.pop(i)
                    changed = True
                    break

    # Convert back to list of lists
    result = [list(all_space[ci]) for ci in best_code_indices]
    return result
