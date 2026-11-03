import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # 130s of 150s limit
    
    N = q ** n
    # N = 6561 for q=3, n=8
    
    # Helper: convert index to word and back
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    # Precompute all neighbors for each word (radius 1)
    # neighbors[i] = list of indices within Hamming distance <= radius of word i
    neighbors = []
    for i in range(N):
        word = idx_to_word(i)
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        word[pos] = new_val
                        nbrs.append(word_to_idx(word))
            word[pos] = orig  # restore, but we don't need it
        neighbors.append(nbrs)
    
    # Build tetracode product as initial code
    # Tetracode: 9 words of length 4 over {0,1,2}
    # C4 = {(a, b, (a+b)%3, (a+2b)%3) : a,b in {0,1,2}}
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    # Product code: C4 x C4 gives 81 codewords of length 8
    init_code = []
    for w1 in tetracode:
        for w2 in tetracode:
            init_code.append(w1 + w2)
    
    # Convert to indices
    code = [word_to_idx(w) for w in init_code]
    code_set = set(code)
    
    # Verify initial code covers everything (it should)
    covered = np.zeros(N, dtype=bool)
    for c in code:
        for w in neighbors[c]:
            covered[w] = True
    
    if not covered.all():
        # Fallback to greedy if product code doesn't cover (shouldn't happen)
        uncovered = set(np.where(~covered)[0])
        while uncovered:
            if time.time() > deadline:
                break
            best_c = None
            best_count = -1
            for c in range(N):
                if c in code_set:
                    continue
                cnt = sum(1 for w in neighbors[c] if w in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c is None:
                break
            code.append(best_c)
            code_set.add(best_c)
            for w in neighbors[best_c]:
                uncovered.discard(w)
        # Rebuild covered
        covered = np.zeros(N, dtype=bool)
        for c in code:
            for w in neighbors[c]:
                covered[w] = True
    
    # Now do reduction: remove redundant codewords
    # Maintain coverage count array
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for w in neighbors[c]:
            cover_count[w] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            # Check if c is redundant: all neighbors have cover_count > 1
            is_redundant = True
            for w in neighbors[c]:
                if cover_count[w] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                for w in neighbors[c]:
                    cover_count[w] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break  # restart scan since indices shifted
    
    # Try swaps: replace a codeword with another that maintains coverage
    # and potentially enables further reductions
    random.seed(42)
    for _ in range(200):  # limited swap attempts
        if time.time() > deadline:
            break
        if not code:
            break
        i = random.randint(0, len(code) - 1)
        old_c = code[i]
        # Try a random new codeword
        new_c = random.randint(0, N - 1)
        if new_c in code_set:
            continue
        # Check if replacing old_c with new_c maintains full coverage
        # A word w is uncovered after swap if:
        #   - w is in neighbors[old_c] but not in neighbors[new_c], AND
        #   - cover_count[w] == 1 (only old_c covers it)
        # So check: for all w in neighbors[old_c] where cover_count[w] == 1,
        #           w must be in neighbors[new_c]
        valid_swap = True
        old_nbrs = set(neighbors[old_c])
        new_nbrs = set(neighbors[new_c])
        for w in old_nbrs:
            if cover_count[w] == 1 and w not in new_nbrs:
                valid_swap = False
                break
        if valid_swap:
            # Perform swap
            for w in neighbors[old_c]:
                cover_count[w] -= 1
            for w in neighbors[new_c]:
                cover_count[w] += 1
            code[i] = new_c
            code_set.discard(old_c)
            code_set.add(new_c)
            # Try to remove redundancies after swap
            for j in range(len(code)):
                if time.time() > deadline:
                    break
                cj = code[j]
                is_redundant = True
                for w in neighbors[cj]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for w in neighbors[cj]:
                        cover_count[w] -= 1
                    code.pop(j)
                    code_set.discard(cj)
                    break
    
    # Final verification and fallback
    covered = np.zeros(N, dtype=bool)
    for c in code:
        for w in neighbors[c]:
            covered[w] = True
    
    if not covered.all():
        # Add missing coverage
        uncovered = set(np.where(~covered)[0])
        while uncovered:
            if time.time() > deadline:
                break
            best_c = None
            best_count = -1
            for c in range(N):
                if c in code_set:
                    continue
                cnt = sum(1 for w in neighbors[c] if w in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c is None:
                break
            code.append(best_c)
            code_set.add(best_c)
            for w in neighbors[best_c]:
                uncovered.discard(w)
    
    # Convert to words
    result = [idx_to_word(c) for c in code]
    return result
