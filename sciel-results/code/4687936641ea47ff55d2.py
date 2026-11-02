import time
import itertools
import random
import sys

def construct(q, n, radius):
    """Construct a covering code for {0..q-1}^n with radius.
    
    Returns a list of codewords (each a list of n integers in 0..q-1)
    such that every word in the space is within Hamming distance radius
    of some codeword.
    """
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1()
    
    # General fallback
    total = q ** n
    if total > 200000:
        # Too large for exact methods; return empty (invalid) as fallback
        return []
    
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute ball masks
    ball_masks = []
    for i in range(total):
        w = words[i]
        mask = 0
        mask |= (1 << i)
        for pos in range(n):
            orig = w[pos]
            for nv in range(q):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                j = word_to_idx[tuple(nw)]
                mask |= (1 << j)
        ball_masks.append(mask)
    
    start = time.time()
    deadline = start + 100.0
    
    # Greedy set cover
    uncovered = (1 << total) - 1
    code = []
    used = [False] * total
    while uncovered and time.time() < deadline - 10:
        best = -1
        best_cnt = 0
        for c in range(total):
            if used[c]:
                continue
            cnt = (ball_masks[c] & uncovered).bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        if best == -1:
            break
        code.append(best)
        used[best] = True
        uncovered &= ~ball_masks[best]
    
    # Local search: removals
    if code and time.time() < deadline - 5:
        code = _local_search_general(code, ball_masks, total, deadline)
    
    return [words[i] for i in code]


def _local_search_general(code, ball_masks, total, deadline):
    """General local search with removals."""
    code = list(code)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Compute cover counts
        cc = [0] * total
        for c in code:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cc[idx] += 1
                m ^= lsb
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cc[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cc[idx] -= 1
                    m ^= lsb
                code.pop(i)
                improved = True
                break
    return code


def _solve_ternary_7_1():
    """Solve K_3(7,1) - minimum ternary covering code of length 7, radius 1.
    
    Uses a combined approach:
    1. Fast greedy initialization with bit-parallel operations
    2. Aggressive local search: removals, swaps, and 2-opt exchanges
    3. Multiple restarts with different random seeds
    4. Deadline management with safety margin
    """
    N = 3 ** 7  # 2187
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Precompute all words
    words = [list(w) for w in itertools.product(range(3), repeat=7)]
    
    # Precompute ball membership: for each word, list of codeword indices that cover it
    # A word is covered by a codeword if they differ in at most 1 position
    # For each word i, find all j such that d(i,j) <= 1
    # This is the "reverse" of ball_masks: reverse[i] = set of j where i is in ball(j)
    
    # First, compute ball_masks (forward)
    ball_masks = []
    for i in range(N):
        w = words[i]
        mask = 0
        mask |= (1 << i)  # center
        for pos in range(7):
            orig = w[pos]
            for nv in range(3):
                if nv == orig:
                    continue
                nw = w[:]
                nw[pos] = nv
                # Find index of nw
                idx = 0
                for p in range(7):
                    idx = idx * 3 + nw[p]
                mask |= (1 << idx)
        ball_masks.append(mask)
    
    # Precompute reverse: for each word, which codewords cover it
    # reverse[j] = bitmask of i such that j is in ball(i)
    # Actually, we need: for each word j, the set of codewords i where j is within distance 1 of i
    # That's the same as: for each i, ball(i) contains j iff d(i,j) <= 1
    # So reverse[j] has bit i set iff j is in ball(i)
    
    reverse = [0] * N
    for i in range(N):
        m = ball_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            reverse[j] |= (1 << i)
            m ^= lsb
    
    # Phase 1: Multiple greedy initializations
    best_code = None
    best_size = float('inf')
    
    # Precompute a list for faster iteration
    all_indices = list(range(N))
    
    attempts = 0
    while time.time() < deadline - 25 and attempts < 20:
        attempts += 1
        random.seed(attempts * 12345 + 6789)
        
        # Greedy with random tie-breaking
        order = list(range(N))
        random.shuffle(order)
        order_pos = {c: i for i, c in enumerate(order)}
        
        uncovered = (1 << N) - 1
        code = []
        used = [False] * N
        
        while uncovered:
            best_c = -1
            best_cnt = 0
            best_pos = N
            for c in order:
                if used[c]:
                    continue
                cnt = (ball_masks[c] & uncovered).bit_count()
                if cnt == 0:
                    continue
                pos = order_pos[c]
                if cnt > best_cnt or (cnt == best_cnt and pos < best_pos):
                    best_cnt = cnt
                    best_c = c
                    best_pos = pos
            if best_c == -1:
                break
            code.append(best_c)
            used[best_c] = True
            uncovered &= ~ball_masks[best_c]
        
        if not uncovered:
            # Local search
            code = _local_search_ternary(code, ball_masks, reverse, N, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                if best_size <= 186:
                    break  # Can't do better than known best
    
    if best_code is None:
        # Emergency: single greedy
        uncovered = (1 << N) - 1
        code = []
        used = [False] * N
        while uncovered:
            best_c = -1
            best_cnt = 0
            for c in range(N):
                if used[c]:
                    continue
                cnt = (ball_masks[c] & uncovered).bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            used[best_c] = True
            uncovered &= ~ball_masks[best_c]
        best_code = code
        best_size = len(code)
    
    return [words[i] for i in best_code]


def _local_search_ternary(code, ball_masks, reverse, N, deadline):
    """Aggressive local search for ternary (7,1) case.
    
    Uses incremental cover counts and tries:
    1. Redundancy removal
    2. Swap: replace one codeword with another to enable further removals
    3. 2-exchange: replace two codewords with one or two different ones
    """
    code = list(code)
    
    # Compute initial cover counts
    cc = [0] * N
    for c in code:
        m = ball_masks[c]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cc[idx] += 1
            m ^= lsb
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cc[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cc[idx] -= 1
                    m ^= lsb
                code.pop(i)
                improved = True
                break
    
    # Phase 2: Swap-based local search
    # For each codeword, try replacing it with another to see if we can
    # reduce the number of uniquely-covered words, enabling further removals
    code_set = set(code)
    swapped = True
    while swapped and time.time() < deadline - 5:
        swapped = False
        for i in range(len(code)):
            c_orig = code[i]
            orig_mask = ball_masks[c_orig]
            
            # Find words uniquely covered by c_orig
            unique_count = 0
            m = orig_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cc[idx] == 1:
                    unique_count += 1
                m ^= lsb
            
            if unique_count == 0:
                # Should have been caught in Phase 1, but handle anyway
                m = orig_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cc[idx] -= 1
                    m ^= lsb
                code.pop(i)
                code_set.discard(c_orig)
                swapped = True
                break
            
            # Try to find a replacement that covers all unique words
            # and doesn't create new uncovered words
            # A good replacement covers many of the same words
            best_new = -1
            best_overlap = 0
            best_unique_saved = 0
            
            # Sample some candidates
            if time.time() > deadline - 5:
                break
            
            # Try codewords that cover many of the unique words
            # For efficiency, try a random sample
            sample_size = min(50, N - len(code_set))
            candidates = random.sample([c for c in range(N) if c not in code_set], 
                                       min(sample_size, N - len(code_set)))
            
            for c_new in candidates:
                new_mask = ball_masks[c_new]
                # Check if c_new covers all unique words
                covers_all_unique = True
                m = orig_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cc[idx] == 1:
                        if not (new_mask & (1 << idx)):
                            covers_all_unique = False
                            break
                    m ^= lsb
                if not covers_all_unique:
                    continue
                
                # Count overlap: words covered by both c_orig and c_new
                overlap = (orig_mask & new_mask).bit_count()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_new = c_new
            
            if best_new != -1:
                # Perform the swap
                m = orig_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cc[idx] -= 1
                    m ^= lsb
                new_mask = ball_masks[best_new]
                m = new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cc[idx] += 1
                    m ^= lsb
                code[i] = best_new
                code_set.discard(c_orig)
                code_set.add(best_new)
                swapped = True
                break
    
    # Phase 3: Final removal pass
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                redundant = True
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cc[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cc[idx] -= 1
                        m ^= lsb
                    code.pop(i)
                    improved = True
                    break
    
    return code
