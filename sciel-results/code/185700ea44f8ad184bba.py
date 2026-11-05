import time
import math
import itertools

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all words
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute neighbors as bitmasks
    # Each word covers itself and words at distance 1
    # Total words covered by one codeword = 1 + n*(q-1) = 1 + 6*2 = 13
    cover_masks = [0] * N
    for i, c in enumerate(words):
        mask = 1 << i
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                idx = word_to_idx[tuple(new_w)]
                mask |= (1 << idx)
        cover_masks[i] = mask
        
    # Precompute exclusion masks for lower bound
    # Two words w1, w2 can be covered by the same codeword iff dist(w1, w2) <= 2
    # So if we pick w1, we "exclude" all w2 with dist(w1, w2) <= 2 from being picked in the greedy packing
    exclusion_masks = [0] * N
    for i, c in enumerate(words):
        mask = 0
        # Find all words within distance 2 of c
        # This is the union of cover_masks for all words at distance 1 from c, plus c itself?
        # Actually, dist(w, c) <= 2 iff w is in cover_masks[x] for some x in cover_masks[c] \ {c}?
        # Simpler: iterate all words j and check distance? 729*729 is ~500k, acceptable once.
        # Or: dist(w, c) <= 2 means w is a neighbor of a neighbor of c.
        # Let's just compute it by iterating j
        for j in range(N):
            # Hamming distance
            # Optimization: precompute distance matrix? 729x729 ints = 500k ints, ~4MB.
            # Or just use the cover masks.
            # w is within dist 2 of c iff w is in cover_masks[x] for some x in neighbors(c) U {c}.
            # Let's use a set of indices for neighbors of c (dist 1)
            pass
            
    # Let's precompute distance <= 2 masks more efficiently
    # dist(w, c) <= 2 iff w is in cover_masks[x] where x is at dist 1 from c, OR w==c.
    # Actually, cover_masks[x] contains x and its dist-1 neighbors.
    # So union of cover_masks[x] for x in {c} U neighbors(c) gives all words at dist <= 2 from c.
    
    neighbors_indices = [[] for _ in range(N)]
    for i in range(N):
        # Extract indices from cover_masks[i]
        mask = cover_masks[i]
        inds = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            inds.append(idx)
            m ^= lsb
        neighbors_indices[i] = inds
        # Build exclusion mask
        em = 0
        for x in inds:
            em |= cover_masks[x]
        exclusion_masks[i] = em

    best_size = N # Upper bound
    best_code = None
    
    # Bitmask of all words
    all_words_mask = (1 << N) - 1
    
    # We fix the first codeword to 000000 (index 0) due to symmetry
    # This is a valid symmetry break because the problem is invariant under global symbol/coord permutations
    # and translation? No, not translation. But we can map 000000 to any word.
    # So we can assume 000000 is in the code.
    
    def popcount(x):
        return bin(x).count("1")
        
    def lower_bound(uncovered_mask):
        # Bound 1: Ceiling of count / 13
        count = popcount(uncovered_mask)
        lb1 = math.ceil(count / 13)
        
        # Bound 2: Greedy packing
        # Pick a word, remove all words within dist 2
        u = uncovered_mask
        count2 = 0
        while u:
            # Pick the first set bit (smallest index)
            lsb = u & -u
            idx = lsb.bit_length() - 1
            count2 += 1
            u &= ~exclusion_masks[idx]
            
        return max(lb1, count2)

    def solve(uncovered_mask, last_idx, current_size, code_list):
        nonlocal best_size, best_code, deadline, start_time
        
        if time.time() > deadline:
            return
            
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = list(code_list)
            return
            
        # Lower bound check
        lb = lower_bound(uncovered_mask)
        if current_size + lb >= best_size:
            return
            
        # Iterate through candidates
        # Candidate j must be > last_idx to ensure unique set and canonical ordering
        # Also, candidate j must cover at least one uncovered word
        for j in range(last_idx + 1, N):
            # Check if j covers any uncovered word
            if not (cover_masks[j] & uncovered_mask):
                continue
                
            # Symmetry breaking refinement:
            # If we haven't picked the first word, we force it to be 0.
            # We start with last_idx = -1, but we force j=0 for the first pick?
            # In the initial call, we can handle this.
            
            # Update state
            new_uncovered = uncovered_mask & ~cover_masks[j]
            code_list.append(j)
            solve(new_uncovered, j, current_size + 1, code_list)
            code_list.pop()
            
            if time.time() > deadline:
                return

    # Initial call
    # Fix first codeword to 0 (index 0)
    # uncovered = all_words_mask ^ cover_masks[0]
    init_uncovered = all_words_mask & ~cover_masks[0]
    solve(init_uncovered, 0, 1, [0])
    
    if best_code is None:
        # Fallback if B&B fails or times out without finding a solution (should not happen)
        # Return the 81-word construction
        k4 = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        selected = set()
        for c in k4:
            for suffix in itertools.product(range(3), repeat=2):
                w = c + list(suffix)
                selected.add(word_to_idx[tuple(w)])
        return [words[i] for i in selected]
        
    return [words[i] for i in best_code]
