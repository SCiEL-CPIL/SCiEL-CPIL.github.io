import itertools
import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Build initial direct-sum code: tetracode x all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    code = initial[:]
    m = len(code)
    
    # Precompute code as numpy array
    code_arr = np.array(code, dtype=np.int32)
    
    # Total space size
    space_size = q ** n
    
    # Coverage count: for each of the 3^n words, how many codewords cover it (within Hamming dist <= radius)
    # For radius=1, q=3, n=7: each codeword covers 1 + n*q = 1+21 = 22 words
    # We need to track coverage for all 2187 words
    
    # Map a word (tuple of n digits in 0..q-1) to an index 0..3^n-1
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    # Precompute all words and their indices
    # For efficiency, precompute the list of words each codeword covers
    # A word w is within distance 1 of codeword c if they differ in at most 1 position
    
    # Precompute coverage: cov[idx] = number of codewords covering word idx
    cov = np.zeros(space_size, dtype=np.int32)
    
    # For each codeword, add 1 to all words within distance radius
    # For radius=1: the word itself + all single-position changes
    # Precompute the neighbor offsets for radius 1
    # For position i, changing digit d to d' (d' != d) gives offset: (d'-d) * q^(n-1-i)
    
    # Precompute powers
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q
    
    # For radius=1, neighbors of a word at index idx:
    # We can enumerate by position and new digit
    def get_neighbors_idx(idx):
        """Get indices of all words within Hamming distance 1 of the word at idx."""
        neighbors = [idx]
        for pos in range(n):
            # Current digit at this position
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                # New index
                delta = (new_digit - digit) * q_powers[n-1-pos]
                neighbors.append(idx + delta)
        return neighbors
    
    # Build coverage: for each codeword, increment coverage of all its neighbors
    # This is O(m * (1+n*q)) = O(729 * 22) = ~16000 ops, fast
    
    # But we need to be able to remove codewords and update coverage
    # cov[i] = sum over remaining codewords of [word i is within dist radius of codeword]
    
    # Initialize coverage
    for c in code:
        c_idx = word_to_idx(c)
        for nb in get_neighbors_idx(c_idx):
            cov[nb] += 1
    
    # Now iteratively try to remove codewords
    # A codeword can be removed if no word has coverage == 1 that is only covered by this codeword
    # i.e., for all neighbors of this codeword, cov[neighbor] >= 2
    
    # We'll use a greedy approach: repeatedly scan and remove removable codewords
    # To speed up, maintain a set of "critical" words (cov==1) and their unique cover
    
    # Actually, let's just do iterative passes:
    # In each pass, try to remove each remaining codeword.
    # If removing it would leave some word with cov==0, skip.
    # This is O(m * (1+n*q)) per pass, and we do multiple passes.
    
    # Better: maintain for each word, the count. A codeword c can be removed iff
    # min(cov[neighbor] for neighbor in neighbors(c)) >= 2
    
    # Let's precompute neighbor lists for all possible indices
    # Space is 3^7 = 2187, each has 22 neighbors. Total: 2187*22 = ~48k entries.
    all_neighbors = {}
    for idx in range(space_size):
        all_neighbors[idx] = get_neighbors_idx(idx)
    
    # Convert to numpy for speed? Let's keep as list of lists for now.
    
    # Iterative removal
    removed = [False] * m
    changed = True
    while changed and (time.time() - t_start) < deadline:
        changed = False
        for i in range(m):
            if removed[i]:
                continue
            if (time.time() - t_start) > deadline:
                break
            c_idx = word_to_idx(code[i])
            nbs = all_neighbors[c_idx]
            # Check if all neighbors have cov >= 2
            can_remove = True
            for nb in nbs:
                if cov[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                removed[i] = True
                for nb in nbs:
                    cov[nb] -= 1
                changed = True
    
    # Build result
    result = [code[i] for i in range(m) if not removed[i]]
    
    # Verify coverage (safety check, but also we should do a more thorough check)
    # Let's do a quick verification
    if (time.time() - t_start) < deadline - 5:
        # Verify: every word in space is covered
        cov_final = np.zeros(space_size, dtype=np.int32)
        for c in result:
            c_idx = word_to_idx(c)
            for nb in all_neighbors[c_idx]:
                cov_final[nb] += 1
        if np.any(cov_final == 0):
            # Coverage broken! Return original
            return initial
    
    return result
