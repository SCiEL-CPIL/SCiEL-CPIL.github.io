import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Specific task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total = 729
        powers = [1, 3, 9, 27, 81, 243]
        
        # Precompute neighbors as bitmasks
        neighbors = [0] * total
        for x in range(total):
            mask = 1 << x
            for i in range(n):
                digit = (x // powers[i]) % q
                if digit != 0:
                    mask |= (1 << (x - powers[i]))
                if digit != q - 1:
                    mask |= (1 << (x + powers[i]))
            neighbors[x] = mask
            
        # Initial construction: Tetracode x Identity(2)
        # Tetracode: 9 words of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tails = list(itertools.product(range(3), repeat=n - 4))
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * 3 + val
            return idx
        
        initial_indices = []
        for h in tetracode:
            for tail in tails:
                initial_indices.append(word_to_idx(h + list(tail)))
                
        best_code = initial_indices.copy()
        best_size = len(best_code)
        
        # Helper to check validity and return coverage
        def get_coverage(code):
            cov = np.zeros(total, dtype=np.int32)
            for idx in code:
                # Iterate bits in mask
                m = neighbors[idx]
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    cov[bit_pos] += 1
                    m ^= lsb
            return cov

        # Local Search
        current_code = initial_indices.copy()
        current_size = len(current_code)
        in_code = np.zeros(total, dtype=bool)
        for idx in current_code:
            in_code[idx] = True
            
        coverage = get_coverage(current_code)
        
        def try_deletion():
            nonlocal current_code, current_size, in_code, coverage
            removed_any = False
            # Try to remove redundant codewords
            # A codeword c is redundant if min(coverage[neighbors[c]]) >= 2
            # We iterate in reverse to allow popping
            for i in range(len(current_code) - 1, -1, -1):
                if time.time() > deadline:
                    return removed_any
                idx = current_code[i]
                mask = neighbors[idx]
                # Check min coverage in neighborhood
                # We need to check if all bits in mask have coverage >= 2
                # Efficient check:
                # Create a mask of words with coverage >= 2
                # cov_mask = sum((1 << i) for i in range(total) if coverage[i] >= 2) is too slow
                # Instead, iterate bits in mask and check coverage
                redundant = True
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    if coverage[bit_pos] < 2:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        coverage[bit_pos] -= 1
                        m ^= lsb
                    in_code[idx] = False
                    current_code.pop(i)
                    current_size -= 1
                    removed_any = True
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code.copy()
            return removed_any

        def try_swap():
            nonlocal current_code, current_size, in_code, coverage
            # Pick a random codeword to potentially swap
            # We look for a codeword c such that removing it leaves U uncovered,
            # and there exists c_new not in code covering U.
            
            # To speed up, pick a random codeword
            if not current_code:
                return False
                
            idx = random.choice(current_code)
            mask = neighbors[idx]
            
            # Find uncovered words U if idx is removed
            # U = { w in mask | coverage[w] == 1 }
            U_mask = 0
            m = mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                if coverage[bit_pos] == 1:
                    U_mask |= lsb
                m ^= lsb
            
            if U_mask == 0:
                # It was redundant, should have been caught by deletion
                return False
                
            # Find intersection of neighborhoods of all words in U
            # Candidates must be in N(u) for all u in U
            # And not in current code
            
            # Initialize candidate mask with all 1s (total bits)
            # But we only care about bits 0..total-1
            cand_mask = (1 << total) - 1
            
            m = U_mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                cand_mask &= neighbors[bit_pos]
                if cand_mask == 0:
                    return False
                m ^= lsb
            
            # Remove current code from candidates
            # We can iterate through current_code and clear bits, or maintain a mask
            # Maintaining a mask is faster
            # Let's maintain code_mask
            # But we don't have it in nonlocal scope easily without updating it everywhere
            # Let's just check membership in the final selection
            
            # Pick a random candidate from cand_mask
            # Get list of bits
            # Since total is 729, iterating is okay
            candidates = []
            m = cand_mask
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                if not in_code[bit_pos]:
                    candidates.append(bit_pos)
                m ^= lsb
            
            if not candidates:
                return False
                
            new_idx = random.choice(candidates)
            
            # Perform Swap: Remove idx, Add new_idx
            # Update coverage for idx (decrement)
            m = neighbors[idx]
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                coverage[bit_pos] -= 1
                m ^= lsb
            in_code[idx] = False
            current_code.remove(idx) # O(N) but N~80, fine
            
            # Update coverage for new_idx (increment)
            m = neighbors[new_idx]
            while m:
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                coverage[bit_pos] += 1
                m ^= lsb
            in_code[new_idx] = True
            current_code.append(new_idx)
            
            # Now try to delete redundant codewords
            try_deletion()
            return True

        # Main Loop
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            # Always try deletion first
            changed = try_deletion()
            
            if current_size < best_size:
                best_size = current_size
                best_code = current_code.copy()
                
            if not changed and iterations % 10 == 0:
                # Try swaps to escape local minima
                # Do a few swaps
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    try_swap()
                    
            # Occasionally restart with a different random initialization if stuck
            if iterations % 500 == 0:
                # Try a greedy set cover with randomization
                uncovered = (1 << total) - 1
                greedy_code = []
                while uncovered:
                    max_cov = -1
                    best_cands = []
                    # Sample a subset of words to speed up? No, 729 is small.
                    for w in range(total):
                        overlap = (neighbors[w] & uncovered).bit_count()
                        if overlap > max_cov:
                            max_cov = overlap
                            best_cands = [w]
                        elif overlap == max_cov:
                            best_cands.append(w)
                    if not best_cands:
                        break
                    chosen = random.choice(best_cands)
                    greedy_code.append(chosen)
                    uncovered &= ~neighbors[chosen]
                
                if len(greedy_code) < current_size:
                    current_code = greedy_code
                    current_size = len(current_code)
                    in_code = np.zeros(total, dtype=bool)
                    for idx in current_code:
                        in_code[idx] = True
                    coverage = get_coverage(current_code)
                    # Immediately prune
                    try_deletion()

        # Convert best_code back to word lists
        res = []
        for idx in best_code:
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        
        return res

    # Generic fallback
    return [[0]*n]
