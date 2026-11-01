import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    space_size = q ** n
    powers = [q ** i for i in range(n)]

    # Precompute neighbors as bitmasks for fast set operations
    # This is the key structural change: using bitmasks allows
    # extremely fast coverage checks via bitwise AND and bit_count.
    neighbors_mask = [0] * space_size
    for x in range(space_size):
        mask = 1 << x
        for i in range(n):
            digit = (x // powers[i]) % q
            if radius >= 1:
                for delta in range(1, radius + 1):
                    if digit - delta >= 0:
                        mask |= 1 << (x - delta * powers[i])
                    if digit + delta < q:
                        mask |= 1 << (x + delta * powers[i])
        neighbors_mask[x] = mask

    all_mask = (1 << space_size) - 1

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # 1. Initial Construction: Direct sum of tetracode and all words of length n-4
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Generate tails
    tails = list(itertools.product(range(q), repeat=n - 4))
    
    # Construct initial codewords as indices
    initial_indices = []
    for h in tetracode:
        for tail in tails:
            idx = 0
            for val in h + list(tail):
                idx = idx * q + val
            initial_indices.append(idx)

    def greedy_reduce(code_indices, uncovered_mask):
        """
        Greedily remove codewords that are redundant.
        A codeword is redundant if removing it does not leave any word uncovered.
        We iterate until no more removals are possible.
        """
        code_set = set(code_indices)
        counts = [0] * space_size
        for idx in code_set:
            # Count coverage contributions
            # We can just recompute uncovered mask after removal attempts
            pass
        
        # More efficient: maintain uncovered mask? 
        # Actually, for removal, we need to know if a codeword is the *only* cover for some word.
        # Let's compute coverage counts.
        # Since space is small (729), we can do this.
        
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                # Check if removing idx leaves any word uncovered
                # Words covered by idx: neighbors_mask[idx]
                # If any word in neighbors_mask[idx] is covered ONLY by idx, we can't remove it.
                # How to check "covered only by idx"?
                # We need to know if (uncovered_mask_after_removal) is 0.
                # uncovered_after_removal = current_uncovered | (neighbors_mask[idx] & ~covered_by_others)
                # This is complex to maintain incrementally without counts.
                
                # Let's use counts.
                pass
            
            # Alternative: 
            # 1. Compute coverage counts for current set.
            # 2. For each codeword, check if min count in its sphere is > 1.
            # 3. If so, remove it and update counts.
            
            counts = np.zeros(space_size, dtype=np.int32)
            for idx in code_set:
                # Add 1 to all bits set in neighbors_mask[idx]
                # This is slow in Python loop. 
                # But space is 729. 
                # Let's stick to the bitmask trick for coverage check if possible.
                
                # Actually, let's just use the counts array updated carefully.
                pass

            # Let's try a different approach for reduction:
            # Iterate through codewords. For each, temporarily remove, check if coverage is still full.
            # To check if coverage is full, we need the union of neighbors of remaining codewords.
            # union = 0
            # for idx in code_set - {idx_to_check}: union |= neighbors_mask[idx]
            # if union == all_mask: remove.
            
            # This is O(N^2) per pass, where N is size of code. N~80. 80^2 * 729 ops?
            # Bitwise OR is fast. 
            # Let's try this.
            
            for idx in list(code_set):
                if time.time() > deadline:
                    break
                
                # Compute union of all OTHER codewords
                union = 0
                for other in code_set:
                    if other != idx:
                        union |= neighbors_mask[other]
                
                if union == all_mask:
                    code_set.remove(idx)
                    changed = True
                    
            if time.time() > deadline:
                break

        return list(code_set)

    def local_search_swap(code_indices, max_swaps=500):
        """
        Try to swap codewords to enable further deletions or just find a smaller code.
        This is a heuristic local search.
        """
        code_set = set(code_indices)
        current_size = len(code_set)
        
        # Precompute neighbor lists for faster iteration? 
        # Bitmasks are good for "is covered" but bad for "list of neighbors".
        # Let's precompute neighbor lists for the swap logic.
        neighbor_lists = []
        for x in range(space_size):
            nb = []
            mask = neighbors_mask[x]
            while mask:
                # Extract lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                nb.append(idx)
                mask ^= lsb
            neighbor_lists.append(nb)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove
            code_set, _ = greedy_reduce(list(code_set), all_mask)
            if len(code_set) < current_size:
                current_size = len(code_set)
                improved = True
                continue
            
            # Try to swap
            # Pick a random codeword
            if not code_set:
                break
            idx_to_swap = random.choice(list(code_set))
            
            # Find a replacement that allows removal of idx_to_swap
            # Or just a replacement that maintains coverage and allows future removals?
            # Simple heuristic: Replace idx_to_swap with a random word, then try greedy_reduce.
            # If size decreases or stays same but different, keep it?
            # We want to minimize size.
            
            # Better: Try to find a word w such that (code_set - {idx_to_swap}) U {w} is a covering code.
            # Then we can try to reduce it.
            
            candidates = []
            # Sample some candidates
            for _ in range(50):
                w = random.randint(0, space_size - 1)
                if w in code_set:
                    continue
                
                new_set = (code_set - {idx_to_swap}) | {w}
                
                # Check if new_set covers everything
                union = 0
                for idx in new_set:
                    union |= neighbors_mask[idx]
                
                if union == all_mask:
                    # Valid swap
                    # Try to reduce
                    reduced = greedy_reduce(list(new_set), all_mask)
                    if len(reduced) < len(code_set):
                        code_set = set(reduced)
                        improved = True
                        break
                    elif len(reduced) == len(code_set):
                        # Maybe this new configuration is "better" for future swaps?
                        # Hard to tell. Let's stick to size reduction for now.
                        # Or we can accept it with some probability to escape local minima.
                        if random.random() < 0.1:
                            code_set = set(reduced)
                            improved = True
                            break

        return list(code_set)

    best_code = None
    best_size = float('inf')

    # Multiple restarts
    # Start with the tetracode construction
    # Also try pure greedy construction from scratch
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        random.seed(42 + restart)
        
        # Strategy 1: Start with tetracode construction
        # Strategy 2: Start with pure greedy covering
        
        if restart % 2 == 0:
            # Tetracode start
            code = list(initial_indices)
            random.shuffle(code)
        else:
            # Pure greedy start
            uncovered = all_mask
            code = []
            while uncovered:
                if time.time() > deadline:
                    break
                
                # Find best candidate
                max_cov = -1
                best_cands = []
                for w in range(space_size):
                    overlap = (neighbors_mask[w] & uncovered).bit_count()
                    if overlap > max_cov:
                        max_cov = overlap
                        best_cands = [w]
                    elif overlap == max_cov:
                        best_cands.append(w)
                        
                if not best_cands:
                    break
                
                chosen = random.choice(best_cands)
                code.append(chosen)
                uncovered &= ~neighbors_mask[chosen]
                
                if time.time() > deadline:
                    break

        # Optimize
        code = greedy_reduce(code, all_mask)
        code = local_search_swap(code, max_swaps=100)
        code = greedy_reduce(code, all_mask)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
        # If we hit the known bound 73, we can keep trying for 72 or 71
        # But 71 is lower bound. 73 is best known.
        if best_size <= 73:
            # Continue for a bit to see if we can improve
            pass

    if best_code is None:
        # Fallback
        best_code = initial_indices

    # Convert indices back to lists
    res = []
    for idx in best_code:
        res.append(idx_to_word(idx))
        
    return res
