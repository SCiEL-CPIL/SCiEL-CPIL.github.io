"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    """Optimized construction for K_3(7,1) using Greedy + Local Search (Removal & Swap)."""
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    # covers[i] is a bitmask of all words within Hamming distance 1 of word i
    covers = [0] * N
    # Also precompute the list of indices for each cover for faster iteration in local search
    cover_indices = [[] for _ in range(N)]
    
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        indices = [i]
        
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * powers[pos]
                    mask |= (1 << new_i)
                    indices.append(new_i)
        
        covers[i] = mask
        cover_indices[i] = indices
    
    full_mask = (1 << N) - 1
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            # Find best candidate: one that covers the most uncovered words
            best_count = -1
            best_indices = []
            
            # Optimization: We can skip checking candidates that are already 'bad' 
            # but for N=2187, checking all is ~2000 operations * 200 iterations = 400k ops.
            # bit_count is fast.
            
            # We can also filter candidates: if a candidate's cover has 0 intersection with uncovered, skip.
            # But we need the count.
            
            for i in all_indices:
                cnt = (uncovered & covers[i]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_indices = [i]
                elif cnt == best_count and cnt > 0:
                    best_indices.append(i)
            
            if best_count == 0:
                # Should not happen if we have a valid code, but if we are stuck?
                # Actually, if uncovered is not 0, there must be a word i such that covers[i] intersects uncovered.
                # Unless we ran out of candidates? No, all_indices is fixed.
                break
                
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search(code, deadline):
        """
        Local search with Removal and Swap moves.
        Returns the reduced code.
        """
        if not code:
            return code
        
        # Use a set for O(1) membership check
        code_set = set(code)
        
        # Initialize counts
        # counts[j] = number of codewords in code that cover word j
        counts = [0] * N
        for c in code_set:
            for j in cover_indices[c]:
                counts[j] += 1
        
        # Check initial validity (should be valid from greedy)
        # if min(counts) < 1: return code (or fix it)
        
        best_code = list(code_set)
        best_size = len(best_code)
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Removals
            # Try to remove any codeword that is redundant
            # Iterate in random order to break ties
            current_list = list(code_set)
            random.shuffle(current_list)
            
            for c in current_list:
                if c not in code_set:
                    continue
                
                # Check if removing c is safe
                # Safe if all j in covers[c] have counts[j] > 1
                safe = True
                for j in cover_indices[c]:
                    if counts[j] <= 1:
                        safe = False
                        break
                
                if safe:
                    # Remove c
                    for j in cover_indices[c]:
                        counts[j] -= 1
                    code_set.remove(c)
                    improved = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
            
            # Phase 2: Swaps
            # Try to swap a codeword with a non-codeword
            # This is more expensive. Limit attempts.
            if time.time() < deadline - 5.0:
                # Try a limited number of random swaps
                # We pick a random codeword c_old and a random non-codeword c_new
                # Check if swap is valid.
                # Valid if:
                # 1. All words covered ONLY by c_old are ALSO covered by c_new.
                #    i.e., {j | counts[j]==1 and j in covers[c_old]} is subset of covers[c_new]
                
                # To make this efficient, we can precompute "essential" words for each codeword?
                # Essential words for c are those j in covers[c] where counts[j]==1.
                # If c is removed, these become uncovered.
                # If we add c_new, they are covered if they are in covers[c_new].
                
                # Let's try a few random swaps
                swap_attempts = 50
                for _ in range(swap_attempts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    if not code_set:
                        break
                    
                    c_old = random.choice(list(code_set))
                    c_new = random.randint(0, N-1)
                    
                    if c_new in code_set:
                        continue
                    
                    # Check validity of swap
                    # Words that would become uncovered:
                    # Those j in covers[c_old] with counts[j] == 1 AND j NOT in covers[c_new]
                    
                    # Optimize: 
                    # 1. Identify words in covers[c_old] with counts[j] == 1
                    # 2. Check if any of them are NOT in covers[c_new]
                    
                    # Since covers[c_old] has 15 elements, this is fast.
                    valid = True
                    for j in cover_indices[c_old]:
                        if counts[j] == 1:
                            # j is only covered by c_old (among current code)
                            # Check if c_new covers j
                            # j is in covers[c_new] iff (covers[c_new] >> j) & 1
                            if not ((covers[c_new] >> j) & 1):
                                valid = False
                                break
                    
                    if valid:
                        # Perform swap
                        # Update counts
                        for j in cover_indices[c_old]:
                            counts[j] -= 1
                        for j in cover_indices[c_new]:
                            counts[j] += 1
                        
                        code_set.remove(c_old)
                        code_set.add(c_new)
                        improved = True
                        # Size doesn't change, but structure might allow future removals
                        # So we set improved=True to trigger another pass of Removals

        return best_code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy + Local Search
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search(code, deadline)
        if code:
            best_code = code
            best_size = len(code)
    
    # 2. Randomized Greedy + Local Search
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts * 12345)
        if not unc:
            # Local search to reduce
            code = local_search(code, deadline)
            if code and len(code) < best_size:
                best_size = len(code)
                best_code = code
                # If we hit 186, we can keep going to try for 185, but it's hard.
                # 186 is the known best.
                if best_size <= 186:
                    # Continue searching briefly to see if we can break the record
                    pass
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result

def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        # Generate all words within radius
        # This is expensive for large n, but for small n it's fine.
        # For general case, we might not have time.
        # Let's just do a simple greedy without precomputation for now.
        pass
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
