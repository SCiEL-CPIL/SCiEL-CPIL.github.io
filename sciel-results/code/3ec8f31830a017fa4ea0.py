"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search: fast greedy init, then iterative swaps to minimize codewords.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # Precompute all words as integers 0..q^n-1
    # Build neighbor list: for each word, which words are within Hamming distance <= radius
    
    # Efficient neighbor generation using digit positions
    def get_neighbors(w_int, q, n, radius):
        """Return set of word ints within Hamming distance <= radius from w_int."""
        # Convert to digits
        digits = []
        tmp = w_int
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits = digits[::-1]  # most significant first
        
        neighbors = {w_int}
        
        if radius == 0:
            return neighbors
        
        # Generate all words at distance 1
        dist1 = set()
        for pos in range(n):
            for new_val in range(q):
                if new_val == digits[pos]:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                val = 0
                for d in new_digits:
                    val = val * q + d
                dist1.add(val)
        
        neighbors.update(dist1)
        
        if radius == 1:
            return neighbors
        
        # Generate words at distance 2
        dist2 = set()
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                for val1 in range(q):
                    if val1 == digits[pos1]:
                        continue
                    for val2 in range(q):
                        if val2 == digits[pos2]:
                            continue
                        new_digits = digits[:]
                        new_digits[pos1] = val1
                        new_digits[pos2] = val2
                        val = 0
                        for d in new_digits:
                            val = val * q + d
                        dist2.add(val)
        
        neighbors.update(dist2)
        
        return neighbors
    
    # Precompute neighbors for all words (this is the expensive part)
    # Use a more efficient approach: precompute for each position and value,
    # which words have that value at that position
    
    # Actually, let's precompute neighbors once
    neighbors_list = [None] * total_words
    for w in range(total_words):
        neighbors_list[w] = get_neighbors(w, q, n, radius)
    
    # Convert to sets for fast lookup
    neighbors_set = [set(nb) for nb in neighbors_list]
    
    def greedy_cover():
        """Fast greedy: repeatedly pick the word that covers the most uncovered words."""
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords = []
        
        # Use a heap-like approach: track uncovered count per candidate
        # For efficiency, we'll use a simple approach with numpy
        
        while uncovered_count > 0:
            best_word = -1
            best_gain = -1
            
            # Sample a subset of candidates to speed up
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for w in candidates:
                if covered[w]:
                    continue
                # Count how many uncovered words are in neighbors of w
                gain = 0
                for u in neighbors_set[w]:
                    if not covered[u]:
                        gain += 1
                        if gain > best_gain:
                            pass  # continue counting for now
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                if best_gain == uncovered_count:
                    break
            
            if best_word == -1 or best_gain == 0:
                break
            
            codewords.append(best_word)
            for u in neighbors_set[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
        
        return codewords
    
    def local_search(codewords, time_limit):
        """
        Iteratively try to remove or swap codewords to reduce count while maintaining coverage.
        A codeword can be removed if all words it covers are covered by at least one other codeword.
        A swap replaces one codeword with another that covers the same uniquely-covered words plus more.
        """
        # Build coverage count array
        cov_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors_set[cw]:
                cov_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_all_redundant():
            """Remove all codewords that are redundant (all their neighbors are covered by others)."""
            removed = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    # Check if removing cw leaves all its neighbors covered
                    safe = True
                    for u in neighbors_set[cw]:
                        if cov_count[u] <= 1:
                            safe = False
                            break
                    if safe:
                        for u in neighbors_set[cw]:
                            cov_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        removed = True
                        changed = True
                        break
            return removed
        
        try_remove_all_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_all_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            # Pick a random codeword to try to replace
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words that would become uncovered if we remove old_cw
            unique_covered = []
            for u in neighbors_set[old_cw]:
                if cov_count[u] == 1:
                    unique_covered.append(u)
            
            if len(unique_covered) == 0:
                # Can remove entirely
                for u in neighbors_set[old_cw]:
                    cov_count[u] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Try to find a replacement that covers all unique_covered words
            # Candidate replacements: neighbors of words in unique_covered
            candidate_set = set()
            for u in unique_covered:
                candidate_set.update(neighbors_set[u])
            
            candidates = [c for c in candidate_set if c not in cw_set and c != old_cw]
            random.shuffle(candidates)
            
            best_candidate = -1
            best_score = -1
            
            for new_cw in candidates:
                # Check if new_cw covers all unique_covered
                valid = True
                for u in unique_covered:
                    if u not in neighbors_set[new_cw]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: number of words that new_cw covers that old_cw doesn't
                # (these would gain extra coverage)
                score = 0
                for u in neighbors_set[new_cw]:
                    if u not in neighbors_set[old_cw]:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_candidate = new_cw
            
            if best_candidate != -1:
                # Perform the swap
                for u in neighbors_set[old_cw]:
                    cov_count[u] -= 1
                for u in neighbors_set[best_candidate]:
                    cov_count[u] += 1
                codewords[idx] = best_candidate
                cw_set.discard(old_cw)
                cw_set.add(best_candidate)
        
        try_remove_all_redundant()
        return codewords
    
    # Main: multiple restarts
    start_time = time.time()
    total_time_limit = 115
    best_codewords = None
    best_len = float('inf')
    
    while time.time() - start_time < total_time_limit:
        remaining_time = total_time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        # Greedy init
        cw = greedy_cover()
        
        # Local search
        opt_time = min(remaining_time - 3, 30)  # Cap per iteration
        if opt_time > 0:
            cw = local_search(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
        
        # Early stop if we hit a very good solution
        if best_len <= 403:
            break
    
    if best_codewords is None:
        best_codewords = greedy_cover()
    
    # Convert back to list of lists
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
