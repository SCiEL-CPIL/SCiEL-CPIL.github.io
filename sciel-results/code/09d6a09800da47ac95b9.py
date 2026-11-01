"""Optimized covering code for K_3(6,1) using local search with codeword swaps."""
import time
import itertools
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as numpy array of shape (q^n, n)."""
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)


def _coverage_matrix(words, q, n, radius=1):
    """
    Build a boolean coverage matrix of shape (total, total) where
    coverage[i, j] = True if word j is within Hamming distance <= radius of word i.
    """
    total = len(words)
    cov = np.zeros((total, total), dtype=bool)
    if radius == 1:
        word_to_idx = {}
        for idx, w in enumerate(words):
            word_to_idx[tuple(w)] = idx
        for i in range(total):
            w = words[i]
            cov[i, i] = True
            for pos in range(n):
                old_val = w[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        neighbor = list(w)
                        neighbor[pos] = new_val
                        j = word_to_idx[tuple(neighbor)]
                        cov[i, j] = True
    else:
        for i in range(total):
            for j in range(total):
                if np.sum(words[i] != words[j]) <= radius:
                    cov[i, j] = True
    return cov


def _greedy_cover(q, n, radius, deadline):
    """
    Vectorized greedy set cover.
    Returns list of codewords (as lists of ints).
    """
    words = _all_words(q, n)
    total = len(words)
    cov = _coverage_matrix(words, q, n, radius)
    
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    coverage_counts = cov.sum(axis=1).astype(np.int32)
    
    while uncovered.any():
        if time.time() > deadline:
            break
        best_idx = np.argmax(coverage_counts)
        best_count = coverage_counts[best_idx]
        if best_count == 0:
            break
        chosen.append(best_idx)
        newly_covered = cov[best_idx] & uncovered
        if newly_covered.any():
            uncovered[newly_covered] = False
            reductions = cov[:, newly_covered].sum(axis=1)
            coverage_counts -= reductions
    
    return [words[i].tolist() for i in chosen]


def _local_search_swap(words, cov, chosen_indices, deadline):
    """
    Local search: try replacing one codeword with another to reduce total size
    or maintain coverage with fewer words.
    
    Strategy: 
    1. Start with greedy solution.
    2. Try to remove a codeword if all its covered words are still covered by others.
    3. Try swapping: remove codeword A, add codeword B if it improves coverage.
    
    Returns improved list of codeword indices.
    """
    chosen = set(chosen_indices)
    total = len(words)
    
    # Build coverage union: for each word j, how many chosen codewords cover it?
    # cover_count[j] = number of chosen codewords that cover word j
    cover_count = np.zeros(total, dtype=np.int32)
    for idx in chosen:
        cover_count[cov[idx]] += 1
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Phase 1: Try to remove redundant codewords
        for idx in list(chosen):
            if time.time() > deadline:
                break
            # If we remove idx, words that were only covered by idx become uncovered
            # A word j is only covered by idx if cover_count[j] == 1 and cov[idx, j]
            # Check if any such word exists
            only_covered_by_idx = cov[idx] & (cover_count == 1)
            if not only_covered_by_idx.any():
                # Safe to remove
                cover_count[cov[idx]] -= 1
                chosen.discard(idx)
                improved = True
                break
        
        # Phase 2: Try to swap - remove one, add another that covers more uncovered
        if not improved and time.time() < deadline:
            # Find words not covered by current set
            uncovered = cover_count == 0
            if uncovered.any():
                # For each candidate to remove, compute what becomes uncovered
                # For each candidate to add, compute what it covers
                # Net gain = new_covered - newly_uncovered
                best_swap = None
                best_gain = 0
                
                # Limit search for speed
                remove_candidates = list(chosen)
                add_candidates = [i for i in range(total) if i not in chosen]
                
                # Sample if too many candidates
                if len(add_candidates) > 200:
                    import random
                    random.seed(42)
                    add_candidates = random.sample(add_candidates, 200)
                
                for rem in remove_candidates:
                    if time.time() > deadline:
                        break
                    # If we remove rem, these words lose one coverage
                    rem_covered = cov[rem]
                    # Words that become uncovered: those with cover_count == 1 that rem covers
                    would_uncover = rem_covered & (cover_count == 1)
                    uncount = would_uncover.sum()
                    
                    if uncount > 0 and best_gain >= uncount:
                        continue
                    
                    for add in add_candidates:
                        if time.time() > deadline:
                            break
                        add_covered = cov[add]
                        # Words that get newly covered: those with cover_count == 0 that add covers
                        new_cover = add_covered & (cover_count == 0)
                        new_count = new_cover.sum()
                        
                        gain = new_count - uncount
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (rem, add)
                
                if best_swap is not None:
                    rem, add = best_swap
                    # Execute swap
                    cover_count[cov[rem]] -= 1
                    cover_count[cov[add]] += 1
                    chosen.discard(rem)
                    chosen.add(add)
                    improved = True
    
    return list(chosen)


def _local_search_add_remove(words, cov, chosen_indices, deadline):
    """
    Alternative local search: try adding a codeword then removing two, or 
    more sophisticated moves.
    """
    chosen = set(chosen_indices)
    total = len(words)
    
    cover_count = np.zeros(total, dtype=np.int32)
    for idx in chosen:
        cover_count[cov[idx]] += 1
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Try: remove 2, add 1 (net -1 codeword)
        # This is expensive, so sample
        if len(chosen) > 70:
            remove_pairs = list(itertools.combinations(list(chosen), 2))
            if len(remove_pairs) > 500:
                import random
                random.seed(123)
                remove_pairs = random.sample(remove_pairs, 500)
            
            add_candidates = [i for i in range(total) if i not in chosen]
            if len(add_candidates) > 150:
                import random
                random.seed(456)
                add_candidates = random.sample(add_candidates, 150)
            
            for (r1, r2) in remove_pairs:
                if time.time() > deadline:
                    break
                # Compute coverage if we remove r1 and r2
                # Words that become uncovered
                would_uncover = (cov[r1] | cov[r2]) & (cover_count <= 2)
                # More precisely: words where cover_count would drop to 0
                # A word j is uncovered if cover_count[j] - (cov[r1,j] + cov[r2,j]) < 0
                # i.e., cover_count[j] < cov[r1,j] + cov[r2,j]
                # Since cov entries are 0 or 1: cover_count[j] < number of {r1,r2} covering j
                r1_r2_cover = cov[r1] | cov[r2]
                r1_only = cov[r1] & ~cov[r2]
                r2_only = cov[r2] & ~cov[r1]
                both = cov[r1] & cov[r2]
                
                # Words that become uncovered:
                # - covered only by r1: r1_only & (cover_count == 1)
                # - covered only by r2: r2_only & (cover_count == 1)  
                # - covered by both: both & (cover_count == 2)
                un1 = r1_only & (cover_count == 1)
                un2 = r2_only & (cover_count == 1)
                unboth = both & (cover_count == 2)
                total_uncover = un1 | un2 | unboth
                uncount = total_uncover.sum()
                
                if uncount > 0:
                    continue
                
                # If safe to remove both, find best add
                best_add = None
                best_gain = 0
                for add in add_candidates:
                    if time.time() > deadline:
                        break
                    add_covered = cov[add]
                    # After removing r1, r2, what does add cover that's uncovered?
                    # New cover_count = cover_count - cov[r1] - cov[r2]
                    # Words with new cover_count == 0 that add covers
                    new_uncovered = (cover_count - cov[r1] - cov[r2]) == 0
                    new_cover = add_covered & new_uncovered
                    gain = new_cover.sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add
                
                if best_add is not None and best_gain > 0:
                    # Execute: remove r1, r2, add best_add
                    cover_count[cov[r1]] -= 1
                    cover_count[cov[r2]] -= 1
                    cover_count[cov[best_add]] += 1
                    chosen.discard(r1)
                    chosen.discard(r2)
                    chosen.add(best_add)
                    improved = True
                    break
    
    return list(chosen)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    words = _all_words(q, n)
    total = len(words)
    cov = _coverage_matrix(words, q, n, radius)
    
    # Step 1: Greedy set cover to get initial solution
    greedy_code = _greedy_cover(q, n, radius, deadline)
    
    if not greedy_code:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Convert to indices
    word_to_idx = {}
    for idx, w in enumerate(words):
        word_to_idx[tuple(w)] = idx
    
    chosen_indices = set()
    for cw in greedy_code:
        chosen_indices.add(word_to_idx[tuple(cw)])
    
    # Step 2: Local search with swaps
    if time.time() < deadline:
        chosen_indices = set(_local_search_swap(words, cov, list(chosen_indices), deadline))
    
    # Step 3: More aggressive local search (remove 2, add 1)
    if time.time() < deadline:
        chosen_indices = set(_local_search_add_remove(words, cov, list(chosen_indices), deadline))
    
    # Step 4: Final greedy fill if any words are uncovered
    if time.time() < deadline:
        cover_count = np.zeros(total, dtype=np.int32)
        for idx in chosen_indices:
            cover_count[cov[idx]] += 1
        uncovered = cover_count == 0
        if uncovered.any():
            # Greedy fill for remaining uncovered
            remaining_uncovered = uncovered.copy()
            while remaining_uncovered.any() and time.time() < deadline:
                # Find codeword covering most remaining uncovered
                best_idx = -1
                best_count = 0
                for i in range(total):
                    if i in chosen_indices:
                        continue
                    count = (cov[i] & remaining_uncovered).sum()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                if best_idx == -1 or best_count == 0:
                    break
                chosen_indices.add(best_idx)
                cover_count[cov[best_idx]] += 1
                remaining_uncovered &= ~cov[best_idx]
    
    # Convert back to codewords
    code = [words[i].tolist() for i in sorted(chosen_indices)]
    
    # Verify coverage
    if len(code) == 0:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    return code
