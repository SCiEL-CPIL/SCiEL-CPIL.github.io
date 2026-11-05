import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Precompute neighbors for the entire space
    # Total words = q^n
    total_words = q ** n
    # For q=3, n=6, total_words=729. Ball size = 1 + n*(q-1) = 13.
    
    # Helper to convert index to word tuple
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return tuple(word)
    
    # Helper to convert word tuple to index
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    # Precompute neighbors for all words
    # neighbors[i] contains indices of words within Hamming distance 'radius' of word i
    # For radius=1, this is 1 + n*(q-1) words.
    ball_size = 1
    if radius >= 1:
        ball_size += n * (q - 1)
        
    neighbors = np.empty((total_words, ball_size), dtype=np.int32)
    
    # Precompute powers of q for fast index calculation
    powers = [q ** p for p in range(n)]
    
    for i in range(total_words):
        word = idx_to_word(i)
        neighbors[i, 0] = i
        k = 1
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = list(word)
                    new_word[pos] = new_val
                    neighbors[i, k] = word_to_idx(tuple(new_word))
                    k += 1

    # 2. Initial Solution: Product Construction
    # Ternary tetracode (9 words of length 4) x all words on remaining n-4 coords
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Extend to length n
        # The tetracode covers {0,1,2}^4 with radius 1 perfectly.
        # We take the product with {0,1,2}^{n-4}.
        # This results in 9 * 3^(n-4) codewords.
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n-4):
                initial_codewords.append(h + list(tail))
    else:
        # Fallback for small n
        initial_codewords = [list(w) for w in itertools.product(range(q), repeat=n)]
        
    # Convert to indices for internal processing
    codeword_indices = [word_to_idx(tuple(cw)) for cw in initial_codewords]
    
    # 3. Local Search with Numpy
    # State: set of codeword indices, coverage counts
    in_set = set(codeword_indices)
    coverage_counts = np.zeros(total_words, dtype=np.int32)
    for c in codeword_indices:
        coverage_counts[neighbors[c]] += 1
        
    best_size = len(codeword_indices)
    best_codewords = list(codeword_indices)
    
    # Local Search Function
    def local_search(indices, in_set, coverage_counts, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            # A codeword c is redundant if all words in its ball are covered >= 2 times
            redundant_indices = []
            for c in indices:
                if np.all(coverage_counts[neighbors[c]] >= 2):
                    redundant_indices.append(c)
            
            if redundant_indices:
                # Remove one (or all) redundant codewords
                # Removing one at a time and checking again is safer for coverage counts
                # But we can remove all currently redundant ones at once?
                # If c1 and c2 are both redundant, removing c1 might make c2 non-redundant?
                # No, if c1 is redundant, it means every word in Ball(c1) is covered by others.
                # Removing c1 decreases coverage of words in Ball(c1).
                # A word w in Ball(c2) might be in Ball(c1). If so, its coverage drops.
                # So c2 might become non-redundant.
                # So we should remove one, update, and restart.
                
                c_to_remove = redundant_indices[0]
                indices.remove(c_to_remove)
                in_set.discard(c_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                improved = True
                continue
            
            # 2. Try swaps if no removals possible
            # Find a codeword c and a candidate c' not in set
            # such that Critical(c) subset Ball(c')
            # Critical(c) = { w in Ball(c) | coverage_counts[w] == 1 }
            
            # To speed this up, we can precompute critical sets for current codewords
            # But doing it on the fly is O(K * BallSize) which is small.
            
            found_swap = False
            # Try a few random swaps to avoid getting stuck in a specific local minimum
            # Or try all? K is ~73, Candidates are 729-73=656. 73*656 ~ 50k checks.
            # Each check is set intersection.
            # Let's try a limited number of swaps per iteration to save time, or just one good one.
            
            # Identify codewords with small critical sets first (easier to swap)
            # Or just iterate.
            
            # Optimization: Precompute neighbor sets for fast subset check
            # neighbor_sets[i] = set(neighbors[i])
            # This is memory heavy? 729 * 13 elements. No, it's fine.
            
            if not hasattr(local_search, 'neighbor_sets'):
                local_search.neighbor_sets = [set(neighbors[i].tolist()) for i in range(total_words)]
            
            ns = local_search.neighbor_sets
            
            # Try to find a valid swap
            # We iterate over current codewords.
            # To avoid bias, we can shuffle the list of codewords.
            shuffled_indices = list(indices)
            random.shuffle(shuffled_indices)
            
            for c in shuffled_indices:
                if time.time() > deadline:
                    return False, indices, in_set, coverage_counts
                
                crit_mask = (coverage_counts[neighbors[c]] == 1)
                critical_words = neighbors[c][crit_mask]
                
                if len(critical_words) == 0:
                    # This shouldn't happen if we checked redundancy above, 
                    # but if it does, it's redundant.
                    indices.remove(c)
                    in_set.discard(c)
                    coverage_counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                # Find candidates c' not in in_set such that critical_words subset Ball(c')
                # We can iterate through all possible c'
                # To make it faster, we can just pick the first valid one found.
                # Or the one that covers the most *other* uncovered words? 
                # Since we are swapping, we don't care about uncovered words (there are none).
                # We just care about maintaining coverage.
                
                # A faster way:
                # For each w in critical_words, we know c' must be in Ball(w).
                # Intersection of Ball(w) for all w in critical_words, minus in_set.
                
                # Since critical_words is small (usually 1 or 2), we can just check candidates.
                # Let's just iterate through all non-set words? 656 iterations.
                # 73 codewords * 656 candidates = ~48k checks.
                # Each check: `set(critical_words).issubset(ns[c_prime])`
                # This is fast in Python.
                
                crit_set = set(critical_words.tolist())
                
                for c_prime in range(total_words):
                    if c_prime in in_set:
                        continue
                    if crit_set.issubset(ns[c_prime]):
                        # Valid swap
                        indices.remove(c)
                        in_set.discard(c)
                        coverage_counts[neighbors[c]] -= 1
                        
                        indices.append(c_prime)
                        in_set.add(c_prime)
                        coverage_counts[neighbors[c_prime]] += 1
                        
                        found_swap = True
                        break
                
                if found_swap:
                    improved = True
                    break
            
            if not found_swap and not improved:
                # No moves possible. Return to caller to decide on perturbation.
                break
                
        return improved, indices, in_set, coverage_counts

    # Iterated Local Search
    current_indices = list(codeword_indices)
    current_in_set = set(current_indices)
    current_coverage = coverage_counts.copy()
    
    # Run local search
    _, current_indices, current_in_set, current_coverage = local_search(
        current_indices, current_in_set, current_coverage, deadline
    )
    
    if len(current_indices) < best_size:
        best_size = len(current_indices)
        best_codewords = list(current_indices)
        
    # If we are still above 73 (known optimum), try perturbations
    # The problem size is small enough that we can do many restarts.
    
    while time.time() < deadline - 10:
        # Perturb: Remove a random codeword (making it invalid) and re-greedy cover?
        # Or swap a random codeword with a random other?
        # Let's try: Remove a random codeword, then add codewords to cover the newly uncovered words (Greedy Repair).
        
        trial_indices = list(best_codewords)
        trial_in_set = set(trial_indices)
        
        # Calculate coverage for best_codewords
        trial_coverage = np.zeros(total_words, dtype=np.int32)
        for c in trial_indices:
            trial_coverage[neighbors[c]] += 1
            
        # Pick a random codeword to remove
        if not trial_indices:
            break
        idx_to_remove = random.randint(0, len(trial_indices) - 1)
        c_removed = trial_indices.pop(idx_to_remove)
        trial_in_set.discard(c_removed)
        trial_coverage[neighbors[c_removed]] -= 1
        
        # Now some words are uncovered (coverage == 0).
        # Greedily add codewords to cover them.
        uncovered_mask = (trial_coverage == 0)
        if uncovered_mask.any():
            # Greedy set cover on the uncovered words
            # We want to minimize the number of added words.
            # Standard greedy: pick word covering most uncovered.
            while uncovered_mask.any() and time.time() < deadline:
                # Count how many uncovered words each candidate covers
                # candidates are all words not in trial_in_set
                # For efficiency, we can just check all 729 words.
                # counts = sum of uncovered_mask over neighbors
                counts = np.sum(uncovered_mask[neighbors], axis=1)
                
                # Exclude words already in set
                counts[list(trial_in_set)] = 0
                
                best_c = int(np.argmax(counts))
                if counts[best_c] == 0:
                    break # Should not happen if coverage is possible
                
                trial_indices.append(best_c)
                trial_in_set.add(best_c)
                trial_coverage[neighbors[best_c]] += 1
                uncovered_mask = (trial_coverage == 0)
                
        # Now we have a valid cover. Run local search on it.
        _, trial_indices, trial_in_set, trial_coverage = local_search(
            trial_indices, trial_in_set, trial_coverage, deadline
        )
        
        if len(trial_indices) < best_size:
            best_size = len(trial_indices)
            best_codewords = list(trial_indices)
            # If we hit the theoretical lower bound or known optimum, we can stop?
            # Known optimum for K_3(6,1) is 73. Lower bound 71.
            # If we reach 73, it's very good.
            if best_size <= 73:
                break

    # Convert best_codewords (indices) back to lists of integers
    result = []
    for idx in best_codewords:
        result.append(idx_to_word(idx))
        
    return result
