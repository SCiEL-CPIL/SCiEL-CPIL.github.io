"""Bootstrap: greedy set-cover for ternary covering code K_3(8,1)."""
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback: direct sum with tetracode for general case
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback: brute force all words
    import itertools
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute all words as integers
    all_words = np.arange(total, dtype=np.int32)
    
    # Precompute the "ball" of radius 1 around each word (as a set of word indices)
    # Each word covers itself and words differing in 1 position
    # For word w, convert to base-3 digits, then for each position, change to 0,1,2
    
    # Convert integer to base-3 digits
    def to_digits(w, n=8):
        digits = np.zeros(n, dtype=np.int8)
        for i in range(n - 1, -1, -1):
            digits[i] = w % 3
            w //= 3
        return digits
    
    # Precompute coverage for all candidates (all 6561 words are candidates)
    # coverage[i] = set/array of word indices covered by candidate i
    # This is expensive to precompute all, so we'll compute on the fly or in chunks
    
    # Better: precompute for each word, the 17 words it covers (as an array)
    # Use vectorized approach
    
    # For each word index, compute its base-3 representation
    # Then for each position, generate the 2 alternative values
    
    # Let's precompute all covered indices for all 6561 candidates
    # This is 6561 * 17 = ~111537 entries, manageable
    
    covered_by = np.zeros((total, 17), dtype=np.int32)
    
    # Vectorized: for word w, digits are [w//3^7, w//3^6 % 3, ..., w%3]
    # Position i from left has weight 3^(n-1-i)
    
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # For each word, compute the 17 covered words
    # Word itself + for each position, word + delta where delta changes that position
    
    # This is complex to fully vectorize, so let's do it in a loop but efficiently
    for w in range(total):
        digits = to_digits(w, n)
        covered_by[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            current_digit = digits[pos]
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                new_w = w + (new_digit - current_digit) * weight
                covered_by[w, idx] = new_w
                idx += 1
    
    # Now greedy set cover
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    selected = []
    candidate_indices = np.arange(total)
    
    while uncovered_count > 0 and time.time() < deadline:
        # For each remaining candidate, count how many uncovered words it covers
        # covered_by[candidate] gives 17 word indices
        # We need to count how many of those are in uncovered
        
        # Vectorized: for each candidate, check if covered_by[candidate] is in uncovered
        # covered_by shape: (total, 17), uncovered shape: (total,)
        
        # Get the covered indices for all candidates
        # covered_by shape (6561, 17), flatten and use advanced indexing
        cov_idx = covered_by  # (6561, 17)
        
        # For each candidate c, count uncovered in its coverage
        # uncovered[cov_idx[c, :]] gives (17,) bools, sum gives count
        # Do this for all candidates at once: uncovered[cov_idx] gives (6561, 17)
        cov_uncovered = uncovered[cov_idx]  # (6561, 17)
        coverage_counts = cov_uncovered.sum(axis=1)  # (6561,)
        
        # Pick the candidate with max coverage count
        best_idx = np.argmax(coverage_counts)
        best_count = coverage_counts[best_idx]
        
        if best_count == 0:
            # No candidate covers any uncovered word - shouldn't happen
            break
        
        selected.append(best_idx)
        
        # Mark covered words as uncovered
        new_cov = covered_by[best_idx]
        uncovered[new_cov] = False
        uncovered_count -= best_count
        
        # Remove best candidate from consideration (optional, but it's now covered)
        coverage_counts[best_idx] = -1  # exclude from future selection
    
    if time.time() < deadline and uncovered_count > 0:
        # Try to fix any remaining uncovered by adding their own codewords
        remaining_uncovered = np.where(uncovered)[0]
        for w in remaining_uncovered:
            if time.time() > deadline - 2.0:
                break
            selected.append(w)
            new_cov = covered_by[w]
            uncovered[new_cov] = False
            uncovered_count -= best_count if False else 0
    
    # Local search: try to remove redundant codewords
    # For each codeword, check if removing it still leaves all words covered
    # If so, remove it
    if time.time() < deadline - 5.0:
        # Build coverage count array: how many selected codewords cover each word
        cov_count = np.zeros(total, dtype=np.int32)
        for w in selected:
            cov_count[covered_by[w]] += 1
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try removing each codeword (iterate in reverse for stability)
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                w = selected[i]
                # Check if removing w leaves any word with cov_count == 0
                # Only words covered by w need checking
                cov_words = covered_by[w]
                # If any of these words has cov_count == 1, removing w would uncover it
                if np.any(cov_count[cov_words] == 1):
                    continue
                # Safe to remove
                cov_count[cov_words] -= 1
                del selected[i]
                improved = True
                break  # Restart scan after removal
    
    # Convert selected word indices to lists of digits
    result = []
    for w in selected:
        digits = to_digits(w, n).tolist()
        result.append(digits)
    
    return result
