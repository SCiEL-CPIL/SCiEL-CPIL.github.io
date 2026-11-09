"""Covering code search for K_q(n, radius) using Tetracode initialization + intensive local search."""
import time
import numpy as np
import random
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _build_coverage_matrix(q, n, radius):
    """
    Build a coverage matrix M of shape (total, total) where
    M[i, j] = 1 if word i covers word j (i.e., hamming(word_i, word_j) <= radius).
    """
    total = q ** n
    all_words = np.array([_idx_to_word(i, n, q) for i in range(total)], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    
    M = (dists <= radius).astype(np.int8)
    return M, all_words


def _direct_sum_tetracode(q, n, radius):
    """
    Direct sum of ternary tetracode (9 words of length 4) with all tails.
    This provides a valid covering code of size 9 * q^(n-4).
    For q=3, n=6, size is 9 * 3^2 = 81.
    """
    if q != 3 or n < 4:
        return None
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    if n == 4:
        return [list(h) for h in tetracode]
    
    tails = list(itertools.product(range(q), repeat=n - 4))
    return [h + list(tail) for h in tetracode for tail in tails]


def _verify_code(code_indices, M, total):
    """Verify that every word is covered."""
    if len(code_indices) == 0:
        return total == 0
    selected = np.zeros(total, dtype=bool)
    selected[code_indices] = True
    # Check if M @ selected >= 1 for all rows
    covered = (M @ selected) >= 1
    return covered.all()


def _greedy_construct(M, all_words, total):
    """
    Greedy set cover: at each step, pick the word that covers the most uncovered words.
    """
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        # Count how many uncovered words each candidate covers
        counts = M[:, uncovered].sum(axis=1)
        # Pick the one with max count
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[M[best_idx] == 1] = False
    
    return selected


def _local_search_polish(code_indices, M, all_words, total, time_limit=60.0, seed=42):
    """
    Intensive local search: greedy deletion followed by targeted swaps.
    """
    code_indices = list(code_indices)
    start_time = time.time()
    rng = random.Random(seed)
    
    # Precompute covers as boolean arrays for speed
    covers = [M[idx] == 1 for idx in code_indices]
    
    # Maintain counts: how many codewords cover each word
    counts = np.zeros(total, dtype=np.int32)
    for cov in covers:
        counts[cov] += 1
        
    code_set = set(code_indices)
    
    def try_deletion():
        """Attempt to remove redundant codewords. Returns True if any removed."""
        nonlocal code_indices, covers, counts, code_set
        changed = False
        # Try multiple passes
        while True:
            any_removed = False
            # Iterate backwards to allow popping
            for i in range(len(code_indices) - 1, -1, -1):
                cov = covers[i]
                # If any word in this codeword's ball is covered ONLY by this codeword, we can't remove it
                if (counts[cov] == 1).any():
                    continue
                # Remove it
                counts[cov] -= 1
                removed_idx = code_indices.pop(i)
                covers.pop(i)
                code_set.remove(removed_idx)
                changed = True
                any_removed = True
            if not any_removed:
                break
        return changed

    # Initial greedy deletion
    while try_deletion():
        pass
        
    best_size = len(code_indices)
    no_improvement_count = 0
    
    while time.time() - start_time < time_limit:
        if len(code_indices) == 0:
            break
            
        # Strategy: Pick a codeword to try removing, find a good replacement
        i = rng.randint(0, len(code_indices) - 1)
        idx_out = code_indices[i]
        cov_out = covers[i]
        
        # Find words that are uniquely covered by idx_out
        unique_mask = (counts == 1) & cov_out
        unique_words = np.where(unique_mask)[0]
        
        if len(unique_words) == 0:
            # Can just delete
            counts[cov_out] -= 1
            code_indices.pop(i)
            covers.pop(i)
            code_set.remove(idx_out)
            best_size = len(code_indices)
            no_improvement_count = 0
            # Try more deletions
            try_deletion()
            continue
        
        # Need to find a replacement that covers all unique_words
        # Candidate: words that are within distance radius of all unique_words
        # and are not already in the code
        # For efficiency, sample candidates
        
        found_swap = False
        for _ in range(50):  # Try up to 50 random candidates
            # Bias towards words near the unique words
            if len(unique_words) > 0:
                # Pick a random unique word and find a random word near it
                uw = unique_words[rng.randint(0, len(unique_words) - 1)]
                # Generate a random word within distance 1 of uw
                uw_word = all_words[uw]
                new_word = uw_word.copy()
                if rng.random() < 0.5:
                    pos = rng.randint(0, len(new_word) - 1)
                    new_word[pos] = (new_word[pos] + 1) % q
                else:
                    # Random word
                    new_word = np.array([rng.randint(0, q - 1) for _ in range(n)], dtype=np.int8)
                
                idx_in = _word_to_idx(new_word, q, n)
            else:
                idx_in = rng.randint(0, total - 1)
            
            if idx_in in code_set:
                continue
            
            cov_in = M[idx_in] == 1
            
            # Check if swap is valid: for all words in cov_out, counts[w] + cov_in[w] >= 2
            # i.e., counts[w] >= 2 - cov_in[w], which means for w in cov_out where counts[w]==1, we need cov_in[w]==1
            risky = cov_out & (counts == 1)
            if not np.all(cov_in[risky]):
                continue
            
            # Also need: for words NOT in cov_out but in cov_in, they must already be covered
            # (they are, since counts >= 1 everywhere initially, and we only maintain that invariant)
            
            # Swap is valid. Perform it.
            counts[cov_out] -= 1
            counts[cov_in] += 1
            code_indices[i] = idx_in
            covers[i] = cov_in
            code_set.remove(idx_out)
            code_set.add(idx_in)
            found_swap = True
            
            # Try to delete redundant words after the swap
            deleted_any = try_deletion()
            
            if deleted_any:
                best_size = len(code_indices)
                no_improvement_count = 0
            else:
                # No immediate improvement. Revert to stay close to good solutions
                counts[cov_in] -= 1
                counts[cov_out] += 1
                code_indices[i] = idx_out
                covers[i] = cov_out
                code_set.remove(idx_in)
                code_set.add(idx_out)
                no_improvement_count += 1
            
            break  # Only try one swap per iteration
        
        if not found_swap:
            no_improvement_count += 1
            
        if no_improvement_count > 200:
            # Reset counter and continue
            no_improvement_count = 0
            
    return code_indices


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    
    # Build coverage matrix
    M, all_words = _build_coverage_matrix(q, n, radius)
    
    # 1. Try Tetracode initialization if applicable
    tetracode_code = _direct_sum_tetracode(q, n, radius)
    if tetracode_code is not None:
        # Convert words to indices
        tetracode_indices = []
        for w in tetracode_code:
            idx = _word_to_idx(w, q, n)
            tetracode_indices.append(idx)
        best_code = tetracode_indices
        best_size = len(best_code)
    else:
        # Fallback to greedy if tetracode not applicable
        greedy_code = _greedy_construct(M, all_words, total)
        best_code = greedy_code
        best_size = len(greedy_code)
    
    # 2. Intensive local search polish with multiple seeds
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 15  # Leave 15s margin for safety
    
    if remaining > 10:
        # Try multiple seeds
        num_restarts = min(5, int(remaining / 20) + 1)
        for seed in range(num_restarts):
            if time.time() - start_time > _DEADLINE - 15:
                break
            t_left = _DEADLINE - time.time() - 15
            if t_left < 5:
                break
            polished_code = _local_search_polish(best_code, M, all_words, total, time_limit=t_left, seed=42 + seed)
            if len(polished_code) < best_size:
                best_code = polished_code
                best_size = len(best_code)
    
    # Convert indices back to words
    best_code_words = [all_words[i].tolist() for i in best_code]
    
    # Final verification
    if not _verify_code(best_code, M, total):
        # Fallback: use a valid greedy solution
        greedy_code = _greedy_construct(M, all_words, total)
        return [all_words[i].tolist() for i in greedy_code]
    
    return best_code_words
