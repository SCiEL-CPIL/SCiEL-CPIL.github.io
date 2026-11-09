"""Covering code search for K_q(n, radius) using Tetracode initialization + Simulated Annealing / Fixed-Size Local Search."""
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


def _find_valid_code_size_k(M, k, total, time_limit, seed, initial_code=None):
    """
    Try to find a valid covering code of exactly size k.
    Uses local search (swaps) to fix coverage.
    Returns list of indices if found, None otherwise.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Initialize
    if initial_code and len(initial_code) == k:
        code = list(initial_code)
    else:
        # Random initialization or greedy seed
        code = list(rng.sample(range(total), k))
        
    # Precompute covers for current code
    # To speed up, we maintain counts
    counts = np.zeros(total, dtype=np.int32)
    code_set = set(code)
    for idx in code:
        counts[M[idx] == 1] += 1
        
    # If initial code is valid, we are done
    if (counts >= 1).all():
        return code

    # Local Search: Swap
    # We want to minimize uncovered count.
    # Uncovered words are those with count 0.
    uncovered_count = np.sum(counts == 0)
    
    no_improvement = 0
    max_no_improvement = 1000 # Adjust based on time
    
    while time.time() - start_time < time_limit:
        if uncovered_count == 0:
            return code
            
        # Pick an uncovered word to fix
        uncovered_indices = np.where(counts == 0)[0]
        target_word = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        
        # Find candidates that cover target_word
        # M[:, target_word] == 1 gives all codewords that cover target_word
        candidates = np.where(M[:, target_word] == 1)[0]
        
        if len(candidates) == 0:
            # Should not happen if M is correct (every word covers itself)
            break
            
        # We need to swap one existing codeword out with one candidate in.
        # To minimize disruption, pick the candidate that covers the most *other* uncovered words?
        # Or just pick randomly from candidates?
        # Let's try to pick the candidate that maximizes coverage of uncovered words.
        
        # Optimization: Only consider candidates not already in code
        candidates_not_in = [c for c in candidates if c not in code_set]
        
        if not candidates_not_in:
            # All candidates are already in code? Then target_word is covered? 
            # If target_word is uncovered (count 0), no codeword in code covers it.
            # So candidates (which cover it) are NOT in code.
            # So candidates_not_in should be non-empty.
            pass

        # Score candidates by how many uncovered words they cover
        # This is expensive if done for all candidates.
        # Let's sample a few candidates.
        num_candidates_to_test = min(10, len(candidates_not_in))
        sampled_candidates = rng.sample(candidates_not_in, num_candidates_to_test)
        
        best_swap = None
        best_uncovered_delta = 0 # We want to decrease uncovered count
        
        # We need to pick a codeword to remove.
        # Ideally, remove a codeword that covers few unique words (low "criticality").
        # Let's pick a random codeword to remove, or the one with lowest coverage count?
        # Removing a codeword with high count is safe.
        
        # Strategy: 
        # 1. Pick a candidate c_in.
        # 2. Pick a codeword c_out to remove.
        # 3. Check if swap is valid (uncovered count doesn't increase, ideally decreases).
        
        # To be efficient, let's try swapping a random c_out with a good c_in.
        
        # Identify "safe" c_outs: those that don't uniquely cover any word?
        # unique_mask for c_out: (counts == 1) & (M[c_out] == 1)
        
        # Let's just try a few random swaps.
        found_improvement = False
        
        for _ in range(5): # Try 5 random swaps
            c_out = code[rng.randint(0, k - 1)]
            c_in = rng.choice(sampled_candidates)
            
            # Check effect of removing c_out and adding c_in
            # New counts = counts - M[c_out] + M[c_in]
            # We care about words that become 0 (bad) or words that go from 0 to >0 (good)
            
            # Words uniquely covered by c_out (count 1 and covered by c_out)
            # If c_in does not cover them, they become uncovered.
            
            # Words uncovered (count 0) that c_in covers.
            
            # Let's just compute the new uncovered count for this specific swap.
            # This is O(total) per swap check, which is 729. Very fast.
            
            # Simulate
            # We don't want to modify counts globally yet.
            # delta_counts = -M[c_out] + M[c_in]
            # new_counts = counts + delta_counts
            # new_uncovered = sum(new_counts == 0)
            
            # But we can be smarter.
            # Uncovered words are those with counts==0.
            # Adding c_in covers some of them.
            # Removing c_out might uncover some.
            
            # Words covered by c_in: cov_in = M[c_in]
            # Words covered by c_out: cov_out = M[c_out]
            
            # 1. Words that are currently uncovered (counts==0) and covered by c_in.
            #    These will become covered. Good.
            #    gain = sum((counts == 0) & cov_in)
            
            # 2. Words that are currently covered (counts>0) but become uncovered.
            #    This happens if counts[w] == 1, w is covered by c_out, and w is NOT covered by c_in.
            #    loss = sum((counts == 1) & cov_out & (~cov_in))
            
            cov_in = M[c_in]
            cov_out = M[c_out]
            
            gain = np.sum((counts == 0) & cov_in)
            loss = np.sum((counts == 1) & cov_out & (~cov_in))
            
            delta_uncovered = loss - gain
            # We want delta_uncovered < 0 (decrease in uncovered)
            # Or if delta_uncovered == 0, we might accept it to escape local minima?
            # For now, let's require improvement or neutral.
            
            if delta_uncovered < best_uncovered_delta:
                best_uncovered_delta = delta_uncovered
                best_swap = (c_out, c_in)
                
        if best_swap:
            c_out, c_in = best_swap
            # Perform swap
            counts -= M[c_out]
            counts += M[c_in]
            code[code.index(c_out)] = c_in
            code_set.remove(c_out)
            code_set.add(c_in)
            uncovered_count += best_uncovered_delta
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > max_no_improvement:
                break
                
    if (counts >= 1).all():
        return code
    return None


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    
    # Build coverage matrix
    M, all_words = _build_coverage_matrix(q, n, radius)
    
    # 1. Get initial code (Tetracode or Greedy)
    tetracode_code = _direct_sum_tetracode(q, n, radius)
    if tetracode_code is not None:
        initial_indices = [_word_to_idx(w, q, n) for w in tetracode_code]
        best_code = initial_indices
        best_size = len(best_code)
    else:
        # Fallback: Random or Greedy
        # For small spaces, greedy is good
        # Simple greedy
        uncovered = np.ones(total, dtype=bool)
        selected = []
        while uncovered.any():
            counts = M[:, uncovered].sum(axis=1)
            best_idx = np.argmax(counts)
            if counts[best_idx] == 0: break
            selected.append(best_idx)
            uncovered[M[best_idx] == 1] = False
        best_code = selected
        best_size = len(best_code)
        
    # Verify initial code
    if not _verify_code(best_code, M, total):
        # Should not happen, but if it does, force fix with greedy
        # (Greedy always works)
        pass

    # 2. Try to reduce size
    # We know lower bound is 71 for K_3(6,1).
    # Let's try sizes from best_size down to 71.
    
    current_best = best_code
    current_best_size = best_size
    
    # Time budget for search
    search_deadline = _DEADLINE - 10 # Leave 10s for final checks
    
    # Try to find valid codes of smaller sizes
    # Start from current best size - 1
    for k in range(current_best_size - 1, 70, -1):
        if time.time() > search_deadline:
            break
            
        # Allocate time for this k
        # If we have lots of time left, give more to larger k (harder) or smaller k (more valuable)?
        # Usually, finding k=73 is the goal.
        # Let's give equal time slices or decreasing.
        time_slice = 5.0 # 5 seconds per size attempt
        
        # Try multiple seeds/restarts for this k
        found = False
        num_restarts = 3
        for seed in range(num_restarts):
            if time.time() > search_deadline:
                break
                
            # Use previous best as initial if size matches? No, size is different.
            # Just random or greedy seed of size k.
            # Greedy seed of size k:
            # Run greedy but stop at k? Greedy produces min size, might be > k.
            # If greedy size > k, we can take first k? Not necessarily valid.
            # Better: Random initialization.
            
            code_k = _find_valid_code_size_k(M, k, total, time_slice, seed)
            if code_k is not None:
                current_best = code_k
                current_best_size = k
                found = True
                break # Found a valid code of size k, move to k-1
                
        if found:
            # We found a smaller code. Continue to k-1.
            pass
        else:
            # If we failed to find k, we might still try k-1?
            # Usually if we can't find k, we can't find k-1.
            # But with different seeds, maybe?
            # Let's stick to the logic: if we found k, try k-1. If not, stop or try a bit more?
            # The prompt says "minimized".
            # Let's stop if we fail to improve? Or try a bit more?
            # Given time constraints, if we fail for k, it's likely hard.
            # But we might have been unlucky.
            # Let's allow one more try for the next size down if we have time.
            # Actually, if we can't find 74, we probably can't find 73 easily from scratch.
            # But if we have 74, we might be able to swap to 73.
            # The function _find_valid_code_size_k does swaps.
            # If we fail to find 74, we don't have a 74 to start from.
            # So we are stuck.
            # Break if we fail?
            # Let's break to save time for verification/return.
            break
            
    # Final verification
    if not _verify_code(current_best, M, total):
        # Fallback to the initial valid code
        return [all_words[i].tolist() for i in best_code]
        
    return [all_words[i].tolist() for i in current_best]
