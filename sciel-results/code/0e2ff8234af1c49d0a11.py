import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius, start_time, deadline)
    
    # Specific optimization for q=3, n=6, r=1
    total_words = 3 ** n  # 729
    
    # Helper functions
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * 3 + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)
    
    # Precompute all neighbor indices for each word using bit masks
    # Each word covers itself + 12 neighbors = 13 words
    # Represent coverage as Python int with 729 bits
    all_mask = (1 << total_words) - 1
    
    # Precompute coverage mask for each word
    cov_masks = [0] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        mask = 1 << idx  # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                nb_idx = word_to_idx(tuple(new_w_list))
                mask |= (1 << nb_idx)
        cov_masks[idx] = mask
    
    # Phase 1: Greedy set cover with bit-parallel operations
    # Start with empty cover, add words that cover most uncovered words
    uncovered = all_mask
    greedy_code = []
    greedy_cov = 0
    
    # Pre-sort candidates by coverage size (all same: 13), use random tie-breaking
    candidates = list(range(total_words))
    random.shuffle(candidates)
    
    while uncovered != 0 and time.time() < deadline - 60.0:
        best_cand = -1
        best_gain = 0
        best_mask = 0
        
        # Sample candidates for speed (check up to 200 per iteration)
        sample_size = min(len(candidates), 200)
        sampled = random.sample(candidates, sample_size)
        
        for cand in sampled:
            gain = (cov_masks[cand] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
                best_mask = cov_masks[cand]
        
        if best_cand == -1 or best_gain == 0:
            break
            
        greedy_code.append(best_cand)
        greedy_cov |= best_mask
        uncovered &= ~best_mask
        candidates.remove(best_cand)
    
    # Phase 2: Local search - try to remove redundant codewords
    current_code = list(greedy_code)
    
    # Build coverage count using bit operations
    def compute_coverage_mask(code):
        cov = 0
        for c in code:
            cov |= cov_masks[c]
        return cov
    
    # Redundancy removal loop
    improved = True
    while improved and time.time() < deadline - 50.0:
        improved = False
        new_code = []
        for i, cw in enumerate(current_code):
            if time.time() > deadline - 50.0:
                new_code.append(cw)
                continue
            # Check if removing this codeword still covers everything
            remaining = [c for j, c in enumerate(current_code) if j != i]
            if len(remaining) == 0:
                new_code.append(cw)
                continue
            rem_cov = compute_coverage_mask(remaining)
            if rem_cov == all_mask:
                improved = True
            else:
                new_code.append(cw)
        current_code = new_code
    
    # Phase 3: Branch and bound with bit-parallel set cover
    # Try to find a smaller cover using B&B
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Sort codewords by coverage overlap with uncovered (for B&B ordering)
    # Use a heuristic: start from the greedy solution and try to improve
    
    # B&B function using bit masks
    def branch_and_bound(uncovered_mask, remaining_candidates, current_size, best_size_ref):
        """
        Returns True if a better solution was found, updates best_code and best_size.
        """
        nonlocal best_code, best_size
        
        if time.time() > deadline - 30.0:
            return False
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_code = list(current_code_bnb)
            return True
        
        # Lower bound: need at least ceil(popcount(uncovered) / 13) more codewords
        min_needed = (uncovered_mask.bit_count() + 12) // 13
        if current_size + min_needed >= best_size:
            return False
        
        if not remaining_candidates:
            return False
        
        # Heuristic: pick the candidate that covers the most uncovered words
        best_cand = -1
        best_gain = 0
        for cand in remaining_candidates:
            gain = (cov_masks[cand] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        
        if best_cand == -1 or best_gain == 0:
            return False
        
        # Branch on best_cand: include or exclude
        # Include branch first (heuristic)
        new_uncovered = uncovered_mask & ~cov_masks[best_cand]
        new_remaining = [c for c in remaining_candidates if c != best_cand]
        current_code_bnb.append(best_cand)
        
        result = branch_and_bound(new_uncovered, new_remaining, current_size + 1, best_size_ref)
        current_code_bnb.pop()
        
        if result:
            return True
        
        # Exclude branch
        new_remaining = [c for c in remaining_candidates if c != best_cand]
        result = branch_and_bound(uncovered_mask, new_remaining, current_size, best_size_ref)
        
        return result
    
    # Try B&B starting from current best
    current_code_bnb = []
    # Use the current code as initial solution
    for _ in range(3):  # Try a few times with different orderings
        if time.time() > deadline - 30.0:
            break
        
        # Randomize candidate order for diversity
        candidates_bnb = list(range(total_words))
        random.shuffle(candidates_bnb)
        
        current_code_bnb = []
        branch_and_bound(all_mask, candidates_bnb, 0, best_size)
    
    # Phase 4: Final local search on the best found code
    final_code = list(best_code)
    
    # More aggressive local search: try single swaps
    for iteration in range(20):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        # Try removing each codeword
        for i in range(len(final_code)):
            if time.time() > deadline - 10.0:
                break
            
            rem_code = final_code[:i] + final_code[i+1:]
            rem_cov = compute_coverage_mask(rem_code)
            if rem_cov == all_mask:
                final_code = rem_code
                improved = True
                break
        
        if not improved:
            # Try swaps: remove one, add one
            for i in range(len(final_code)):
                if time.time() > deadline - 10.0:
                    break
                
                rem_code = final_code[:i] + final_code[i+1:]
                rem_cov = compute_coverage_mask(rem_code)
                uncovered_after_removal = all_mask & ~rem_cov
                
                if uncovered_after_removal == 0:
                    final_code = rem_code
                    improved = True
                    break
                
                # Find a replacement that covers the uncovered words
                for cand in range(total_words):
                    if cand in rem_code:
                        continue
                    if (cov_masks[cand] & uncovered_after_removal) == uncovered_after_removal:
                        final_code = rem_code + [cand]
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    # Verify final coverage
    final_cov = compute_coverage_mask(final_code)
    if final_cov != all_mask:
        # Fallback to greedy code if coverage is broken
        final_code = list(greedy_code)
    
    # Convert back to list of lists
    return [list(idx_to_word(cw)) for cw in final_code]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback implementation for other parameters
    total_words = q ** n
    if total_words > 10000:
        return []
    
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors
    neighbors = {}
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbs = {idx}
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = new_val
                nbs.add(word_to_idx(tuple(nw)))
        neighbors[idx] = nbs
    
    # Greedy set cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_cand = -1
        best_gain = 0
        for cand in list(uncovered) + list(random.sample(range(total_words), min(100, total_words))):
            gain = len(neighbors[cand] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand == -1 or best_gain == 0:
            break
        code.append(best_cand)
        uncovered -= neighbors[best_cand]
    
    return [list(idx_to_word(cw)) for cw in code]
