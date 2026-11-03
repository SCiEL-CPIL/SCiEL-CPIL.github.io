import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute index conversions
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors: neighbors_arr[i] contains indices of words within Hamming distance <= radius of word i
    # For radius 1, size is 1 + n*(q-1)
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    j = tuple_to_idx(tuple(new_w))
                    idxs.append(j)
        # Fill the array, padding with -1 if necessary (though for radius 1 it's exact)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx
        # If radius > 1, we'd need more, but task is radius 1. 
        # If idxs is shorter than cover_count (shouldn't be for radius 1), pad with -1
        if len(idxs) < cover_count:
             neighbors_arr[i, len(idxs):] = -1

    # Precompute reverse cover: for each word w, which codewords cover it?
    # reverse_cover_arr[w] contains indices of codewords c such that w is in neighbors_arr[c]
    # Since the graph is regular and symmetric, the number of codewords covering a word is also cover_count
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    # Initialize with -1 to handle potential padding if needed, though symmetry suggests exact fit
    reverse_cover_arr[:] = -1
    
    # We can fill this efficiently by iterating over all codewords and their neighbors
    for c in range(total_words):
        for neighbor_idx in neighbors_arr[c]:
            if neighbor_idx == -1:
                continue
            # We need to find the slot in reverse_cover_arr[neighbor_idx]
            # Since it's a regular graph, we can just append? No, numpy array is fixed size.
            # Better approach: use a list of lists first, then convert, or just count and fill.
            # Actually, for radius 1, each word is covered by exactly cover_count codewords.
            # We can fill it by iterating neighbors again.
            pass
            
    # Let's just build reverse_cover as a list of lists for simplicity and speed of construction, 
    # then convert to numpy if needed, or keep as lists if access is fast enough.
    # Given total_words=2187 and cover_count=14, lists are very fast.
    reverse_cover = [[] for _ in range(total_words)]
    for c in range(total_words):
        for neighbor_idx in neighbors_arr[c]:
            if neighbor_idx != -1:
                reverse_cover[neighbor_idx].append(c)
    
    # Convert to numpy arrays for faster intersection logic in local search if needed, 
    # but lists might be fine for small sizes. Let's stick to lists for reverse_cover 
    # as intersection of sets is easy.
    
    def greedy_construct_vectorized(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        
        while uncovered.any():
            # Vectorized gain computation: for each candidate c, count how many uncovered words it covers
            # uncovered[neighbors_arr] creates a 2D boolean array of shape (total_words, cover_count)
            # Summing along axis 1 gives the gain for each candidate
            gains = uncovered[neighbors_arr].sum(axis=1)
            
            valid = gains > 0
            if not valid.any():
                break
            
            max_gain = gains[valid].max()
            # Find all candidates with max gain
            best_candidates = np.where((gains == max_gain) & valid)[0]
            
            # Pick one randomly
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            # Update uncovered
            uncovered[neighbors_arr[chosen, :]] = False
            
        return code

    def reduce_code_fast(code):
        """Remove redundant codewords."""
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        changed = True
        while changed:
            changed = False
            new_code = []
            # Shuffle to randomize removal order
            code_iter = code[:]
            random.shuffle(code_iter)
            
            for c in code_iter:
                # Tentatively remove c
                counts[neighbors_arr[c, :]] -= 1
                
                if np.any(counts == 0):
                    # Cannot remove, put it back
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    # Can remove
                    changed = True
            
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """Perform swaps to enable further reductions."""
        code = list(code)
        if not code:
            return code
            
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() >= deadline:
                    break
                
                c_remove = code[i]
                
                # Find words uniquely covered by c_remove
                # These are words w where counts[w] == 1 AND w is in neighbors_arr[c_remove]
                # We need to be careful: counts currently includes c_remove.
                # So we look for w in neighbors_arr[c_remove] where counts[w] == 1.
                
                neighbors_c = neighbors_arr[c_remove, :]
                # Mask of neighbors that are uniquely covered by c_remove
                unique_mask = (counts[neighbors_c] == 1)
                unique_words = neighbors_c[unique_mask]
                
                if len(unique_words) == 0:
                    # c_remove is redundant, remove it
                    counts[neighbors_c] -= 1
                    code.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement c_new that covers all unique_words
                # c_new must be in the intersection of reverse_cover[w] for all w in unique_words
                # And c_new != c_remove
                # And c_new not already in code (to avoid duplicates, though logic handles it)
                
                if len(unique_words) > 4:
                    # Too many constraints, skip for efficiency
                    continue
                
                candidates = set()
                first = True
                for w in unique_words:
                    rc = set(reverse_cover[w])
                    if first:
                        candidates = rc
                        first = False
                    else:
                        candidates.intersection_update(rc)
                    
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                candidates.discard(c_remove)
                
                # Filter out current codewords? 
                # If c_new is already in code, swapping c_remove with c_new is a no-op or invalid.
                # Actually, if c_new is in code, it's already there. We are replacing c_remove with c_new.
                # If c_new is already in code, we would have a duplicate. 
                # The problem requires distinct codewords.
                # So c_new must NOT be in code.
                code_set = set(code)
                candidates -= code_set
                
                if not candidates:
                    continue
                
                # Score candidates: how many NEW words does c_new cover?
                # i.e., words in neighbors_arr[c_new] that are NOT in neighbors_arr[c_remove]
                # And are currently covered (counts > 0)?
                # Actually, we just want to maximize the "gain" of the swap in terms of 
                # potentially allowing future removals.
                # A good heuristic is to pick a candidate that covers many words 
                # that are currently covered by only 1 or 2 codewords (besides c_remove).
                
                best_candidate = None
                best_score = -1
                
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for c_new in cand_list:
                    # Calculate score
                    # We want c_new to cover unique_words (it does by construction).
                    # We also want it to cover other words that are "weakly" covered.
                    # Let's just pick the first one or one with high degree of overlap with weakly covered words.
                    # For speed, let's just pick randomly from a few top candidates.
                    
                    # Simple score: number of words in neighbors_arr[c_new] that have count <= 2
                    # (excluding c_remove's contribution, but counts includes it)
                    # When we remove c_remove, counts drop by 1 for its neighbors.
                    # So we look at counts[neighbors_c_new] after hypothetical removal.
                    
                    # Hypothetical counts: counts - indicator(neighbors_c_remove)
                    # We can't easily vectorize this for each candidate without a loop.
                    # Let's just use a simple heuristic: prefer candidates that cover many words 
                    # that are currently covered by exactly 2 codewords (one being c_remove).
                    
                    # For now, just pick the first valid candidate to keep it fast.
                    # Or pick one with max overlap with unique_words? They all cover all unique_words.
                    # Let's pick one that maximizes the number of neighbors w where counts[w] == 2.
                    # This suggests w is covered by c_remove and one other. If we replace c_remove with c_new,
                    # w is still covered. If c_new also covers other such words, it might help.
                    
                    # Let's just pick randomly from the first 10 candidates.
                    if best_score < 0:
                        best_candidate = c_new
                        best_score = 0
                    if random.random() < 0.1:
                        break # Early exit for speed
                
                if best_candidate is not None:
                    # Perform swap
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_candidate, :]] += 1
                    code[i] = best_candidate
                    improved = True
                    break
        
        return code

    # Main loop
    best_code = None
    best_size = float('inf')
    
    # First run a deterministic greedy + reduce to get a baseline
    # But greedy_construct_vectorized uses random tie-breaking. 
    # We can force deterministic by setting seed or modifying it.
    # For now, just run multiple random restarts.
    
    seed = 0
    while time.time() < deadline - 5.0:
        code = greedy_construct_vectorized(seed)
        code = reduce_code_fast(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we have time, try to improve with local search
            if time.time() < deadline - 10.0:
                improved_code = local_search_improve(code, deadline - 5.0)
                improved_code = reduce_code_fast(improved_code)
                
                if len(improved_code) < best_size:
                    best_size = len(improved_code)
                    best_code = improved_code[:]
        
        seed += 1
        
        # Optional: try a different strategy every few iterations
        # e.g., perturb the best code and repair
        if seed % 10 == 0 and best_code is not None and time.time() < deadline - 20.0:
            # Perturb best_code
            perturbed = best_code[:]
            n_remove = min(5, len(perturbed) // 4)
            if n_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), n_remove)
                for ri in sorted(remove_indices, reverse=True):
                    perturbed.pop(ri)
                
                # Repair: greedily add codewords to cover uncovered words
                # This is essentially running greedy on the uncovered set
                # But we can just run greedy_construct_vectorized and merge?
                # Simpler: run local search on the perturbed code, but it needs to be valid.
                # Perturbed code is invalid. We need to repair it.
                
                # Repair step:
                # Count coverage
                counts = np.zeros(total_words, dtype=np.int16)
                for c in perturbed:
                    counts[neighbors_arr[c, :]] += 1
                
                uncovered = np.where(counts == 0)[0]
                
                while len(uncovered) > 0:
                    # Greedily pick a codeword that covers max uncovered
                    # Only consider codewords not in perturbed
                    # This is expensive. 
                    # Instead, let's just discard this perturbation strategy if it's too complex.
                    # Stick to full restarts.
                    break
                
                # If we implemented repair, we would add codewords here.
                # Since we didn't, we skip this branch.
    
    # Final polish on the best code found
    if best_code is not None and time.time() < deadline - 2.0:
        polished = local_search_improve(best_code, deadline - 1.0)
        polished = reduce_code_fast(polished)
        if len(polished) < len(best_code):
            best_code = polished

    # Convert indices to words
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    
    return result

# EVOLVE-BLOCK-END
