import numpy as np
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # 1. Precomputation
    total_words = q ** n
    # Map word (tuple) to index
    # Index to word (tuple)
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (radius 1)
    # neighbors[i] is a set of indices
    neighbors = [None] * total_words
    for i, w in enumerate(words):
        neigh = set()
        neigh.add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val: continue
                # Create new word
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                neigh.add(word_to_idx[new_w])
        neighbors[i] = neigh

    # 2. Greedy Construction
    def greedy_build(random_seed=0):
        import random
        random.seed(random_seed)
        covered_count = np.zeros(total_words, dtype=int)
        code = []
        for _ in range(total_words): # Max iterations
            # Find word with max new coverage
            best_idx = -1
            max_new = 0
            # To speed up, we can just iterate all
            # But 2187 * 2187 is ~5M ops, which is fine in Python if optimized?
            # Better: maintain a list of uncovered words.
            
            # Actually, simpler:
            # score[i] = number of j in neighbors[i] where covered_count[j] == 0
            # We need to update this efficiently?
            # For N=2187, O(N^2) per step is 4.7M ops. 2187 steps -> 10^10 ops. Too slow in Python.
            
            # We need a faster way to find the best next word.
            # Maintain `uncovered_set` (set of indices).
            # For each candidate c, count how many of its neighbors are in `uncovered_set`.
            # This is O(N * |neighbors|) = 2187 * 29 ~ 63k ops per step.
            # 2187 steps -> 10^8 ops. This might be okay in 140s?
            # 10^8 simple Python operations might take 10-50s.
            # Let's try to optimize.
            
            uncovered = set(range(total_words))
            code = []
            
            while uncovered:
                best_idx = -1
                best_count = 0
                # Iterate through all potential candidates
                # We can prune: only consider candidates that are in uncovered?
                # No, a codeword can cover uncovered words even if it's already covered.
                # But adding a covered word is usually not optimal unless it covers many uncovered.
                
                for i in range(total_words):
                    # Count neighbors in uncovered
                    cnt = 0
                    for j in neighbors[i]:
                        if j in uncovered:
                            cnt += 1
                            if cnt > best_count:
                                # Early exit? No, we need exact max to be safe, 
                                # but if we just want *a* good one, we can stop early?
                                # Greedy usually needs the *best*.
                                pass
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        # Random tie-breaking
                        # If cnt == best_count, we can randomize later.
                        # For now, keep first max.
                        
                if best_idx == -1:
                    break
                    
                code.append(best_idx)
                # Update uncovered
                for j in neighbors[best_idx]:
                    if j in uncovered:
                        uncovered.remove(j)
                        
                # Check time
                if time.time() > deadline - 20:
                    break
                    
        return code

    # 3. Local Search
    def local_search(code):
        code_set = set(code)
        covered_count = np.zeros(total_words, dtype=int)
        for i in code:
            for j in neighbors[i]:
                covered_count[j] += 1
        
        # Helper to get uncovered if we remove i
        def get_uncovered_if_remove(i):
            unc = []
            for j in neighbors[i]:
                if covered_count[j] == 1: # Only covered by i
                    unc.append(j)
            return unc

        # Helper to find best replacement for a set of uncovered words
        def find_best_replacement(unc_set):
            if not unc_set:
                return -1
            best_idx = -1
            best_cov = 0
            unc_list = list(unc_set)
            for i in range(total_words):
                cnt = 0
                for j in neighbors[i]:
                    if j in unc_set:
                        cnt += 1
                if cnt > best_cov:
                    best_cov = cnt
                    best_idx = i
            # Check if it covers all
            if best_idx != -1:
                # Verify coverage
                # Actually, we can just return best_idx and let caller check
                pass
            return best_idx

        # Main loop
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try removals
            for i in code:
                unc = get_uncovered_if_remove(i)
                if not unc:
                    # Remove i
                    code.remove(i)
                    # Update covered_count
                    for j in neighbors[i]:
                        covered_count[j] -= 1
                    improved = True
                    break # Restart loop after change
            
            if not improved:
                # Try swaps (size preserving) to escape local minima
                # Pick a random codeword to try swapping
                # Or iterate through all?
                # Let's try a few random swaps
                swaps_made = 0
                for _ in range(10): # Limit swaps per iteration to avoid infinite loop
                    if not code: break
                    i = random.choice(code)
                    unc = get_uncovered_if_remove(i)
                    if not unc:
                        # This shouldn't happen if we checked removals above
                        # But if it does, remove it
                        code.remove(i)
                        for j in neighbors[i]:
                            covered_count[j] -= 1
                        improved = True
                        break
                    
                    # Find best j covering unc
                    # We want j to cover all of unc
                    # Find j that covers max unc
                    best_j = -1
                    max_cov = 0
                    for j in range(total_words):
                        if j == i: continue
                        cnt = 0
                        for k in neighbors[j]:
                            if k in unc:
                                cnt += 1
                        if cnt > max_cov:
                            max_cov = cnt
                            best_j = j
                    
                    if best_j != -1 and max_cov == len(unc):
                        # Swap i -> best_j
                        # Remove i
                        code.remove(i)
                        for j in neighbors[i]:
                            covered_count[j] -= 1
                        # Add best_j
                        code.append(best_j)
                        for j in neighbors[best_j]:
                            covered_count[j] += 1
                        swaps_made += 1
                        # We don't set improved=True here because size didn't change
                        # But we changed the state.
                        # We break to re-evaluate removals.
                        break
                
                if swaps_made == 0:
                    # Stuck?
                    # Try a random restart? Or just stop?
                    # Given the small size, we might just stop.
                    # Or we can try to add a word and remove two? (3-swap)
                    # Too complex.
                    # Let's break.
                    break
        
        return code

    # 4. Main Execution
    best_code = None
    best_size = float('inf')
    
    # Run a few iterations with different random seeds for Greedy
    for seed in range(10):
        if time.time() > deadline - 30:
            break
            
        code = greedy_build(seed)
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    # Convert back to lists of integers
    result = [list(words[i]) for i in best_code]
    return result
