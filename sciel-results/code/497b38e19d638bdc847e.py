import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
        
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all words and neighbor structure
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Map from word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Build neighbors list for each word
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [None] * total
    for i in range(total):
        w = all_words[i]
        nbrs = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    nbrs.append(word_to_idx[tuple(w_new)])
        neighbors[i] = nbrs
        
    # Precompute adjacency matrix M (6561x6561) for vectorized greedy
    # M[i, j] = 1 if j is a neighbor of i
    M = np.zeros((total, total), dtype=np.float32)
    for i in range(total):
        for j in neighbors[i]:
            M[i, j] = 1.0

    # 2. Greedy Construction
    uncovered = np.ones(total, dtype=np.float32)
    code_indices = []
    
    while np.any(uncovered) and time.time() < deadline - 60.0:
        # scores[i] = number of uncovered words covered by word i
        scores = M @ uncovered
        best_idx = int(np.argmax(scores))
        
        if scores[best_idx] == 0:
            break
            
        code_indices.append(best_idx)
        # Mark neighbors of best_idx as covered
        uncovered[neighbors[best_idx]] = 0.0

    # Ensure full coverage (if greedy didn't finish due to time, finish it)
    if np.any(uncovered):
        # Fallback: simple loop
        while np.any(uncovered):
            # Find word covering most uncovered
            # This is slow, but should be rare
            best_idx = -1
            best_cnt = -1
            for i in range(total):
                if i in code_indices: continue # Optimization: skip already in code? No, we need to check all
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0: break
            code_indices.append(best_idx)
            uncovered[neighbors[best_idx]] = 0.0

    # 3. Local Search
    # State:
    code_set = set(code_indices)
    code_list = list(code_indices)
    
    # cover_count[j] = number of codewords that cover word j
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors[c]] += 1

    best_size = len(code_list)
    best_code = list(code_list)
    
    # Local search loop
    # We alternate between removal and swap attempts
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # --- Phase 1: Removal ---
        # Shuffle to avoid bias
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            c = code_list[i]
            # Check if c is redundant: all its neighbors are covered by at least one other codeword
            # i.e., cover_count[neighbors[c]] > 1 for all neighbors
            if np.all(cover_count[neighbors[c]] > 1):
                # Remove c
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors[c]] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)

        if improved:
            no_improvement_count = 0
            continue
        
        # --- Phase 2: Swap ---
        # Try to swap a codeword with another one to maintain coverage but potentially allow future removals?
        # Actually, standard 1-for-1 swap: Remove c, find c' that covers the gap.
        # This doesn't reduce size, but might lead to a state where removal is possible.
        
        # Pick a random codeword to attempt swap on
        if code_list:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            cover_count[neighbors[c]] -= 1
            
            # Find uncovered words
            # Uncovered are those with cover_count == 0
            # We only care about neighbors of c, as only they changed
            nbrs_c = neighbors[c]
            uncovered_words = [j for j in nbrs_c if cover_count[j] == 0]
            
            if not uncovered_words:
                # c was redundant, but Phase 1 should have caught it? 
                # Maybe it became redundant due to previous swaps in this iteration?
                # Just remove it.
                code_list.pop(idx_c)
                code_set.discard(c)
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
            else:
                # Try to find a replacement c'
                # c' must be a neighbor of all words in uncovered_words
                # Intersection of neighbors of uncovered_words
                candidates = set(neighbors[uncovered_words[0]])
                for u in uncovered_words[1:]:
                    candidates &= set(neighbors[u])
                
                # Remove existing codewords from candidates
                candidates -= code_set
                
                if candidates:
                    # Pick a candidate. 
                    # Heuristic: Pick the one that covers the most *other* uncovered words? 
                    # Or just pick one that is not in code_set.
                    # Let's pick the one with highest cover_count potential? 
                    # Simpler: just pick one randomly or the first one.
                    # To be deterministic-ish but diverse, pick max index?
                    c_prime = max(candidates) 
                    
                    # Perform swap
                    code_list[idx_c] = c_prime
                    code_set.add(c_prime)
                    cover_count[neighbors[c_prime]] += 1
                    improved = True
                    # Note: Size didn't change, so best_size doesn't update here.
                    # But this new configuration might allow removals.
                else:
                    # Revert
                    cover_count[neighbors[c]] += 1
                    no_improvement_count += 1

        if not improved:
            no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Restart local search with a slightly perturbed state?
            # Or just reset counter and try again with different random seed/shuffle
            no_improvement_count = 0
            random.shuffle(code_list)
            # Optionally, do a small random restart: remove 5 random words and re-greedy
            if time.time() < deadline - 30.0 and random.random() < 0.1:
                # Aggressive restart
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, num_remove)
                for c in to_remove:
                    code_set.discard(c)
                    cover_count[neighbors[c]] -= 1
                    code_list.remove(c)
                
                # Re-greedy to fix coverage
                # Use vectorized greedy on current uncovered
                current_uncovered = (cover_count == 0).astype(np.float32)
                while np.any(current_uncovered) and time.time() < deadline - 15.0:
                    scores = M @ current_uncovered
                    # Mask out words already in code_set? 
                    # M @ current_uncovered gives score for ALL words.
                    # We need to pick a word NOT in code_set.
                    # Create a mask
                    mask = np.ones(total, dtype=bool)
                    for c in code_set:
                        mask[c] = False
                    scores_masked = np.where(mask, scores, -1)
                    best_idx = int(np.argmax(scores_masked))
                    
                    if scores_masked[best_idx] <= 0:
                        break
                        
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    cover_count[neighbors[best_idx]] += 1
                    current_uncovered[neighbors[best_idx]] = 0.0

    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
