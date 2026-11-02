import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187
    p3 = [1, 3, 9, 27, 81, 243, 729]
    
    # Precompute ball masks as numpy boolean arrays for vectorized ops
    # ball_mask[i] is a boolean array of length N, True where j is within distance 1 of i
    # This uses 2187 * 2187 bytes ~ 4.8 MB, fine.
    
    # Precompute digits
    digits = np.zeros((N, 7), dtype=np.int8)
    for i in range(N):
        tmp = i
        for pos in range(7):
            digits[i, pos] = tmp % 3
            tmp //= 3
    
    # For each position and value, precompute the set of indices
    # pos_val_mask[pos][val] = boolean array of indices where digit at pos == val
    pos_val_mask = [[None] * 3 for _ in range(7)]
    for pos in range(7):
        for val in range(3):
            pos_val_mask[pos][val] = (digits[:, pos] == val)
    
    # Precompute ball_mask for each word: union of self and all neighbors
    # ball_mask[i] = pos_val_mask[digits[i,0]][digits[i,0]] & ... no wait
    # Actually: ball_mask[i] has True at j if j == i or j differs from i in exactly one position
    # Let's compute it as: for each i, start with self, then for each pos, add all j with same digits except pos, different val at pos
    
    # More efficient: precompute for each pos, the "same except pos" groups
    # For position pos, words are partitioned by their digits except pos.
    # Within each group, all words are at distance 1 from each other (differ only at pos).
    
    # Let's just compute ball_mask as a list of numpy arrays
    ball_mask = [None] * N
    for i in range(N):
        bm = np.zeros(N, dtype=bool)
        bm[i] = True
        for pos in range(7):
            orig_val = digits[i, pos]
            for val in range(3):
                if val == orig_val:
                    continue
                # Find all j with same digits as i except at pos, and digit pos = val
                # This is intersection of pos_val_mask[p][digits[i,p]] for p != pos, and pos_val_mask[pos][val]
                mask = pos_val_mask[pos][val]
                for p in range(7):
                    if p != pos:
                        mask = mask & pos_val_mask[p][digits[i, p]]
                bm |= mask
        ball_mask[i] = bm
    
    # Convert to int8 arrays for arithmetic (1 = covered, 0 = not)
    ball_mask_int = [bm.astype(np.int8) for bm in ball_mask]
    
    # Helper: greedy initialization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            # Sample a subset of candidates to speed up, or scan all
            # For speed, scan all but with early termination heuristics
            for i in range(N):
                if i in selected:
                    continue
                cnt = np.sum(uncovered & ball_mask[i])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            uncovered &= ~ball_mask[best_idx]
            
            if time.time() > deadline - 10.0:
                break
        
        return selected
    
    # Helper: local search with add/remove/swap
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        
        # Coverage count: number of codewords covering each word
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt += ball_mask_int[c]
        
        best_code = code.copy()
        best_size = len(code)
        
        while time.time() < deadline - 3.0:
            improved = False
            
            # Phase 1: Remove redundant codewords
            changed = True
            while changed:
                changed = False
                new_code = []
                for c in code:
                    # Check if c is essential: does it uniquely cover any word?
                    # A word is uniquely covered by c if cnt[word] == 1 and c covers it
                    # c is essential if any word in ball_mask[c] has cnt == 1
                    unique_covered = np.sum((cnt == 1) & ball_mask[c])
                    if unique_covered > 0:
                        new_code.append(c)
                    else:
                        # Remove c
                        cnt -= ball_mask_int[c]
                        code_set.discard(c)
                        changed = True
                code = new_code
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                    improved = True
            
            if not code:
                break
            
            # Phase 2: Try swaps
            # For a random codeword c, find words it uniquely covers
            # Then find candidates that cover all those words
            # Swap if candidate is not already in code
            
            swaps_attempted = 0
            max_swaps = 100
            
            while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
                swaps_attempted += 1
                c = rng.choice(code)
                
                # Find words uniquely covered by c
                unique_words = np.where((cnt == 1) & ball_mask[c])[0]
                
                if len(unique_words) == 0:
                    # Should have been removed in Phase 1, but handle anyway
                    cnt -= ball_mask_int[c]
                    code.remove(c)
                    code_set.discard(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code.copy()
                    improved = True
                    break
                
                # Find candidates that cover all unique_words
                # A candidate j covers all unique_words if ball_mask[j][w] is True for all w in unique_words
                # This is: AND of ball_mask[j][w] over w in unique_words
                # Equivalently: the intersection of the sets {j : ball_mask[j][w] for w in unique_words}
                
                # For each unique word w, the set of j that cover w is the set of j where ball_mask[j][w] is True
                # We need j such that for ALL w in unique_words, ball_mask[j][w] is True
                # This is: j is in the intersection over w of {j : ball_mask[j][w]}
                
                # Precompute: for each word w, the set of codeword indices that cover w
                # That's just the column w of the ball_mask matrix
                # But we don't have the full matrix as a 2D array... 
                # Let's compute it on the fly for the unique_words
                
                # candidate_mask[j] = True if j covers all unique_words
                candidate_mask = np.ones(N, dtype=bool)
                for w in unique_words:
                    # Which j cover w? That's where ball_mask[j][w] is True
                    # We need to check for each j if ball_mask[j][w]
                    # This is expensive. Alternative: 
                    # For a candidate j to cover w, j must be within distance 1 of w
                    # So j is in the ball of w
                    # We can precompute for each w, the list of j in ball(w)
                    # But that's the same as ball_mask[w]
                    
                    # Actually, ball_mask[j][w] is True iff j is within distance 1 of w
                    # Which is the same as w is within distance 1 of j (symmetric)
                    # So ball_mask[j][w] = ball_mask[w][j]
                    # So the set of j that cover w is exactly ball_mask[w]
                    
                    candidate_mask &= ball_mask[w]
                
                # Remove candidates already in code
                for c2 in code:
                    candidate_mask[c2] = False
                
                candidates = np.where(candidate_mask)[0]
                
                if len(candidates) == 0:
                    continue
                
                # Pick a random candidate
                c_prime = rng.choice(candidates.tolist())
                
                # Perform swap: remove c, add c_prime
                # But we need to check that after removing c, all words still covered
                # The words uniquely covered by c are covered by c_prime (by construction)
                # Other words: removing c might uncover some, but c_prime might cover them
                # We need to verify the full cover after swap
                
                # Tentative: remove c, add c_prime
                cnt_new = cnt - ball_mask_int[c] + ball_mask_int[c_prime]
                
                # Check if all words are still covered
                if np.all(cnt_new > 0):
                    # Accept the swap
                    cnt = cnt_new
                    code.remove(c)
                    code_set.discard(c)
                    code.append(c_prime)
                    code_set.add(c_prime)
                    
                    if len(code) < best_size:
                        # This can't happen since we swapped, size is same
                        pass
                    improved = True
                # Else: reject, try another swap
                
            # Phase 3: Perturbation if no improvement
            if not improved:
                # Remove a random codeword and re-add greedily
                if len(code) > 1:
                    c = rng.choice(code)
                    cnt -= ball_mask_int[c]
                    code.remove(c)
                    code_set.discard(c)
                    
                    # Re-add greedily to restore coverage
                    uncovered = np.where(cnt == 0)[0]
                    if len(uncovered) > 0:
                        # Greedily add codewords to cover uncovered
                        while len(uncovered) > 0:
                            best_idx = -1
                            best_count = -1
                            for i in range(N):
                                if i in code_set:
                                    continue
                                cnt2 = np.sum((cnt == 0) & ball_mask[i])
                                if cnt2 > best_count:
                                    best_count = cnt2
                                    best_idx = i
                            if best_idx == -1 or best_count == 0:
                                break
                            cnt += ball_mask_int[best_idx]
                            code.append(best_idx)
                            code_set.add(best_idx)
                            uncovered = np.where(cnt == 0)[0]
                
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
        
        return best_code
    
    # Main loop: iterated local search
    best_code = None
    best_size = float('inf')
    
    # Start with greedy
    seed = 42
    while time.time() < deadline - 5.0:
        if time.time() > deadline - 15.0 and best_code is not None:
            break
        
        code = greedy_init(seed)
        if len(code) == 0:
            seed += 1
            continue
        
        # Run local search
        code = local_search(code, seed)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
        
        seed += 1
    
    if best_code is None:
        # Fallback: return a valid but possibly large code
        code = greedy_init(0)
        best_code = code
    
    # Convert indices back to words
    words = []
    for i in best_code:
        w = []
        tmp = i
        for pos in range(7):
            w.append(tmp % 3)
            tmp //= 3
        words.append(w)
    
    return words

def _construct_general(q, n, radius, deadline):
    # Fallback for other parameters: simple greedy
    import itertools
    N = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Compute balls
    balls = []
    for i, w in enumerate(words):
        ball = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                neigh = list(w)
                neigh[pos] = val
                j = word_to_idx[tuple(neigh)]
                ball.add(j)
        balls.append(ball)
    
    # Greedy
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in range(N):
            if i in code:
                continue
            cnt = len(uncovered & balls[i])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1 or best_count == 0:
            break
        code.append(best_idx)
        uncovered -= balls[best_idx]
        if time.time() > deadline - 5.0:
            break
    
    return [words[i] for i in code]
