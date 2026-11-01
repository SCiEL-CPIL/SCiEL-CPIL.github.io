import numpy as np
import time
import itertools

def construct(q, n, radius):
    return _construct_impl(q, n, radius)

def _construct_impl(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_space = q ** n
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    # Precompute balls
    balls = []
    for i in range(total_space):
        c = all_centers[i]
        indices = [i]
        if radius >= 1:
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig: continue
                    new_c = c.copy()
                    new_c[pos] = val
                    indices.append(center_to_idx[tuple(new_c)])
        balls.append(np.array(indices, dtype=np.int32))
        
    balls_matrix = np.array(balls, dtype=np.int32)
    
    def greedy_cover(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(total_space, dtype=bool)
        chosen = []
        while True:
            if time.time() > deadline:
                break
            # Vectorized score calculation
            # covered is bool array
            # balls_matrix is (N, ball_size)
            # ~covered[balls_matrix] gives bool matrix (N, ball_size)
            # sum axis=1 gives count of uncovered in each ball
            scores = (~covered[balls_matrix]).sum(axis=1)
            max_score = scores.max()
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[rng.integers(len(candidates))]
            chosen.append(best_idx)
            covered[balls[best_idx]] = True
        return chosen

    def local_search(chosen, seed):
        # chosen is list of indices
        chosen_set = set(chosen)
        # counts: how many times each word is covered
        counts = np.zeros(total_space, dtype=np.int32)
        for c in chosen:
            counts[balls[c]] += 1
            
        rng = np.random.default_rng(seed)
        
        # Heuristic: Try to remove or swap
        no_improvement = 0
        max_iters = 2000 # Adjust based on time
        
        while time.time() < deadline - 5.0:
            if no_improvement > 100:
                break
                
            # Pick random codeword
            c = chosen_set.pop() # Remove from set temporarily for logic? No, keep set separate.
            # Let's pick from list
            c = list(chosen_set)[rng.integers(len(chosen_set))]
            
            # Try removal
            ball_c = balls[c]
            counts[ball_c] -= 1
            
            if np.min(counts[ball_c]) > 0:
                # Successful removal
                chosen_set.remove(c)
                no_improvement = 0
            else:
                # Failed removal, try to find a swap
                # Uncovered words
                uncovered = np.where(counts == 0)[0]
                
                if len(uncovered) == 0:
                    # This shouldn't happen if min > 0 check failed?
                    # Ah, min > 0 failed means some are 0.
                    pass
                
                # Find best replacement
                # For each candidate not in chosen_set, count how many of 'uncovered' it covers
                # This is expensive.
                # Optimization: Only check candidates that cover at least one uncovered word.
                # Or just pick a random candidate?
                
                # Simple strategy: Find candidate that covers max uncovered.
                # Precompute inverse? No.
                
                # Just try a few random candidates
                best_gain = 0
                best_candidate = -1
                
                # Sample some random non-codewords
                # Or iterate through all? 6561 is small.
                # But doing this every step is slow.
                
                # Let's try:
                # 1. Identify uncovered words.
                # 2. For each uncovered word, look at its neighbors (potential centers).
                # 3. Count frequency of these neighbors.
                # 4. Pick the neighbor with highest frequency.
                
                candidate_counts = {}
                for u in uncovered:
                    # u is index of word
                    # Potential centers for u are neighbors of u?
                    # No, center c covers u if u in ball[c].
                    # So c is in ball_inv[u].
                    # ball_inv[u] is list of centers that cover u.
                    # We need to precompute ball_inv.
                    pass

        # Restore counts if we messed up?
        # The logic above is messy.
        
        # Let's restart with a cleaner Local Search implementation.
        return list(chosen_set)

    # Precompute inverse balls for local search efficiency
    # ball_inv[i] = list of centers j such that i in ball[j]
    ball_inv = [[] for _ in range(total_space)]
    for j in range(total_space):
        for i in balls[j]:
            ball_inv[i].append(j)
    ball_inv = [np.array(b, dtype=np.int32) for b in ball_inv]

    def local_search_v2(chosen, seed):
        chosen_set = set(chosen)
        counts = np.zeros(total_space, dtype=np.int32)
        for c in chosen:
            counts[balls[c]] += 1
            
        rng = np.random.default_rng(seed)
        improvement_found = True
        while improvement_found and time.time() < deadline - 5.0:
            improvement_found = False
            
            # Shuffle to avoid bias
            current_list = list(chosen_set)
            rng.shuffle(current_list)
            
            for c in current_list:
                if time.time() > deadline - 5.0:
                    break
                    
                # Try remove
                counts[balls[c]] -= 1
                if np.min(counts[balls[c]]) > 0:
                    chosen_set.remove(c)
                    improvement_found = True
                else:
                    # Try swap
                    uncovered = np.where(counts == 0)[0]
                    
                    # Find best replacement
                    # Candidates are in union of ball_inv[u] for u in uncovered
                    # But we only want candidates NOT in chosen_set
                    
                    # Efficiently find candidate:
                    # Count how many uncovered words each candidate covers
                    # Use a dictionary or array
                    cand_scores = {}
                    
                    for u in uncovered:
                        for cand in ball_inv[u]:
                            if cand not in chosen_set:
                                cand_scores[cand] = cand_scores.get(cand, 0) + 1
                                
                    if cand_scores:
                        # Find max
                        max_cov = max(cand_scores.values())
                        # If max_cov == len(uncovered), it's a perfect swap
                        # Even if not perfect, we might accept it if it reduces "waste"?
                        # For now, only accept if it covers ALL uncovered.
                        if max_cov == len(uncovered):
                            # Pick one
                            best_cands = [k for k, v in cand_scores.items() if v == max_cov]
                            new_c = best_cands[rng.integers(len(best_cands))]
                            
                            # Apply swap
                            counts[balls[c]] += 1 # Undo removal of c
                            chosen_set.remove(c)
                            chosen_set.add(new_c)
                            counts[balls[new_c]] += 1
                            improvement_found = True
                    else:
                        # No candidate found, undo
                        counts[balls[c]] += 1
                        
        return list(chosen_set)

    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 10.0:
        code = greedy_cover(seed)
        code = local_search_v2(code, seed)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        seed += 1

    if best_code is None:
        code = greedy_cover(0)
        best_code = local_search_v2(code, 0)

    return [all_centers[i] for i in best_code]
