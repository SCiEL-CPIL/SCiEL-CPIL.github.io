"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    def word_to_idx(word, q, n):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx, q, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for each word (for radius=1)
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx, q, n)
        neighbors[idx].append(idx)  # self
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        nb_idx = word_to_idx(w, q, n)
                        neighbors[idx].append(nb_idx)
                w[pos] = orig
    
    # Convert neighbors to numpy arrays for faster operations
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute neighbor matrix once (shared across all functions)
    nm = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        nm[i, neighbors_np[i]] = True
    
    # Convert nm to int32 for faster dot products
    nm_int = nm.astype(np.int32)
    
    def greedy_cover(restart_seed=0, initial_codewords=None):
        """Greedy set cover with coverage counting."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(restart_seed)
        
        # Coverage count: how many codewords cover each word
        coverage = np.zeros(total_words, dtype=np.int32)
        codewords = []
        
        # If we have initial codewords, add them
        if initial_codewords:
            for cw in initial_codewords:
                if cw not in codewords:
                    codewords.append(cw)
                    coverage[neighbors_np[cw]] += 1
        
        # Uncovered words as a boolean array (True = uncovered)
        uncovered_arr = np.ones(total_words, dtype=np.int32)
        for cw in codewords:
            uncovered_arr[neighbors_np[cw]] = 0
        
        # Greedy: repeatedly pick the codeword that covers the most uncovered words
        while time.time() < deadline:
            # Check if all covered
            if uncovered_arr.sum() == 0:
                break
            
            # Count uncovered neighbors for each candidate
            counts = nm_int.dot(uncovered_arr)
            
            # Find the candidate with max count
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Get all candidates with max count
            candidates = np.where(counts == best_count)[0]
            
            # Tie-breaking: Choose randomly among max-count candidates
            if len(candidates) > 1:
                best_cw = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                best_cw = int(candidates[0])
            
            # Add best_cw
            codewords.append(best_cw)
            uncovered_arr[neighbors_np[best_cw]] = 0
            coverage[neighbors_np[best_cw]] += 1
        
        return codewords
    
    def remove_redundant_fast(codewords):
        """
        Remove redundant codewords using critical-words tracking.
        A codeword is removable iff all words in its ball are covered by >= 2 codewords.
        We maintain a set of 'critical' words (coverage == 1) for O(1) amortized checks.
        """
        if time.time() > deadline:
            return codewords
        
        if not codewords:
            return codewords
        
        codewords = list(codewords)
        
        # Build coverage count
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1
        
        # Critical words: those with coverage exactly 1
        critical_set = set(np.where(coverage == 1)[0].tolist())
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                cw_neighbors = neighbors_np[cw]
                
                # Check if any neighbor is critical
                has_critical = False
                for nb in cw_neighbors:
                    if nb in critical_set:
                        has_critical = True
                        break
                
                if not has_critical:
                    # Remove codeword i
                    coverage[cw_neighbors] -= 1
                    # Update critical set
                    for nb in cw_neighbors:
                        if coverage[nb] == 1:
                            critical_set.add(nb)
                        elif nb in critical_set:
                            critical_set.discard(nb)
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def local_search_aggressive(codewords, max_rounds=100):
        """
        Aggressive local search:
        1. Remove redundancies using critical-words method.
        2. Try to swap codewords to enable further removals.
        3. Try to remove codewords by adding a replacement that covers its unique words.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        codewords = remove_redundant_fast(codewords)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            codewords = remove_redundant_fast(codewords)
            
            if len(codewords) <= 73:
                break
            
            # Compute coverage
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
            
            # Find codewords with most slack (words covered by >= 2)
            slack_scores = []
            for cw in codewords:
                slack = (coverage[neighbors_np[cw]] >= 2).sum()
                slack_scores.append((slack, cw))
            
            slack_scores.sort(reverse=True)
            
            # Try swapping each high-slack codeword with a better neighbor
            for slack_val, cw in slack_scores[:20]:  # Limit to top 20 for speed
                if time.time() > deadline:
                    return codewords
                
                # Find words covered only by cw
                unique_mask = (coverage == 1) & nm[cw, :]
                unique_words = np.where(unique_mask)[0]
                
                # Find best replacement among neighbors of cw
                neighbors_of_cw = neighbors_np[cw]
                best_new_cw = -1
                best_score = -1
                
                for new_cw in neighbors_of_cw:
                    if new_cw in codewords:
                        continue
                    
                    # Check validity: new_cw must cover all unique words of cw
                    if len(unique_words) > 0:
                        if not np.all(nm[new_cw, unique_words]):
                            continue
                    
                    # Score: minimize overlap with already-covered words (maximize new coverage)
                    # Or: maximize words that are covered by >= 2 (slack)
                    score = (nm[new_cw, :] & (coverage >= 2)).sum()
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Execute swap
                    codewords.remove(cw)
                    codewords.append(best_new_cw)
                    
                    # Try to remove redundancies
                    new_codewords = remove_redundant_fast(codewords)
                    
                    if len(new_codewords) < len(codewords):
                        codewords = new_codewords
                        improved = True
                        break
                    else:
                        # Rollback
                        codewords.remove(best_new_cw)
                        codewords.append(cw)
        
        return remove_redundant_fast(codewords)
    
    def final_polish_critical(codewords):
        """
        Final high-precision polish using critical-words method.
        Try removing each codeword one at a time, checking if critical set becomes empty.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        codewords = remove_redundant_fast(codewords)
        
        # Try removing each codeword one at a time
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                
                cw = codewords[i]
                cw_neighbors = neighbors_np[cw]
                
                # Check if any neighbor is critical (coverage == 1)
                coverage = np.zeros(total_words, dtype=np.int32)
                for j, other_cw in enumerate(codewords):
                    if j != i:
                        coverage[neighbors_np[other_cw]] += 1
                
                # Check if any word has coverage 0
                if coverage.min() >= 1:
                    # cw is redundant
                    codewords.pop(i)
                    changed = True
                    break
        
        return codewords
    
    def lagrangian_relaxation(codewords, max_iters=30):
        """
        Lagrangian relaxation: use subgradient method to escape local minima.
        """
        if time.time() > deadline:
            return codewords
        
        codewords = list(codewords)
        codewords = remove_redundant_fast(codewords)
        
        # Compute coverage
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            coverage[neighbors_np[cw]] += 1
        
        # Dual variables
        dual_vars = np.ones(total_words, dtype=np.float64)
        step_size = 1.0
        best_codewords = codewords.copy()
        best_count = len(codewords)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Weighted marginal benefit
            uncovered_weight = np.where(coverage < 1, dual_vars, 0.0)
            scores = nm_int.dot(uncovered_weight.astype(np.int32))
            
            # Mask out current codewords
            in_codewords = np.zeros(total_words, dtype=bool)
            for cw in codewords:
                in_codewords[cw] = True
            scores[in_codewords] = -1
            
            best_score = scores.max()
            if best_score <= 0:
                break
            
            best_cw = int(np.argmax(scores))
            codewords.append(best_cw)
            coverage[neighbors_np[best_cw]] += 1
            
            codewords = remove_redundant_fast(codewords)
            
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
            
            if len(codewords) < best_count:
                best_count = len(codewords)
                best_codewords = codewords.copy()
            
            gradient = np.where(coverage < 1, 1.0, 0.0)
            dual_vars += step_size * gradient
            step_size *= 0.95
        
        return best_codewords
    
    # Strategy:
    # 1. Start with tetracode-based construction (81 words)
    # 2. Aggressively remove redundant codewords using critical-words method
    # 3. Try multiple random restarts of greedy
    # 4. Use aggressive local search
    # 5. Apply Lagrangian relaxation
    # 6. Final polish
    
    best_result = None
    best_count = float('inf')
    
    # Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word, q, n)
                initial_codewords.append(idx)
    
    # Remove redundant from initial
    reduced = remove_redundant_fast(initial_codewords.copy())
    if len(reduced) < best_count:
        best_count = len(reduced)
        best_result = reduced
    
    # Try greedy from scratch with different seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        
        result = greedy_cover(restart_seed=seed, initial_codewords=None)
        if result is not None:
            result = remove_redundant_fast(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Try hybrid: start with partial tetracode, then greedy
    for seed in range(15):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        initial = rng.sample(initial_codewords, k=min(len(initial_codewords), rng.randint(30, 60)))
        result = greedy_cover(restart_seed=seed, initial_codewords=initial)
        if result is not None:
            result = remove_redundant_fast(result)
            if len(result) < best_count:
                best_count = len(result)
                best_result = result
    
    # Apply aggressive local search to best results
    if best_result is not None:
        ls_result = local_search_aggressive(best_result)
        if len(ls_result) < best_count:
            best_count = len(ls_result)
            best_result = ls_result
    
    # Apply Lagrangian relaxation
    if best_result is not None:
        lag_result = lagrangian_relaxation(best_result, max_iters=25)
        lag_result = remove_redundant_fast(lag_result)
        if len(lag_result) < best_count:
            best_count = len(lag_result)
            best_result = lag_result
    
    # Final local search
    if best_result is not None:
        best_result = local_search_aggressive(best_result)
        best_result = remove_redundant_fast(best_result)
        
        # Final polish
        best_result = final_polish_critical(best_result)
    
    # Convert back to lists
    if best_result is None:
        return [idx_to_word(idx, q, n) for idx in initial_codewords]
    
    return [idx_to_word(idx, q, n) for idx in best_result]
# EVOLVE-BLOCK-END
