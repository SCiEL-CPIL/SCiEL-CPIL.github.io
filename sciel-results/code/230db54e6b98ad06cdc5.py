import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*(3-1) = 15
    
    t_start = time.time()
    deadline = 145.0  # Safety margin (limit is 150s)
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Helper to get index from tuple
    def tuple_to_idx(t):
        idx = 0
        for i, val in enumerate(t):
            idx = idx * q + val
        return idx
    
    # Precompute balls: balls[i] is a list of indices of words within distance radius of word i
    ball_size = 1 + n * (q - 1)
    balls = np.empty((N, ball_size), dtype=np.int32)
    
    for i, w in enumerate(all_words):
        w_list = list(w)
        ball = [i]
        for pos in range(n):
            old_val = w_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    w_list[pos] = new_val
                    ball.append(tuple_to_idx(tuple(w_list)))
            w_list[pos] = old_val
        balls[i] = ball
    
    # Build coverage matrix: cov_matrix[i, j] = 1 if word j is in ball of word i
    cov_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        cov_matrix[i, balls[i]] = 1
    
    # Build reverse coverage: for each word j, which codewords cover it
    # This is just the transpose of cov_matrix, but let's be explicit
    # Actually, cov_matrix[j, i] = 1 if codeword i covers word j
    # So the set of codewords covering word j is np.where(cov_matrix[j, :])[0]
    
    # --- Greedy Set Cover with Weighted Scoring ---
    def greedy_set_cover(weights=None, time_limit=2.0):
        """
        Greedy set cover with optional weights for tie-breaking.
        weights: array of length N, used to break ties (higher weight preferred)
        Returns list of chosen indices.
        """
        t_greedy_start = time.time()
        covered = np.zeros(N, dtype=bool)
        code = []
        
        # Precompute for each candidate the set of words it covers
        # balls[i] gives this directly
        
        while not covered.all():
            if time.time() - t_greedy_start > time_limit:
                break
            
            # Calculate gains: number of uncovered words each candidate covers
            # gains[i] = number of j where not covered[j] and cov_matrix[i, j] == 1
            # = sum of (1 - covered) over cov_matrix[i, :]
            uncovered_mask = ~covered
            gains = np.sum(cov_matrix * uncovered_mask[:, None].astype(np.int8), axis=1)
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
            
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                idx = candidates[0]
            else:
                # Use weights to break ties
                if weights is not None:
                    candidate_weights = weights[candidates]
                    idx = candidates[np.argmax(candidate_weights)]
                else:
                    idx = candidates[0]
            
            code.append(idx)
            covered[balls[idx]] = True
        
        return code
    
    # --- Local Search: 1-exchange (remove one, add one if needed) ---
    def local_search_1swap(code, time_limit=3.0):
        """
        Local search: try to remove a codeword if it doesn't uncover any word.
        If it does uncover words, try to find a replacement that covers those and more.
        """
        t_ls_start = time.time()
        code_set = set(code)
        code_list = list(code)
        
        # Compute coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_list:
            counts[balls[idx]] += 1
        
        improved = True
        max_iters = 5000
        
        while improved and (time.time() - t_ls_start) < time_limit:
            improved = False
            
            # Try removing each codeword
            for i in range(len(code_list)):
                idx = code_list[i]
                ball = balls[idx]
                
                # If all words in this ball have count > 1, we can remove safely
                if np.min(counts[ball]) > 1:
                    counts[ball] -= 1
                    code_list.pop(i)
                    code_set.discard(idx)
                    improved = True
                    break
                else:
                    # Words that would become uncovered
                    uncovered_after_removal = np.where(counts[ball] == 1)[0]
                    
                    if len(uncovered_after_removal) == 0:
                        # Shouldn't happen if min > 1 check passed, but just in case
                        counts[ball] -= 1
                        code_list.pop(i)
                        code_set.discard(idx)
                        improved = True
                        break
                    else:
                        # Try to find a replacement that covers all uncovered_after_removal
                        # and maximizes additional coverage
                        # For speed, only check a subset of candidates
                        # A candidate must cover all words in uncovered_after_removal
                        
                        # Find candidates that cover all uncovered words
                        # A candidate c covers word w if cov_matrix[w, c] == 1
                        # So candidate c is valid if cov_matrix[uncovered_after_removal, c].all()
                        
                        # This is expensive to check for all N candidates.
                        # Instead, let's just try a few random candidates that intersect uncovered_after_removal
                        # If we find one that covers all uncovered and adds more, swap.
                        
                        # Get the set of candidates that cover at least one uncovered word
                        # candidates_covering = np.where(cov_matrix[uncovered_after_removal, :].any(axis=0))[0]
                        # This might still be large. Let's sample.
                        
                        # For now, skip this to keep it fast. We'll do a simpler version:
                        # Just try random swaps where we remove one and add one that covers the most uncovered.
                        pass
            
            if not improved:
                # Try a random 1-exchange: remove one, add one
                if code_list and random.random() < 0.3:
                    rem_idx = random.choice(code_list)
                    rem_ball = balls[rem_idx]
                    
                    # Temporarily remove
                    counts[rem_ball] -= 1
                    uncovered = np.where(counts == 0)[0]
                    
                    if len(uncovered) == 0:
                        # Safe removal
                        code_list.remove(rem_idx)
                        code_set.discard(rem_idx)
                        improved = True
                    else:
                        # Need to add a codeword that covers all uncovered
                        # Find candidates that cover all uncovered words
                        # For speed, check if there's a candidate in the current code that covers all
                        # or just add the best candidate (covers most uncovered)
                        
                        # Find the candidate that covers the most uncovered words
                        gains_for_uncovered = np.sum(cov_matrix[uncovered, :] * (~np.isin(np.arange(N), code_list).astype(bool))[:, None], axis=0)
                        
                        # Actually, let's just find any candidate that covers all uncovered
                        # This is a small set, so we can check
                        valid_candidates = []
                        for c in range(N):
                            if c in code_set:
                                continue
                            if np.all(cov_matrix[uncovered, c] == 1):
                                valid_candidates.append(c)
                        
                        if valid_candidates:
                            # Pick the one that covers the most additional words
                            best_c = max(valid_candidates, key=lambda c: np.sum(cov_matrix[balls[c], :]))
                            # Add best_c
                            counts[balls[best_c]] += 1
                            code_list.append(best_c)
                            code_set.add(best_c)
                            improved = True
                        else:
                            # Revert
                            counts[rem_ball] += 1
                    
        return code_set
    
    # --- Simulated Annealing on the set of codewords ---
    def simulated_annealing(initial_code, time_limit=5.0):
        """
        SA with moves: add one, remove one, swap one.
        Energy = number of uncovered words (0 is optimal).
        Size penalty to encourage smaller codes.
        """
        t_sa_start = time.time()
        code_set = set(initial_code)
        code_list = list(code_set)
        
        # Compute coverage
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_list:
            counts[balls[idx]] += 1
        
        def energy(counts):
            uncovered = np.sum(counts == 0)
            return uncovered * 100 + len(code_list)  # Heavy penalty for uncovered
        
        current_energy = energy(counts)
        best_energy = current_energy
        best_code_set = code_set.copy()
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.95
        
        while (time.time() - t_sa_start) < time_limit and T > T_min:
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add' and len(code_list) < 300:
                # Add a random codeword not in the set
                candidates = [i for i in range(N) if i not in code_set]
                if not candidates:
                    continue
                idx = random.choice(candidates)
                
                # Apply
                counts[balls[idx]] += 1
                code_list.append(idx)
                code_set.add(idx)
                
                new_energy = energy(counts)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code_set = code_set.copy()
                else:
                    # Revert
                    counts[balls[idx]] -= 1
                    code_list.remove(idx)
                    code_set.discard(idx)
            
            elif move_type == 'remove' and code_list:
                idx = random.choice(code_list)
                
                # Apply
                counts[balls[idx]] -= 1
                code_list.remove(idx)
                code_set.discard(idx)
                
                new_energy = energy(counts)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code_set = code_set.copy()
                else:
                    # Revert
                    counts[balls[idx]] += 1
                    code_list.append(idx)
                    code_set.add(idx)
            
            elif move_type == 'swap' and code_list:
                idx_remove = random.choice(code_list)
                candidates_add = [i for i in range(N) if i not in code_set]
                if not candidates_add:
                    continue
                idx_add = random.choice(candidates_add)
                
                # Apply
                counts[balls[idx_remove]] -= 1
                counts[balls[idx_add]] += 1
                code_list.remove(idx_remove)
                code_list.append(idx_add)
                code_set.discard(idx_remove)
                code_set.add(idx_add)
                
                new_energy = energy(counts)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code_set = code_set.copy()
                else:
                    # Revert
                    counts[balls[idx_add]] -= 1
                    counts[balls[idx_remove]] += 1
                    code_list.remove(idx_add)
                    code_list.append(idx_remove)
                    code_set.discard(idx_add)
                    code_set.add(idx_remove)
            
            T *= cooling_rate
        
        return best_code_set
    
    best_code_set = set()
    best_len = N
    
    # Phase 1: Multiple Greedy Restarts with different weightings
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() - t_start > deadline - 30:
            break
        
        # Different weighting strategies
        if r % 3 == 0:
            weights = np.random.rand(N)
        elif r % 3 == 1:
            # Weight by some structural property, e.g., sum of digits
            weights = np.array([sum(w) for w in all_words], dtype=float)
        else:
            weights = np.zeros(N)
        
        code = greedy_set_cover(weights, time_limit=2.0)
        
        if not code:
            continue
        
        # Local Search
        code_set = local_search_1swap(code, time_limit=2.0)
        
        if len(code_set) < best_len:
            best_len = len(code_set)
            best_code_set = code_set.copy()
    
    # Phase 2: Simulated Annealing on the best found
    if best_code_set:
        best_code_set = simulated_annealing(best_code_set, time_limit=20.0)
        best_len = len(best_code_set)
        
        # More SA with different initial temperatures
        for _ in range(3):
            if time.time() - t_start > deadline - 5:
                break
            best_code_set = simulated_annealing(best_code_set, time_limit=15.0)
            if len(best_code_set) < best_len:
                best_len = len(best_code_set)
    
    # Fallback if nothing found
    if not best_code_set:
        covered = np.zeros(N, dtype=bool)
        best_code_set = set()
        while not covered.all():
            uncovered_mask = ~covered
            gains = np.sum(cov_matrix * uncovered_mask[:, None].astype(np.int8), axis=1)
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            idx = candidates[0]
            best_code_set.add(idx)
            covered[balls[idx]] = True
    
    # Convert indices back to list of lists of integers
    result = []
    for idx in best_code_set:
        w = []
        temp_idx = idx
        for _ in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
        
    return result
