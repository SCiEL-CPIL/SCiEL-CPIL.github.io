import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute all balls as integer indices
    # Using a fast base-q conversion
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        for pos in range(n):
            orig_val = t_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t_new = t_i[:]
                t_new[pos] = new_val
                ball.append(tuple_to_int(t_new, q))
        balls[i] = ball

    # Coverage matrix M where M[i, j] = 1 if j is in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    def greedy_cover(covered_mask, existing_codewords=None, time_limit=None):
        """
        Greedily add codewords to cover 'covered_mask' (True where uncovered).
        Returns list of added codewords and the final covered_mask.
        """
        if existing_codewords is None:
            existing_codewords = []
        
        # Initialize coverage state
        # We need to know which points are already covered by existing_codewords
        # If covered_mask is passed as 'uncovered', then covered = ~covered_mask
        # Let's just work with a boolean array 'uncovered'
        
        uncovered = covered_mask.copy()
        added = []
        
        # If we have existing codewords, we should have updated 'uncovered' already?
        # The caller should pass the current uncovered mask.
        
        while np.any(uncovered):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Score: number of uncovered points covered by each candidate
            # M.dot(uncovered) gives count of uncovered points in each ball
            scores = M.dot(uncovered.astype(np.uint8))
            best_idx = np.argmax(scores)
            
            if scores[best_idx] == 0:
                break # Should not happen if uncovered is not empty and M is valid
            
            added.append(best_idx)
            # Update uncovered mask
            uncovered[balls[best_idx]] = False
            
        return added, uncovered

    # Phase 1: Initial Greedy Set Cover
    initial_uncovered = np.ones(total_words, dtype=bool)
    codewords, _ = greedy_cover(initial_uncovered, time_limit=deadline - 5)
    
    # Maintain cover_count for local search
    cover_count = np.zeros(total_words, dtype=np.uint8)
    for c in codewords:
        cover_count[balls[c]] += 1

    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # Phase 2: Local Search
    # Strategy: 
    # 1. Try to remove redundant codewords (where min cover count in ball > 1)
    # 2. If stuck, try "Swap" or "Kick" (remove one, greedily re-add)
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    random.seed(42) # Deterministic for reproducibility within run, but maybe use system time? 
    # Let's use a fixed seed for stability in evaluation unless we want diversity.
    # The prompt implies deterministic or near-deterministic is fine.
    
    while time.time() < deadline - 10:
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Randomly remove a few codewords and re-greedy
                # Or just restart local search with a different random kick
                # Let's try a kick: remove a random codeword, greedily fill.
                if len(codewords) > 1:
                    rand_idx = random.randint(0, len(codewords) - 1)
                    c_to_remove = codewords[rand_idx]
                    
                    old_codewords = list(codewords)
                    old_cover_count = cover_count.copy()
                    
                    codewords.pop(rand_idx)
                    cover_count[balls[c_to_remove]] -= 1
                    
                    # Greedily add back until covered
                    uncovered_mask = cover_count == 0
                    added, _ = greedy_cover(uncovered_mask, time_limit=deadline - 10)
                    
                    for c in added:
                        codewords.append(c)
                        cover_count[balls[c]] += 1
                    
                    if len(codewords) < len(old_codewords):
                        improved = True
                        no_improve_count = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                    else:
                        # Revert
                        codewords = old_codewords
                        cover_count = old_cover_count
                        # If we keep failing, break eventually
                        if no_improve_count > max_no_improve * 2:
                            break
                else:
                    break
            else:
                # Try standard moves again
                improved = True # Reset for this pass? No, we check specific moves.
                improved = False
                no_improve_count = 0 # Reset if we try a new set of moves? 
                # Actually, let's structure it as:
                # 1. Check for removals
                # 2. If none, check for swaps/kicks
                # 3. If none, increment no_improve_count
                
                # Check for redundant codewords
                redundant_found = False
                for idx, c in enumerate(codewords):
                    if np.min(cover_count[balls[c]]) > 1:
                        # Redundant
                        c_to_remove = codewords[idx]
                        cover_count[balls[c_to_remove]] -= 1
                        codewords.pop(idx)
                        redundant_found = True
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                        break # Restart loop to re-check all
                
                if redundant_found:
                    continue
                
                # No redundant codewords. Try a kick (remove one, re-greedy)
                # To make it more effective, we can try removing a codeword that has the
                # lowest "redundancy" or just random.
                # Let's try a few random kicks in this phase if we haven't tried many.
                # For simplicity, we do one kick per iteration when stuck.
                
                if len(codewords) > 1:
                    rand_idx = random.randint(0, len(codewords) - 1)
                    c_to_remove = codewords[rand_idx]
                    
                    old_codewords = list(codewords)
                    old_cover_count = cover_count.copy()
                    
                    codewords.pop(rand_idx)
                    cover_count[balls[c_to_remove]] -= 1
                    
                    uncovered_mask = cover_count == 0
                    added, _ = greedy_cover(uncovered_mask, time_limit=deadline - 10)
                    
                    for c in added:
                        codewords.append(c)
                        cover_count[balls[c]] += 1
                    
                    if len(codewords) < len(old_codewords):
                        improved = True
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                    else:
                        codewords = old_codewords
                        cover_count = old_cover_count
                        # No improvement from kick
                        # We need to increment no_improve_count if we don't improve
                        # But the loop logic above handles no_improve_count if improved is False
                        # So we just let it fall through.
                else:
                    break
        else:
            no_improve_count = 0
            improved = False

    # Ensure we return the best found
    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    # Convert back to list of lists
    result = []
    for c in codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
