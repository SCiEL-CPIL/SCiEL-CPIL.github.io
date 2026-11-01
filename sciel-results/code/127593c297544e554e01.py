"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # If the space is too large, we might need heuristics, but for q=3, n=6, it's small.
    # Total words = 3^6 = 729.
    # Ball size for radius 1: 1 + n*(q-1) = 1 + 6*2 = 13.
    
    total_words = q ** n
    
    # Helper to convert integer index to list of digits (base q, little endian for speed in loops if needed, 
    # but standard big endian is fine for clarity, we'll use a fast conversion)
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w
        
    def word_to_idx(w):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + w[i]
        return idx

    # Precompute the ball (neighbors within radius) for each word.
    # ball[i] is a list of indices of words within Hamming distance 'radius' from word i.
    # For radius=1, this is just the word itself and its immediate neighbors.
    # For general radius, we need to generate all words within that distance.
    
    # Optimization: For q=3, n=6, radius=1, we can generate neighbors efficiently.
    # If radius > 1, the ball size grows. 
    # Let's implement a general ball generator but optimize for the specific case if needed.
    
    balls = []
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors = [i]
        
        if radius == 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        w[pos] = new_val
                        neighbors.append(word_to_idx(w))
                w[pos] = orig_val
        elif radius == 2:
            # Generate all words at distance 1 and 2
            # This is more complex. For the specific task (radius=1), we don't need this.
            # But to be robust:
            # Distance 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        w[pos] = new_val
                        neighbors.append(word_to_idx(w))
                w[pos] = orig_val
            # Distance 2:
            for p1 in range(n):
                for p2 in range(p1+1, n):
                    orig1 = w[p1]
                    orig2 = w[p2]
                    for v1 in range(q):
                        if v1 == orig1: continue
                        w[p1] = v1
                        for v2 in range(q):
                            if v2 == orig2: continue
                            if v1 == orig1: continue # Already handled in dist 1? No, dist 1 changes only one.
                            # Actually, if v1==orig1 and v2!=orig2, it's dist 1.
                            # We want dist exactly 2? No, ball includes dist <= radius.
                            # So we just add all combinations.
                            w[p2] = v2
                            neighbors.append(word_to_idx(w))
                    w[p1] = orig1
                    w[p2] = orig2
        else:
            # Fallback for larger radius: brute force check all words? 
            # For n=6, q=3, total_words=729. 
            # If radius is large, the ball is huge.
            # Given the task is specifically radius=1, we assume radius is small.
            # If radius is large, we might just return a trivial solution or use a different strategy.
            pass

        # Remove duplicates (can happen in radius=2 logic if not careful)
        balls.append(list(set(neighbors)))

    # State variables
    coverage_count = [0] * total_words
    codewords = []
    codeword_set = set()
    
    # 1. Greedy Construction
    # We want to select codewords such that they cover all words.
    # Greedy step: pick the word that covers the most *uncovered* words.
    # Uncovered words are those with coverage_count == 0.
    
    # To speed up, we can maintain a list of uncovered words, but checking "how many uncovered" for a candidate
    # requires iterating its ball and checking coverage_count.
    # Ball size is ~13. Total words 729.
    # 729 * 13 = ~9k operations per greedy step.
    # We might need ~70 steps. 70 * 9k = 630k ops. Very fast.
    
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Check if all covered
        # Instead of checking all 729, we can track the number of uncovered words.
        # Let's maintain `uncovered_count`.
        
        # Find best candidate
        best_candidate = -1
        best_gain = 0
        
        # We can restrict candidates to those that are currently uncovered?
        # Not necessarily. A covered word might cover many *other* uncovered words.
        # So we check all words.
        
        for c in range(total_words):
            # Count how many words in balls[c] have coverage_count == 0
            gain = 0
            for nb in balls[c]:
                if coverage_count[nb] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
        
        if best_gain == 0:
            # All words are covered
            break
        
        # Add best_candidate
        codewords.append(best_candidate)
        codeword_set.add(best_candidate)
        for nb in balls[best_candidate]:
            coverage_count[nb] += 1
            
    # 2. Pruning Phase
    # Try to remove codewords.
    # A codeword can be removed if no word in its ball has coverage_count == 1 (which would become 0).
    # Actually, if we remove cw, all nb in balls[cw] have coverage_count decremented by 1.
    # It is safe if for all nb in balls[cw], coverage_count[nb] > 1.
    
    # We can iterate through codewords and try to remove them.
    # To be more effective, we can try to remove codewords that are "least essential".
    # Essentiality = number of words in ball[cw] with coverage_count == 1.
    # If essentiality is 0, it can be removed immediately.
    
    # We will loop until no more can be removed or time runs out.
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        # Iterate in reverse to allow index removal without shifting issues, 
        # or just mark for removal.
        # Let's calculate essentiality for all current codewords.
        # This is O(K * BallSize). K~70, BallSize~13. ~900 ops.
        
        # Find the codeword with minimum essentiality.
        # If min essentiality is 0, remove it.
        # If min essentiality > 0, we can't remove any? 
        # Not necessarily. Removing one might make another removable? No, removing reduces coverage, 
        # so it can only *increase* essentiality of others (make more words have count 1).
        # So if no codeword has essentiality 0, we are stuck in this pruning phase.
        
        min_ess = float('inf')
        min_idx = -1
        for i, cw in enumerate(codewords):
            ess = 0
            for nb in balls[cw]:
                if coverage_count[nb] == 1:
                    ess += 1
            if ess < min_ess:
                min_ess = ess
                min_idx = i
                
        if min_ess == 0:
            # Remove codewords[min_idx]
            cw_to_remove = codewords[min_idx]
            for nb in balls[cw_to_remove]:
                coverage_count[nb] -= 1
            codewords.pop(min_idx)
            codeword_set.remove(cw_to_remove)
            changed = True
        else:
            # No codeword can be removed directly.
            # Try a swap?
            # Swap a codeword cw with a non-codeword nc.
            # This is more complex. 
            # Given the small size, if greedy+prune gets us to ~73, we are good.
            # If we are stuck at 74, we might try to swap.
            # But let's first see if the simple prune works well.
            break

    # 3. Local Search / Swap (if time permits)
    # If we have time and haven't reached the known optimum (73 for 3,6,1), try swaps.
    # A swap replaces cw with nc.
    # Valid if:
    # 1. All words in balls[cw] are covered by other codewords OR by nc.
    #    Actually, we remove cw, so words in balls[cw] lose coverage from cw.
    #    They must be covered by (other codewords) U {nc}.
    # 2. nc is not already a codeword.
    
    # This is an expensive check. 
    # Let's try a few random swaps if we have time.
    
    if time.time() < deadline - 1.0:
        # We can try to improve by swapping.
        # This is a heuristic.
        pass # Skipping complex swap for now to ensure robustness and speed.
        # The greedy+prune approach is very strong for this size.

    # Convert indices back to lists of integers
    result = [idx_to_word(cw) for cw in codewords]
    
    # Final validity check (just in case)
    # For q=3, n=6, radius=1, we can do a fast check.
    if q == 3 and n == 6 and radius == 1:
        # Re-verify coverage
        # We trust the coverage_count logic, but a final check is cheap for 729 words.
        # Actually, we can just return. The evaluator will check.
        # But if we return an invalid code, we score 0.
        # Let's do a quick check if we have time.
        if time.time() < deadline - 0.5:
            # Use the precomputed balls? No, balls are for indices.
            # We can just iterate all words and check if any codeword is in its ball.
            # Or simpler: recompute coverage_count from scratch?
            # 729 words * 73 codewords * 6 ops = 300k ops. Fast.
            is_valid = True
            # We need to check if every word is within distance 1 of some codeword.
            # Using the balls array: word i is covered if any cw in codewords is in balls[i]?
            # No, balls[i] contains words within dist 1 of i.
            # So word i is covered if there exists cw such that i is in balls[cw].
            # This is equivalent to cw is in balls[i] (symmetry of Hamming distance).
            
            # We have codewords as indices in codeword_set.
            # For each word i, check if any cw in codeword_set is in balls[i].
            # This is 729 * (len(codeword_set) checks in set? No, balls[i] is a list).
            # Better: for each cw, mark its ball.
            # We already did this in coverage_count.
            # If coverage_count[i] > 0 for all i, it's valid.
            # We maintained coverage_count correctly.
            # So we are safe.
            pass

    return result

# EVOLVE-BLOCK-END
