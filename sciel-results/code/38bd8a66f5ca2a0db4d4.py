import itertools
import random
import time

def construct(q, n, radius):
    # This task is specifically for q=3, n=6, radius=1
    # But we should handle the general signature, though the contract implies this specific instance.
    # The evaluator passes q, n, radius.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though likely not used in this specific task context
        # Simple greedy or direct sum if applicable
        # For now, return a trivial cover (all words) or greedy
        # Greedy set cover
        all_words = list(itertools.product(range(q), repeat=n))
        balls = {}
        for w in all_words:
            ball = [w]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(tuple(new_w))
            balls[w] = ball
            
        code = []
        covered = set()
        # Greedy
        remaining = set(all_words)
        while remaining:
            best_word = None
            best_count = -1
            # Sample a few words to speed up if n is large, but n=6 is small
            for w in remaining:
                cnt = len(set(balls[w]) & remaining)
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
            if best_word is None:
                break
            code.append(list(best_word))
            remaining -= set(balls[best_word])
        return code

    # Specific case q=3, n=6, radius=1
    # Precompute balls for all 729 words
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    balls = [None] * 729
    
    for w in all_words:
        ball = [w]
        for i in range(6):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                ball.append(tuple(new_w))
        balls[word_to_idx[w]] = ball

    # Initial code: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
    
    code = [tuple(c) for c in initial_code]
    
    # State: cover_count for each word
    # cover_count[i] = number of codewords covering word i
    cover_count = [0] * 729
    
    def update_counts(cw, delta):
        idx = word_to_idx[cw]
        for neighbor in balls[idx]:
            cover_count[word_to_idx[neighbor]] += delta

    # Initialize counts
    for c in code:
        update_counts(c, 1)
        
    def is_valid():
        return min(cover_count) > 0
        
    def try_remove(cw):
        idx = word_to_idx[cw]
        # Check if removing cw leaves any word uncovered
        # We need to check if any neighbor of cw has count == 1
        # If so, removing it makes count 0 -> invalid
        for neighbor in balls[idx]:
            if cover_count[word_to_idx[neighbor]] == 1:
                return False
        # Safe to remove
        update_counts(cw, -1)
        code.remove(cw)
        return True

    def find_swap(cw_old):
        # Try to find a word cw_new such that replacing cw_old with cw_new is valid
        # Remove cw_old temporarily
        idx_old = word_to_idx[cw_old]
        for neighbor in balls[idx_old]:
            cover_count[word_to_idx[neighbor]] -= 1
            
        # Find uncovered words
        uncovered = [i for i in range(729) if cover_count[i] == 0]
        
        if not uncovered:
            # We could just remove it, but try_remove handles that.
            # If we are in swap, we assume removal failed, so uncovered should not be empty.
            # But just in case:
            update_counts(cw_old, 1)
            return None

        # We need a word cw_new that covers all words in 'uncovered'
        # cw_new must be in the intersection of balls of all uncovered words
        # Since |uncovered| is usually small, we can find candidates
        
        # Optimization: If |uncovered| > 1, the candidate must be close to all of them.
        # Let's find the intersection of balls of the first few uncovered words.
        
        # Get ball of first uncovered word
        u0 = uncovered[0]
        candidates = set(balls[u0])
        
        for u in uncovered[1:]:
            if not candidates:
                break
            candidates &= set(balls[u])
            
        if not candidates:
            # Restore
            update_counts(cw_old, 1)
            return None
            
        # Try candidates
        # Prefer candidates that are not already in the code (to change structure)
        # Or just pick a random one
        code_set = set(code)
        valid_candidates = [c for c in candidates if c not in code_set]
        
        if not valid_candidates:
            # If all candidates are in the code, swapping with them doesn't change the set (just permutation)
            # Unless we consider them distinct? No, code is a set of words.
            # So no valid *new* structure swap here.
            update_counts(cw_old, 1)
            return None
            
        # Pick a random candidate
        cw_new = random.choice(valid_candidates)
        
        # Verify validity (should be valid by construction, but double check)
        # Adding cw_new
        idx_new = word_to_idx[cw_new]
        for neighbor in balls[idx_new]:
            cover_count[word_to_idx[neighbor]] += 1
            
        # Check if any count became 0? No, we added.
        # Check if we covered the uncovered ones? Yes, by construction.
        # Check if we broke anything else?
        # The only changes were -1 on old ball, +1 on new ball.
        # Words in old ball \ new ball: lost 1. They were >= 1 (since we checked uncovered before adding new? No).
        # Wait.
        # We removed old. Some words became 0 (uncovered).
        # We add new. New covers uncovered.
        # What about words in old ball that are NOT in new ball?
        # Their count decreased by 1.
        # Before removal, they were >= 1.
        # If they were 1, they became 0.
        # Did we check for this?
        # `uncovered` list contains ALL words with count 0 after removal.
        # So if a word in old ball \ new ball had count 1, it is in `uncovered`.
        # And `cw_new` covers all `uncovered`.
        # So `cw_new` covers that word.
        # So its count goes back to 1.
        # So it is valid.
        
        # Accept swap
        code.remove(cw_old)
        code.append(cw_new)
        
        return True

    deadline = time.time() + 100.0
    best_code = code[:]
    best_size = len(code)
    
    # Phase 1: Exhaustive Removal
    while time.time() < deadline:
        progress = False
        # Randomize order to avoid bias? Or just iterate.
        # Iterating is fine.
        for i in range(len(code)):
            if time.time() > deadline:
                break
            cw = code[i]
            if try_remove(cw):
                progress = True
                # After removal, indices shift, so restart loop
                break
        if not progress:
            break
            
    if len(code) < best_size:
        best_code = code[:]
        best_size = len(code)
        
    # Phase 2: Swap Search to escape local minima
    # If we are still > 73, we try swaps
    if len(code) > 73:
        stuck_count = 0
        while time.time() < deadline:
            progress = False
            # Try directed swaps for each word
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                cw = code[i]
                if find_swap(cw):
                    progress = True
                    break # Restart loop as code changed
            if not progress:
                # Try random valid swaps to shuffle
                # Pick random cw_old
                i = random.randint(0, len(code) - 1)
                cw_old = code[i]
                # Try random cw_new from space
                # To make it efficient, only try if it covers the deficit
                # This is what find_swap does.
                # If find_swap failed for all, we are stuck.
                # Let's try a "shake": remove a word (even if invalid) and re-add best?
                # No, that increases size.
                
                # Alternative: Randomly swap two codewords? No, must maintain validity.
                
                # If stuck, we might need to increase size temporarily?
                # Or just accept we are at local min.
                # Given the problem size, the local min from 81 should be 73 or close.
                # If we are at 74, we might need a more aggressive search.
                
                # Let's try one more thing: Random restart of the swap phase
                # by shuffling the code order? No, set is unordered.
                
                # We can try to swap a word with a word at distance 2?
                # find_swap already searches all 729 words.
                
                # If we are stuck at > 73, we might have to settle.
                # But 73 is known to exist.
                # The directed swap heuristic is quite strong.
                
                # If no progress in a while, break?
                # Or just run until deadline.
                pass
                
            if len(code) < best_size:
                best_code = code[:]
                best_size = len(code)
                
    # Return the best code found
    # Ensure it's a list of lists
    return [list(c) for c in best_code]
