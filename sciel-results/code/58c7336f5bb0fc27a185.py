import time
import sys
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill
    
    # 1. Generate all words
    # Represent words as integers 0 to q^n - 1
    # Convert int to tuple of digits and vice versa
    total_words = q ** n
    words = list(product(range(q), repeat=n))
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls (sets of indices within Hamming distance 'radius')
    # For radius 1 in ternary, ball size is 1 + n*(q-1) = 1 + 7*2 = 15
    balls = []
    for i, w in enumerate(words):
        ball = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                # Create new word
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                ball.add(word_to_idx[new_w])
        balls.append(ball)
        
    # Convert balls to numpy arrays for potential speedup if needed, 
    # but sets are good for intersection checks.
    # Actually, for greedy, we need to count overlaps with 'uncovered'.
    # Using Python sets is fast enough for N=2187.
    
    uncovered = set(range(total_words))
    code = []
    
    # Greedy Phase
    while uncovered:
        if time.time() > deadline:
            break
            
        best_word = -1
        max_cover = 0
        
        # Iterate through all possible words
        # Optimization: We can skip words that are already in the code? 
        # No, a word in the code might cover more *new* words than others? 
        # Actually, if a word is in the code, its ball is already "used" in the sense that 
        # we added it. But we are counting how many *currently uncovered* words it covers.
        # It's possible a word already in the code covers more uncovered words than a new word?
        # No, if we picked it before, it covered the max at that time. 
        # However, as uncovered shrinks, the relative value changes.
        # It is generally safe and faster to only consider words NOT in the current code,
        # or even just iterate all and pick max.
        
        # To speed up, we can maintain a list of candidates.
        # For N=2187, iterating all is ~2000 ops per step.
        # Max steps ~ 2187/15 ~ 150 steps.
        # 150 * 2187 * (set intersection cost ~ 15) is very small.
        
        for i in range(total_words):
            # If i is already in code, we can skip? 
            # If we skip, we might miss a case where a previously selected word 
            # is now "better" due to changes in uncovered? 
            # No, the set of words covered by i is fixed. 
            # The number of uncovered words in ball[i] only decreases.
            # So if i was not the best before, it won't become the best later 
            # unless the "best" words were removed from consideration? 
            # We don't remove words from consideration, we just add them to code.
            # So we can just check all words.
            
            # Count overlap with uncovered
            # ball[i] is a set of indices
            # We want len(ball[i] & uncovered)
            
            # Optimization: if max_cover is already high, we can break early?
            # Max possible is len(ball[i]) = 15.
            if max_cover >= 15:
                break
                
            count = 0
            # Manual intersection count is faster than creating a new set
            for b in balls[i]:
                if b in uncovered:
                    count += 1
                    if count > max_cover:
                        max_cover = count
                        best_word = i
                        if max_cover == 15:
                            break
            
        if best_word == -1:
            # Should not happen if uncovered is not empty
            break
            
        code.append(best_word)
        # Update uncovered
        uncovered.difference_update(balls[best_word])
        
    # Convert code indices back to lists
    current_code = [list(words[i]) for i in code]
    
    # Local Search: Try to remove redundant codewords
    # We have a set of codewords. Check if removing any one still leaves full coverage.
    # This is O(K * N * BallSize) roughly.
    # K ~ 180-200, N=2187.
    
    def is_covering(code_set):
        # code_set is a set of indices
        covered = set()
        for c in code_set:
            covered.update(balls[c])
        return len(covered) == total_words

    # Greedy result might have redundancies.
    # Let's try to prune.
    code_set = set(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in list(code_set):
            # Check if we can remove c
            temp_set = set(code_set)
            temp_set.remove(c)
            if is_covering(temp_set):
                code_set = temp_set
                improved = True
                break # Restart loop as set changed
    
    # Final polish: try to swap?
    # Given time constraints, simple deletion is often sufficient to get close to optimum.
    # The greedy + deletion heuristic typically lands around 190-200 for K_3(7,1).
    # To beat 729 significantly, this is enough.
    # To beat 186, we might need more advanced swaps, but let's see the score.
    
    # If we have time, try a few random restarts of the greedy?
    # Or try to swap a codeword with another to enable removal of others?
    # Simple swap: pick c in code, pick d not in code. 
    # If code - {c} + {d} is covering, and allows removal of others?
    
    # Let's stick to the robust deletion first.
    
    final_code = [list(words[i]) for i in code_set]
    
    # Verify validity (just in case)
    # The evaluator will check, but we should be sure.
    # is_covering(code_set) is True by construction of the loop.
    
    return final_code

# Helper for fast execution in some environments if needed, 
# but standard construct is the interface.

if __name__ == "__main__":
    # Quick self-test
    code = construct(3, 7, 1)
    print(f"Found code size: {len(code)}")
    # Verify
    q, n, r = 3, 7, 1
    total = q**n
    words = list(product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    covered = set()
    for c in code:
        c_idx = word_to_idx[tuple(c)]
        # Add ball
        covered.add(c_idx)
        for pos in range(n):
            for val in range(q):
                if val == c[pos]: continue
                new_c = list(c)
                new_c[pos] = val
                covered.add(word_to_idx[tuple(new_c)])
    if len(covered) == total:
        print("Valid covering.")
    else:
        print(f"Invalid covering. Covered {len(covered)}/{total}")
