import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute the ball (neighbors within Hamming distance 'radius') for each word
    # For radius=1, ball size is 1 + n*(q-1)
    # We store balls as lists of word indices
    balls = []
    for w in range(total_words):
        # Convert w to base-q representation
        coords = []
        tmp = w
        for _ in range(n):
            coords.append(tmp % q)
            tmp //= q
        coords = tuple(coords)
        
        ball = set([w])
        # Generate all neighbors at distance 1
        for i in range(n):
            for v in range(q):
                if v == coords[i]:
                    continue
                new_coords = list(coords)
                new_coords[i] = v
                new_w = 0
                for j in range(n):
                    new_w = new_w * q + new_coords[j]
                ball.add(new_w)
        balls.append(list(ball))
        
    best_code = None
    best_size = total_words # Upper bound is trivially all words
    
    # We will run multiple restarts
    # Estimate max restarts based on time
    # Each restart involves O(total_words * len(code)) operations roughly
    # For small n, this is fast. Let's just loop until deadline.
    
    while time.time() < deadline:
        # Greedy Set Cover with randomization
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # To speed up, we can precompute for each word, which balls cover it?
        # Or just iterate. Given 729 words, iterating is fine.
        
        # Optimization: Keep track of "potential" for each word (how many new words it covers)
        # Initially, every word covers len(balls[i]) new words (if uncovered)
        # But we need to be careful with overlaps.
        
        # Simple approach:
        # 1. Start with all words as candidates.
        # 2. Pick the one that covers the most uncovered words.
        # 3. Mark those as covered.
        # 4. Repeat until all covered.
        
        # To make it efficient, we can maintain a count of how many times each word is covered by the current code.
        # But for greedy "max marginal gain", we need to know which specific words are still uncovered.
        
        # Let's use a list of candidates.
        candidates = list(range(total_words))
        
        while not np.all(covered):
            best_gain = -1
            best_candidates = []
            
            # Iterate through all candidates to find the one with max gain
            # This is O(total_words * ball_size) per step.
            # 729 * 13 ~ 9000 ops per step.
            # Code size ~ 73. Steps ~ 73. Total ~ 650k ops per restart.
            # Very fast.
            
            for c in candidates:
                # Count how many words in balls[c] are not yet covered
                gain = 0
                # Optimization: if covered[balls[c]] are mostly True, gain is small.
                # We can break early if gain + remaining_possible < best_gain?
                # Simpler: just sum.
                gain = np.sum(~covered[balls[c]])
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if best_gain == 0:
                # This shouldn't happen if we haven't covered everything, unless logic error
                break
                
            # Randomly pick from best_candidates to introduce diversity
            chosen = random.choice(best_candidates)
            code.append(chosen)
            
            # Update covered
            covered[balls[chosen]] = True
            
            # Optional: Remove 'chosen' from candidates? 
            # No, we might pick it again? No, it's already in code. 
            # We can remove it to save time in next iteration? 
            # Actually, once picked, its marginal gain is 0. 
            # We can keep it in candidates but it will have gain 0.
            # Or remove it. Removing is safer/faster.
            candidates.remove(chosen)
            
        # Local Improvement: Remove redundant codewords
        # A codeword is redundant if all words in its ball are covered by the rest of the code.
        changed = True
        while changed:
            changed = False
            for i in range(len(code)):
                # Check if code[i] is redundant
                # Temporarily remove it? 
                # Easier: Check if union of balls of all other codewords covers ball of code[i].
                # Or: Check if every word in ball[code[i]] is covered by at least one other codeword in code.
                
                # Let's check if code[i] is essential.
                # For each word w in ball[code[i]], is there another c in code (c != code[i]) such that w in ball[c]?
                is_redundant = True
                for w in balls[code[i]]:
                    # Is w covered by any other codeword?
                    found = False
                    for j in range(len(code)):
                        if i == j:
                            continue
                        if w in balls[code[j]]:
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                    
                if is_redundant:
                    code.pop(i)
                    changed = True
                    break # Restart loop as indices changed
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we hit the theoretical lower bound or known optimum, we can stop early?
            # Known optimum is 73. Lower bound 71.
            if best_size <= 73:
                # We can't get better than 71. If we get 71, we are done.
                # If we get 73, we might still try for 72/71.
                pass

    # Convert best_code indices back to lists of integers
    if best_code is None:
        # Fallback: return all words (should not happen)
        best_code = list(range(total_words))
        
    result = []
    for idx in best_code:
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse() # Wait, the conversion above was least significant first?
        # Let's re-verify the conversion.
        # In the ball generation:
        # coords = []
        # tmp = w
        # for _ in range(n):
        #     coords.append(tmp % q)
        #     tmp //= q
        # This gives coords[0] as least significant digit.
        # new_w = 0
        # for j in range(n):
        #     new_w = new_w * q + new_coords[j]
        # This reconstructs w from coords where coords[0] is the lowest place value.
        # So the list `coords` is [d0, d1, ..., dn-1] where d0 is units.
        # The problem statement says "list of n integers". Usually position 0 is the first digit.
        # Does the order matter for Hamming distance? No.
        # Does the order matter for the evaluator? 
        # "Each codeword is a list of n integers in 0..q-1"
        # The evaluator checks Hamming distance. As long as the mapping is consistent, it's fine.
        # However, standard representation is usually most significant first or just index order.
        # My `coords` list is [least_sig, ..., most_sig].
        # I should probably reverse it to be [most_sig, ..., least_sig] or just keep it consistent.
        # Let's stick to the order produced by the conversion used in ball generation to ensure consistency if I were to compare.
        # But for the output, I just need a list of integers.
        # Let's reverse to make it look like standard base-q expansion (most significant first).
        word.reverse()
        result.append(word)
        
    return result
