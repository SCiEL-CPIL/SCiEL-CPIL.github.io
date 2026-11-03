import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    total_words = q ** n
    
    # Precompute all words as indices 0..total_words-1
    # Convert index to base-q representation and back is expensive.
    # We work with indices directly.
    
    # Precompute balls: ball[i] is a list of indices within Hamming distance <= radius of word i
    # For q=3, n=8, radius=1:
    # Word i corresponds to a tuple of n digits.
    # Neighbors: change one digit to one of the other q-1 values.
    
    # To speed up, we can generate the neighbor offsets.
    # But since n is small, we can just compute neighbors on the fly or precompute a list of lists.
    # Precomputing 6561 lists of 17 integers is fast.
    
    balls = []
    for i in range(total_words):
        # Convert i to base-q digits
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % q
            temp //= q
        
        ball = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                # Calculate new index
                # index = sum(d_j * q^j)
                # delta = (new_val - original) * q^pos
                delta = (new_val - original) * (q ** pos)
                ball.append(i + delta)
        balls.append(ball)
    
    balls = np.array(balls, dtype=np.int32)
    
    covered = np.zeros(total_words, dtype=bool)
    uncovered_count = total_words
    code = []
    
    # Greedy Set Cover
    # We need to find the word that covers the most uncovered words.
    # Efficient way:
    # Maintain an array `cover_count` where cover_count[w] is the number of uncovered words that are in ball[w].
    # Initially, cover_count[w] = number of uncovered words in ball[w]. Since all are uncovered, it's just len(ball[w]) = 17 for all w.
    # But as we add codewords, we need to update this.
    
    # Alternative:
    # While uncovered > 0:
    #   Find w that maximizes |ball[w] intersect uncovered|
    #   Add w to code
    #   Remove ball[w] from uncovered
    
    # To find w efficiently:
    # We can maintain a counter `votes` for each word.
    # For each uncovered word u, increment votes for all v in ball[u].
    # The w with max votes is the best candidate.
    # Then we select w, and for all u in ball[w], we decrement votes for all v in ball[u] (because u is no longer uncovered).
    
    votes = np.zeros(total_words, dtype=np.int32)
    
    # Initialize votes: each word u contributes 1 to its neighbors.
    # Since all are uncovered initially, votes[v] = number of u such that v in ball[u].
    # By symmetry, this is just the size of the ball, which is constant (17).
    # So we can start with votes = 17 * ones.
    # But wait, we need to update as we cover words.
    
    # Let's use a different approach for the first few steps or just do the full update.
    # Full update:
    # 1. Calculate votes for all words based on current uncovered set.
    #    This is O(U * BallSize). U starts at 6561. 6561 * 17 ~ 100k ops. Fast.
    # 2. Pick max vote word w.
    # 3. Add w to code.
    # 4. For each u in ball[w]:
    #      If u is uncovered:
    #         Mark u as covered.
    #         For each v in ball[u]:
    #             votes[v] -= 1
    #         (This removes the contribution of u to the votes of its neighbors)
    
    # This is O(|ball[w]| * |ball[u]|) per step = 17 * 17 = 289 ops.
    # Number of steps is roughly total_words / 17 ~ 400 steps.
    # Total ops ~ 400 * 289 ~ 100k. Very fast.
    
    # However, finding the max vote word takes O(total_words) = 6561 per step.
    # 400 * 6561 ~ 2.6M ops. Also very fast.
    
    # Let's implement this.
    
    # Initial votes: all words are uncovered.
    # votes[v] = number of u in Uncovered such that v in ball[u].
    # Initially Uncovered = All.
    # votes[v] = |ball[v]| = 17.
    votes = np.full(total_words, 17, dtype=np.int32)
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        # Find word with max votes
        w = np.argmax(votes)
        
        # Add w to code
        code.append(w)
        
        # Update uncovered and votes
        for u in balls[w]:
            if not covered[u]:
                covered[u] = True
                uncovered_count -= 1
                # u is now covered, so it no longer contributes to votes.
                # Decrement votes for all neighbors of u.
                for v in balls[u]:
                    votes[v] -= 1
                    
    # Pruning phase
    # Try to remove codewords that are redundant.
    # A codeword c is redundant if all words in ball[c] are covered by other codewords in the code.
    
    # We can do this by iterating and checking.
    # To check if c is redundant:
    #   For each u in ball[c]:
    #       Is u covered by some c' in code, c' != c?
    #   This is expensive if done naively.
    
    # Better:
    #   Maintain a count `cover_count[u]` = number of codewords in `code` that cover u.
    #   A word u is "safe" if cover_count[u] >= 1.
    #   If we remove c, we decrement cover_count[u] for all u in ball[c].
    #   c is redundant if min(cover_count[u] for u in ball[c]) >= 2 BEFORE removal?
    #   No, if min >= 2, then after removal min >= 1, so it's safe.
    #   If min == 1, we need to check if the unique cover is c itself.
    
    # Let's just do a simple removal loop.
    # For each c in code:
    #   Check if all u in ball[c] have cover_count[u] > 1 OR (cover_count[u] == 1 and the cover is not c).
    #   Actually, simpler:
    #   Temporarily remove c.
    #   Check if any u in ball[c] becomes uncovered (cover_count[u] becomes 0).
    #   If no, keep it removed. Else, put it back.
    
    # Let's maintain cover_count array.
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        for u in balls[c]:
            cover_count[u] += 1
            
    # Now try to remove
    new_code = []
    for c in code:
        if time.time() > deadline:
            new_code.append(c)
            continue
            
        # Check if c is redundant
        redundant = True
        for u in balls[c]:
            if cover_count[u] == 1:
                # Only covered by c?
                # We need to verify that c is indeed the one covering it.
                # Since we are iterating code, and we haven't removed c yet from the "logical" set for this check,
                # if cover_count[u] is 1, it must be c, unless we already removed a previous codeword that also covered u.
                # Wait, if we removed a previous codeword c_prev that covered u, cover_count[u] would have decreased.
                # So if cover_count[u] is 1 now, and c is in the current `code` list being processed,
                # it's highly likely c is the sole coverer.
                # To be safe, we can just say: if cover_count[u] <= 1, we can't remove c.
                # Because if cover_count[u] == 1, it's covered by exactly one codeword in the *remaining* set (including c).
                # If we remove c, it becomes 0. So c is NOT redundant.
                redundant = False
                break
                
        if redundant:
            # Remove c
            for u in balls[c]:
                cover_count[u] -= 1
        else:
            new_code.append(c)
            
    code = new_code
    
    # Convert indices back to lists of integers
    result = []
    for idx in code:
        word = [0] * n
        temp = idx
        for j in range(n):
            word[j] = temp % q
            temp //= q
        result.append(word)
        
    return result
