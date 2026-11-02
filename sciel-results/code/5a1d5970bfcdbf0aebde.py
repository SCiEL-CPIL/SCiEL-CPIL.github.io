import numpy as np
import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=7, radius=1
    # We can hardcode the logic for efficiency or keep it general.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we should keep it general, but optimize for the likely case.
    
    total_words = q ** n
    
    # Generate all words as tuples of integers
    # Using itertools.product
    words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls (neighborhoods)
    # Ball size is 1 + n * (q - 1)
    # For q=3, n=7, size = 15.
    balls = []
    
    for i, w in enumerate(words):
        ball = set()
        # Center
        ball.add(i)
        # Neighbors
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                ball.add(word_to_idx[new_w])
        balls.append(ball)
        
    best_code = None
    best_size = total_words + 1
    
    # Number of restarts
    # 120 seconds limit. 
    # Each greedy run is roughly O(N * BallSize * log N) or similar.
    # N=2187. BallSize=15.
    # It should be very fast. Let's do 200 restarts.
    
    for _ in range(200):
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            # Find the word that covers the most uncovered words
            max_covered = -1
            best_candidates = []
            
            # Optimization: Only consider words that are themselves uncovered?
            # Actually, a word that is already covered might still be a good center 
            # if it covers many *other* uncovered words. 
            # However, standard greedy set cover usually picks from the universe of sets.
            # Here, the "sets" are the balls. The "elements" are the words.
            # We want to pick a set (ball) that covers the most remaining elements.
            
            # Iterating over all total_words (2187) at each step is O(N^2) total.
            # 2187^2 ~ 4.7 million. 
            # Inner loop checks intersection size.
            # len(balls[i] & uncovered)
            
            # To speed up, we can just iterate over all i.
            for i in range(total_words):
                # If the ball is already fully covered, skip? 
                # No, we just need the count of intersection.
                # If len(balls[i] & uncovered) == 0, skip.
                # 
                # Intersection calculation:
                # balls[i] is a set. uncovered is a set.
                # len(balls[i] & uncovered)
                
                # Optimization: If the current best is high, we can break early?
                # No, we need max.
                
                covered_count = len(balls[i] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_candidates = [i]
                elif covered_count == max_covered:
                    best_candidates.append(i)
                    
            if max_covered == 0:
                # This shouldn't happen if uncovered is not empty
                break
                
            # Pick one randomly from best_candidates to diversify
            chosen_idx = random.choice(best_candidates)
            code.append(words[chosen_idx])
            uncovered -= balls[chosen_idx]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
            
            # Early stopping if we hit a known good bound?
            # Best known is 186. If we hit 186, we can stop.
            if best_size <= 186:
                break
                
    return [list(c) for c in best_code]
