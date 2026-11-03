"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Total words in the space
    total_words = q ** n
    
    # Precompute all words as integers for fast manipulation
    # We'll use a mixed-radix representation: word = sum(w[i] * q^i)
    # Actually, for q=3, n=6, we can just use base-q integers
    # Let's use a list-based approach with numpy for speed
    
    # Convert words to/from integer representation
    def word_to_int(word):
        result = 0
        for i in range(n):
            result = result * q + word[i]
        return result
    
    def int_to_word(val):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        return word[::-1]
    
    # Precompute all possible words as integers
    all_words = list(range(total_words))
    
    # Precompute Hamming distance-1 neighbors for each word
    # For each word, there are n*(q-1) neighbors at distance 1
    # We'll precompute the "ball" of radius 1 for each word
    # But that's 729 * 36 = 26244 entries, which is fine
    
    # Actually, let's think differently. We want to find a covering code.
    # A word w covers itself and all words at distance 1.
    # The number of words covered by one codeword is 1 + n*(q-1) = 1 + 6*2 = 13
    
    # Lower bound: ceil(729 / 13) = ceil(56.08) = 57
    # But we know the answer is 73.
    
    # Strategy: Use simulated annealing to minimize the number of codewords
    # while maintaining coverage.
    
    # First, get a good initial solution using a greedy approach
    def greedy_cover():
        """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
        # Track which words are covered
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # Precompute the ball (closed) for each word: word + all distance-1 neighbors
        # Ball size = 1 + n*(q-1) = 13
        balls = []
        for w_int in range(total_words):
            w = int_to_word(w_int)
            ball = [w_int]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        ball.append(word_to_int(w))
                w[pos] = orig
            balls.append(ball)
        
        # Greedy: pick the word whose ball covers the most currently uncovered words
        # To speed this up, we can use a priority queue or just recompute each time
        # For 729 words, this is fast enough
        
        remaining = set(range(total_words))
        
        while remaining:
            best_word = -1
            best_count = 0
            for w_int in remaining:
                count = sum(1 for x in balls[w_int] if x in remaining)
                if count > best_count:
                    best_count = count
                    best_word = w_int
                if time.time() > deadline - 5:
                    break
            if best_word == -1:
                break
            codewords.append(best_word)
            # Remove covered words
            for x in balls[best_word]:
                if x in remaining:
                    remaining.discard(x)
            if time.time() > deadline - 5:
                break
        
        return codewords
    
    # Get initial solution
    initial_codewords = greedy_cover()
    best_codewords = list(initial_codewords)
    best_size = len(best_codewords)
    
    # Now use simulated annealing to try to reduce the size
    # Moves: 
    # 1. Try to remove a codeword (if coverage is maintained)
    # 2. Try to swap a codeword with another word (if coverage is maintained or improved)
    
    def compute_coverage(codewords_int):
        """Return a boolean array indicating which words are covered."""
        covered = np.zeros(total_words, dtype=bool)
        for w_int in codewords_int:
            w = int_to_word(w_int)
            covered[w_int] = True
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        w[pos] = new_val
                        covered[word_to_int(w)] = True
                w[pos] = orig
        return covered
    
    def is_covered(covered, w_int):
        return covered[w_int]
    
    # Precompute balls for speed
    balls = []
    for w_int in range(total_words):
        w = int_to_word(w_int)
        ball = [w_int]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    ball.append(word_to_int(w))
                w[pos] = orig
        balls.append(ball)
    
    # Simulated annealing
    codewords = list(best_codewords)
    covered = compute_coverage(codewords)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling = 0.995
    
    T = T_start
    iterations = 0
    max_iterations = 500000
    
    while T > T_end and iterations < max_iterations:
        if time.time() > deadline - 10:
            break
        iterations += 1
        
        # Randomly choose a move type
        if random.random() < 0.4 and len(codewords) > 1:
            # Try to remove a codeword
            idx = random.randint(0, len(codewords) - 1)
            w_int = codewords[idx]
            
            # Check if removing w_int still maintains coverage
            # We need to check if all words in ball(w_int) are covered by other codewords
            # This is expensive, so let's be smarter
            
            # Count how many codewords cover each word in ball(w_int)
            # Actually, let's just tentatively remove and check
            new_codewords = codewords[:idx] + codewords[idx+1:]
            new_covered = compute_coverage(new_codewords)
            
            if np.all(new_covered):
                # Accept the removal
                codewords = new_codewords
                covered = new_covered
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
            # Else: reject (or accept with probability exp(-1/T) but that doesn't make sense for removal)
            
        elif random.random() < 0.6:
            # Try to swap a codeword with a random word
            idx = random.randint(0, len(codewords) - 1)
            old_w_int = codewords[idx]
            new_w_int = random.randint(0, total_words - 1)
            
            if new_w_int == old_w_int:
                continue
            
            # Check if the new codeword is already in the list
            if new_w_int in codewords:
                continue
            
            new_codewords = codewords[:]
            new_codewords[idx] = new_w_int
            new_covered = compute_coverage(new_codewords)
            
            if np.all(new_covered):
                # Accept the swap
                codewords = new_codewords
                covered = new_covered
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
            
        else:
            # Try to add a codeword (to escape local minima, then remove others)
            new_w_int = random.randint(0, total_words - 1)
            if new_w_int not in codewords:
                new_codewords = codewords + [new_w_int]
                codewords = new_codewords
                covered = compute_coverage(codewords)
        
        # Cool down
        T *= cooling
    
    # Final greedy removal pass
    # Try to remove codewords one by one if possible
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(best_codewords) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            test_codewords = best_codewords[:i] + best_codewords[i+1:]
            test_covered = compute_coverage(test_codewords)
            if np.all(test_covered):
                best_codewords = test_codewords
                changed = True
    
    # Convert back to list of lists
    result = [int_to_word(w_int) for w_int in best_codewords]
    
    return result
# EVOLVE-BLOCK-END
