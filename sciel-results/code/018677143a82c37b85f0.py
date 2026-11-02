import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
    
    def to_word(val):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    # Precompute all balls as a numpy array: balls[i] contains the indices in the ball of word i
    # For q=3, n=8, radius=1: ball size = 1 + 8*2 = 17
    def build_balls():
        balls_list = []
        for w in range(total):
            word = to_word(w)
            ball = [w]
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        word[pos] = new_val
                        ball.append(to_int(word))
                word[pos] = orig
            balls_list.append(ball)
        return np.array(balls_list, dtype=np.int32)
    
    balls = build_balls()
    
    # Greedy set cover with NumPy-accelerated coverage counting
    covered = np.zeros(total, dtype=np.bool_)
    covered_count = 0
    codewords = []
    
    # Precompute: for each word, its ball as a set for fast lookup? No, use numpy
    # Greedy: repeatedly pick the word that covers the most uncovered points
    # To speed up: maintain a "gain" array
    
    # Initial gain: for each candidate word, how many uncovered points it covers
    # We'll use a different approach: maintain covered set, and for each step compute gains
    
    while covered_count < total:
        if time.time() > deadline:
            break
        
        # Compute gain for each word: number of uncovered points in its ball
        # balls is (total, ball_size) array
        # covered is (total,) bool array
        # gain[w] = sum over x in balls[w] of (not covered[x])
        
        # Vectorized: for each word w, gain[w] = ball_size - sum(covered[balls[w]])
        # covered[balls[w]] is (total, ball_size) boolean array
        # Sum along axis 1 gives number of covered points in each ball
        covered_in_balls = covered[balls]  # (total, ball_size) boolean
        covered_count_per_ball = covered_in_balls.sum(axis=1)  # (total,)
        gains = ball_size - covered_count_per_ball  # (total,)
        
        best_word = int(np.argmax(gains))
        best_new = int(gains[best_word])
        
        if best_new == 0:
            break
        
        codewords.append(best_word)
        ball_points = balls[best_word]
        newly_covered = ~covered[ball_points]
        if newly_covered.any():
            covered[ball_points] = True
            covered_count += int(newly_covered.sum())
    
    # Try to remove redundant codewords
    if time.time() < deadline:
        # Build coverage count: for each word, how many codewords cover it
        cov_count = np.zeros(total, dtype=np.int32)
        for cw in codewords:
            cov_count[balls[cw]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # Check if removing cw leaves all its covered words still covered
                if (cov_count[balls[cw]] > 1).all():
                    # Remove cw
                    cov_count[balls[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
    
    result = [to_word(w) for w in codewords]
    
    # Ensure distinct
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    
    return unique_result
