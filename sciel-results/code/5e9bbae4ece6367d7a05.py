"""Football pool problem: ternary covering code K_3(8,1)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total = q ** n  # 6561 for q=3, n=8
    
    # Precompute ball: for each word, list of words within Hamming distance <= radius
    # For radius=1, ball size = 1 + n*(q-1) = 1 + 8*2 = 17
    ball_indices = []
    for i in range(total):
        w = all_words[i]
        ball = {i}
        for pos in range(n):
            orig = w[pos]
            for alt in range(q):
                if alt != orig:
                    new_w = w.copy()
                    new_w[pos] = alt
                    # Find index via encoding
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_w[j]
                    ball.add(idx)
        ball_indices.append(ball)
    
    # Convert to lists for faster access
    ball_indices = [list(s) for s in ball_indices]
    
    # Greedy: iteratively pick the word that covers the most uncovered words
    covered = np.zeros(total, dtype=bool)
    uncovered_count = total
    centers = []
    
    while uncovered_count > 0 and time.time() < deadline - 30:
        best_idx = -1
        best_gain = 0
        # Sample a subset of uncovered words to speed up
        uncov_indices = np.where(~covered)[0]
        if len(uncov_indices) > 200:
            sample = np.random.choice(uncov_indices, 200, replace=False)
        else:
            sample = uncov_indices
        
        for cand in sample:
            gain = sum(1 for b in ball_indices[cand] if not covered[b])
            if gain > best_gain:
                best_gain = gain
                best_idx = cand
        
        if best_idx < 0:
            break
        
        for b in ball_indices[best_idx]:
            if not covered[b]:
                covered[b] = True
                uncovered_count -= 1
        centers.append(best_idx)
    
    # Local search: try to remove centers if still covered
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(len(centers) - 1, -1, -1):
            # Temporarily remove center i
            temp_centers = centers[:i] + centers[i+1:]
            # Check if all words still covered
            still_covered = np.zeros(total, dtype=bool)
            for c in temp_centers:
                for b in ball_indices[c]:
                    still_covered[b] = True
            if still_covered.all():
                centers = temp_centers
                improved = True
    
    # Return as list of lists
    return [all_words[c].tolist() for c in centers]
# EVOLVE-BLOCK-END
