import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    total_words = q ** n
    if total_words > 100000:
        return []
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    ball_size = 1 + n * (q - 1)
    
    balls = np.zeros((total_words, ball_size), dtype=np.int32)
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    for i in range(total_words):
        w = all_words[i]
        neighbors = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors.append(idx)
        balls[i] = neighbors
        
    is_covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    while not np.all(is_covered):
        if time.time() > deadline - 5:
            break
        newly_covered = (~is_covered)[balls]
        scores = newly_covered.sum(axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
        codewords.append(best_idx)
        is_covered[balls[best_idx]] = True
        
    return [list(all_words[c]) for c in codewords]

def _solve_ternary_8_1():
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for return overhead
    
    # 1. Precompute balls efficiently using base-3 arithmetic
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute frozenset of neighbors for fast intersection in swap logic
    # This is critical for the 1-swap logic in local search
    ball_frozensets = [frozenset(balls[w].tolist()) for w in range(total)]
    
    # Precompute the inverse mapping: for each word, which codewords cover it?
    # This is balls itself, but we need it as a list of lists for fast access
    # Actually, balls[w] already gives us the codewords that cover w.
    
    def run_greedy_and_refine(random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Greedy Initialization
        is_covered = np.zeros(total, dtype=bool)
        codewords = []
        
        # Optimization: Precompute the "uncovered count" for each ball
        # uncovered_count[b] = number of words in ball[b] that are not yet covered
        uncovered_count = np.zeros(total, dtype=np.int32)
        for w in range(total):
            uncovered_count[balls[w]] += 1  # Initially all are uncovered
        
        while not np.all(is_covered):
            if time.time() > deadline - 10:
                break
            
            # Find word covering most uncovered
            # We need to find b that maximizes uncovered_count[b]
            # But uncovered_count changes as we add codewords
            # We can maintain it incrementally
            
            # For speed, let's just recompute scores periodically or use a simpler heuristic
            # Actually, the original approach of computing scores from scratch is O(total * ball_size)
            # which is 6561 * 17 ~ 111k operations per step. With ~400 steps, that's 44M ops.
            # This should be fast enough in NumPy.
            
            scores = (~is_covered)[balls].sum(axis=1)
            max_score = np.max(scores)
            
            if max_score == 0:
                break
            
            # Pick randomly among best to add diversity
            candidates = np.where(scores == max_score)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            codewords.append(best_idx)
            is_covered[balls[best_idx]] = True
            
        if len(codewords) == 0:
            return [], 0
            
        # Local Search
        # cover_counts[w] = number of codewords covering w
        cover_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            cover_counts[balls[c]] += 1
            
        improved = True
        iterations = 0
        max_iterations = 2000
        
        while improved and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Phase 1: Remove redundant codewords
            i = 0
            while i < len(codewords):
                c = codewords[i]
                cover_counts[balls[c]] -= 1
                if not np.any(cover_counts == 0):
                    codewords.pop(i)
                    improved = True
                else:
                    cover_counts[balls[c]] += 1
                    i += 1
                    
            if not improved:
                # Phase 2: Swaps to enable removal
                # Try swapping each codeword with a non-codeword to see if it allows a removal
                swap_found = False
                for i in range(len(codewords)):
                    if swap_found:
                        break
                    c = codewords[i]
                    
                    # Find words uniquely covered by c
                    unique_mask = (cover_counts == 1)
                    unique_in_ball = unique_mask[balls[c]]
                    unique_words = balls[c][unique_in_ball]
                    
                    if len(unique_words) == 0:
                        continue # Should have been removed, but maybe counts changed?
                    
                    # Find candidates b (not in codewords) that cover all unique_words
                    # candidates = intersection of covers[w] for w in unique_words
                    # covers[w] = balls[w]
                    
                    # Use frozenset intersection for speed
                    candidates = ball_frozensets[unique_words[0]]
                    for w in unique_words[1:]:
                        if not candidates:
                            break
                        candidates = candidates.intersection(ball_frozensets[w])
                        
                    if not candidates:
                        continue
                        
                    # Exclude current codewords
                    current_set = set(codewords)
                    candidates -= current_set
                    
                    if not candidates:
                        continue
                        
                    # Try each candidate
                    # Randomize order of candidates?
                    candidate_list = list(candidates)
                    random.shuffle(candidate_list)
                    
                    for b in candidate_list:
                        # Tentative swap: remove c, add b
                        cover_counts[balls[c]] -= 1
                        cover_counts[balls[b]] += 1
                        
                        if not np.any(cover_counts == 0):
                            # Valid swap. Check if it enables a removal.
                            removal_found = False
                            for j in range(len(codewords)):
                                if j == i:
                                    continue
                                c_j = codewords[j]
                                cover_counts[balls[c_j]] -= 1
                                if not np.any(cover_counts == 0):
                                    # We can remove c_j
                                    removal_found = True
                                    cover_counts[balls[c_j]] += 1 # Revert check
                                    break
                                else:
                                    cover_counts[balls[c_j]] += 1
                                    
                            if removal_found:
                                # Accept swap
                                codewords[i] = b
                                improved = True
                                swap_found = True
                                break # Restart main while loop
                            else:
                                # Revert swap
                                cover_counts[balls[b]] -= 1
                                cover_counts[balls[c]] += 1
                        else:
                            # Invalid swap
                            cover_counts[balls[b]] -= 1
                            cover_counts[balls[c]] += 1
                            
        return codewords, len(codewords)

    best_codewords = None
    best_size = total
    
    # Run multiple restarts with different seeds
    seed = 42
    while time.time() < deadline - 10:
        codewords, size = run_greedy_and_refine(seed)
        if size < best_size:
            best_size = size
            best_codewords = codewords[:]
            
        seed += 1
        
        # If we are getting close to the known best (486), maybe stop?
        if best_size <= 486:
            break
            
    if best_codewords is None:
        return []
        
    return [list(all_words[c]) for c in best_codewords]
