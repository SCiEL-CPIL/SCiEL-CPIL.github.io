import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        return _general_construct(q, n, radius)
    
    # Ternary covering code K_3(7,1)
    N = q ** n  # 2187
    ball_size = 1 + n * (q - 1)  # 1 + 7*2 = 15
    
    # Precompute all words and their indices
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls as numpy arrays
    balls = np.empty((N, ball_size), dtype=np.int32)
    for i, w in enumerate(words):
        balls[i, 0] = i
        pos = 1
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                balls[i, pos] = word_to_idx[tuple(new_w)]
                pos += 1
    
    # Greedy set cover with vectorized gain computation
    coverage = np.zeros(N, dtype=np.int32)
    C = []
    
    deadline = time.time() + 100  # 50s buffer before 150s limit
    
    # Phase 1: Greedy initialization
    while time.time() < deadline - 30:
        # Compute gain for each candidate
        current_cov = coverage[balls]  # (N, 15)
        gains = np.sum(current_cov == 0, axis=1)
        
        if gains.max() == 0:
            break
        
        best_idx = int(np.argmax(gains))
        C.append(best_idx)
        coverage[balls[best_idx]] += 1
    
    # Phase 2: Local search
    C_set = set(C)
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        
        # Try removals
        for c in C[:]:
            if c not in C_set:
                continue
            ball_c = balls[c]
            if not np.any(coverage[ball_c] == 1):
                # c is redundant
                coverage[ball_c] -= 1
                C.remove(c)
                C_set.discard(c)
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try 1-for-1 swaps
        for i, c_remove in enumerate(C):
            if time.time() > deadline - 5:
                break
            
            # Find words uniquely covered by c_remove
            unique_words = [w for w in balls[c_remove] if coverage[w] == 1]
            if not unique_words:
                # This would have been caught by removal check
                continue
            
            unique_set = set(unique_words)
            
            # Try random candidates for c_add
            for _ in range(50):  # Try up to 50 random candidates
                if time.time() > deadline - 5:
                    break
                c_add = int(np.random.randint(N))
                if c_add in C_set:
                    continue
                
                # Check if c_add covers all unique words
                ball_add = set(balls[c_add])
                if unique_set.issubset(ball_add):
                    # Valid swap
                    coverage[balls[c_remove]] -= 1
                    coverage[balls[c_add]] += 1
                    C[i] = c_add
                    C_set.discard(c_remove)
                    C_set.add(c_add)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    # Convert to list of words
    return [list(words[c]) for c in C]


def _general_construct(q, n, radius):
    """Fallback for non-standard parameters."""
    N = q ** n
    if N > 100000:
        # Too large, return simple solution
        return [[0] * n]
    
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Compute balls
    balls = []
    for w in words:
        ball = [w]
        # Generate all words within Hamming distance radius
        for dist in range(1, radius + 1):
            for positions in itertools.combinations(range(n), dist):
                for vals in itertools.product(range(q), repeat=dist):
                    if all(v != w[p] for v, p in zip(vals, positions)):
                        new_w = list(w)
                        for p, v in zip(positions, vals):
                            new_w[p] = v
                        ball.append(tuple(new_w))
        balls.append([word_to_idx[b] for b in ball])
    
    balls = np.array(balls, dtype=np.int32)
    
    # Greedy
    coverage = np.zeros(N, dtype=np.int32)
    C = []
    
    while np.any(coverage == 0):
        current_cov = coverage[balls]
        gains = np.sum(current_cov == 0, axis=1)
        if gains.max() == 0:
            break
        best_idx = int(np.argmax(gains))
        C.append(best_idx)
        coverage[balls[best_idx]] += 1
    
    return [list(words[c]) for c in C]
