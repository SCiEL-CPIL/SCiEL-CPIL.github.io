import numpy as np
import time
import itertools

def construct(q, n, radius):
    deadline = time.time() + 140
    
    total_words = q ** n
    code_arr = np.zeros((n,), dtype=np.int32)
    
    # Precompute ball for any word
    def get_ball(w):
        """Return list of word indices covered by word w."""
        idx = 0
        for v in w:
            idx = idx * q + v
        ball = [idx]
        for pos in range(n):
            for alt in range(q):
                if alt != w[pos]:
                    w2 = w.copy()
                    w2[pos] = alt
                    idx2 = 0
                    for v in w2:
                        idx2 = idx2 * q + v
                    ball.append(idx2)
        return ball
    
    # Greedy set cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    while uncovered.any():
        best_idx = -1
        best_count = 0
        # Sample or iterate through all words
        for idx in range(total_words):
            if not uncovered[idx]:
                continue
            # Count how many uncovered words are in this word's ball
            # This is expensive: 17 checks per word, 6561 words
            # Total: ~111K operations per iteration
            w = np.array([idx // (q ** (n - 1 - i)) % q for i in range(n)])
            ball = get_ball(w)
            count = sum(uncovered[b] for b in ball)
            if count > best_count:
                best_count = count
                best_idx = idx
                if best_count == 17:  # max possible
                    break
        
        if best_idx < 0:
            break
        
        w = np.array([best_idx // (q ** (n - 1 - i)) % q for i in range(n)])
        code.append(w)
        ball = get_ball(w)
        for b in ball:
            uncovered[b] = False
    
    code_arr = np.array(code, dtype=np.int32)
    m = len(code_arr)
    
    # Now do redundancy removal
    # Precompute balls for all codewords
    balls = []
    for i in range(m):
        balls.append(get_ball(code_arr[i]))
    
    cov = np.zeros(total_words, dtype=np.int32)
    for i in range(m):
        for idx in balls[i]:
            cov[idx] += 1
    
    active = np.ones(m, dtype=bool)
    
    # Repeatedly remove redundant codewords
    while time.time() < deadline:
        removed = False
        for i in range(m):
            if not active[i]:
                continue
            if np.all(cov[balls[i]] > 1):
                active[i] = False
                for idx in balls[i]:
                    cov[idx] -= 1
                removed = True
        if not removed:
            break
    
    result = [code_arr[i].tolist() for i in range(m) if active[i]]
    return result
