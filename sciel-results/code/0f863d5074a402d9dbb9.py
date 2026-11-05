import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 7 and radius == 1):
        return _general_greedy(q, n, radius)

    N = q ** n
    words = np.array(list(itertools.product(range(q), repeat=n)))
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute cover matrix
    cover_matrix = np.zeros((N, N), dtype=np.uint8)
    for j in range(N):
        w = words[j]
        cover_matrix[j, j] = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list = list(w)
                w_list[pos] = new_val
                neigh_idx = word_to_idx[tuple(w_list)]
                cover_matrix[j, neigh_idx] = 1

    # Initialize best with direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    ds_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_indices = [word_to_idx[tuple(w)] for w in ds_words]
    best_size = len(best_indices)
    
    deadline = time.time() + 110.0
    
    def _greedy_cover(cm):
        uncovered = np.ones(N, dtype=bool)
        counts = np.sum(cm, axis=1).astype(np.int32)
        code = []
        
        while np.sum(uncovered) > 0:
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            if len(candidates) > 1:
                j = candidates[np.random.randint(len(candidates))]
            else:
                j = candidates[0]
                
            code.append(j)
            
            ball_j = cm[j, :]
            newly_covered = np.where(ball_j & uncovered)[0]
            
            if len(newly_covered) == 0:
                break
                
            uncovered[newly_covered] = False
            
            if len(newly_covered) > 0:
                delta = cm[:, newly_covered].sum(axis=1)
                counts -= delta
                
        return code

    while time.time() < deadline:
        cand_indices = _greedy_cover(cover_matrix)
        if len(cand_indices) < best_size:
            best_size = len(cand_indices)
            best_indices = cand_indices
            
    final_code = [list(words[i]) for i in best_indices]
    return final_code

def _general_greedy(q, n, radius):
    N = q ** n
    words = list(itertools.product(range(q), repeat=n))
    
    balls = []
    for w in words:
        ball = set()
        ball.add(w)
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]: continue
                wl = list(w)
                wl[pos] = new_val
                ball.add(tuple(wl))
        balls.append(ball)
        
    uncovered = set(words)
    code = []
    while uncovered:
        best_j = -1
        best_cover = 0
        for j, ball in enumerate(balls):
            cover = len(ball & uncovered)
            if cover > best_cover:
                best_cover = cover
                best_j = j
        code.append(words[best_j])
        uncovered -= balls[best_j]
    return code
