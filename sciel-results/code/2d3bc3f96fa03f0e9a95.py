import time
import random
import itertools

# Global cache for specific case
_N_WORDS_CACHE = 0
_NEIGHBORS_CACHE = []

def _init_specific_neighbors(q, n):
    global _N_WORDS_CACHE, _NEIGHBORS_CACHE
    _N_WORDS_CACHE = q ** n
    _NEIGHBORS_CACHE = []
    for i in range(_N_WORDS_CACHE):
        v = []
        tmp = i
        for _ in range(n):
            v.append(tmp % q)
            tmp //= q
        # v[0] is least significant digit
        
        neigh = [i]
        for pos in range(n):
            orig = v[pos]
            for new_val in range(q):
                if new_val != orig:
                    # Only for radius 1, we change 1 position
                    # If radius > 1, this logic is insufficient, but task is radius 1
                    new_v = v.copy()
                    new_v[pos] = new_val
                    new_idx = 0
                    for k in range(n):
                        new_idx += new_v[k] * (q ** k)
                    neigh.append(new_idx)
        _NEIGHBORS_CACHE.append(neigh)

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        start_time = time.time()
        deadline = start_time + 115
        
        if _N_WORDS_CACHE != 2187:
            _init_specific_neighbors(3, 7)
            
        best_code = None
        best_len = float('inf')
        
        while time.time() < deadline:
            code_indices = _greedy_random_specific()
            if len(code_indices) < best_len:
                best_len = len(code_indices)
                best_code = code_indices
                
        if best_code is None:
            return _seed_solution()
            
        # Convert indices to list of lists
        return [ternary_rep(idx, 7) for idx in best_code]
    else:
        return _general_greedy(q, n, radius)

def _greedy_random_specific():
    N = _N_WORDS_CACHE
    neigh = _NEIGHBORS_CACHE
    uncovered = [True] * N
    uncovered_count = N
    code = []
    
    while uncovered_count > 0:
        best_score = -1
        best_candidates = []
        
        # Calculate scores
        for w in range(N):
            score = 0
            for nb in neigh[w]:
                if uncovered[nb]:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidates = [w]
            elif score == best_score and score > 0:
                best_candidates.append(w)
                
        if not best_candidates:
            break
            
        # Random choice with slight noise to encourage diversity
        # Actually, just random choice from ties is good.
        w = random.choice(best_candidates)
        code.append(w)
        
        for nb in neigh[w]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1
                
    return code

def ternary_rep(idx, n):
    res = []
    for _ in range(n):
        res.append(idx % 3)
        idx //= 3
    return res

def _seed_solution():
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]

def _general_greedy(q, n, radius):
    from itertools import product
    space = list(product(range(q), repeat=n))
    space_size = q**n
    idx_map = {word: i for i, word in enumerate(space)}
    
    def dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    neighbors = []
    for w in space:
        neigh = []
        for v in space:
            if dist(w, v) <= radius:
                neigh.append(idx_map[v])
        neighbors.append(neigh)
        
    uncovered = [True] * space_size
    uncovered_count = space_size
    code = []
    
    while uncovered_count > 0:
        best_score = -1
        best_w = -1
        for w_idx in range(space_size):
            score = sum(1 for nb in neighbors[w_idx] if uncovered[nb])
            if score > best_score:
                best_score = score
                best_w = w_idx
        code.append(space[best_w])
        for nb in neighbors[best_w]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1
    return code
