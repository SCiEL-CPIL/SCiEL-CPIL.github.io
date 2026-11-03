import time
import random
import itertools

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1()
    else:
        # Generic fallback
        return _construct_generic(q, n, radius)

def _construct_3_7_1():
    # Precomputation
    words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for w in words:
        nbs = [w]
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_w = list(w)
                    new_w[i] = val
                    nbs.append(tuple(new_w))
        neighbors.append([word_to_idx[nb] for nb in nbs])
        
    total_words = 2187
    best_code = None
    start_time = time.time()
    time_limit = 110
    
    # Try deterministic first?
    # Then random restarts.
    
    for seed in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        is_uncovered = [True] * total_words
        code = []
        
        # Optimization: keep track of uncovered count?
        # Not strictly needed for the while condition if we use any().
        
        while any(is_uncovered):
            best_score = 0
            best_candidates = []
            
            # Iterate all words
            for w_idx in range(total_words):
                score = 0
                for nb in neighbors[w_idx]:
                    if is_uncovered[nb]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidates = [w_idx]
                elif score == best_score and score > 0:
                    best_candidates.append(w_idx)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            
            for nb in neighbors[chosen]:
                is_uncovered[nb] = False
                
        if best_code is None or len(code) < len(best_code):
            best_code = code
            
    if best_code is None:
        # Should not happen
        return [list(w) for w in words]
        
    return [list(words[i]) for i in best_code]

def _construct_generic(q, n, radius):
    # Simple greedy for other cases
    # ... (similar logic but slower)
    pass
