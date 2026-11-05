import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 135.0
    n = 16
    
    def is_linear(triples, new_triple):
        new_set = set(new_triple)
        for t in triples:
            if len(new_set & set(t)) > 1:
                return False
        return True
    
    def compute_alpha(triples):
        if not triples:
            return n
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        for s in range(1, n + 1):
            for subset in combinations(range(n), s):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                all_hit = True
                for tm in triple_masks:
                    if smask & tm == 0:
                        all_hit = False
                        break
                if all_hit:
                    return n - s
        return 0
    
    all_triples = [list(t) for t in combinations(range(n), 3)]
    all_triple_tuples = [tuple(t) for t in all_triples]
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        triples = []
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        for t in shuffled:
            if is_linear(triples, t):
                triples.append(t)
        
        return triples
    
    best_result = None
    best_score = n + 1
    
    # Phase 1: Multiple greedy runs
    for seed in range(80):
        if time.time() - start_time > DEADLINE:
            break
        
        triples = greedy_construct(seed)
        alpha = compute_alpha(triples)
        
        if alpha < best_score:
            best_score = alpha
            best_result = triples[:]
        
        if best_score <= 5:
            break
    
    # Phase 2: Local search on best
    if best_result is not None and time.time() - start_time < DEADLINE - 3:
        rng = random.Random(999)
        current = best_result[:]
        current_alpha = compute_alpha(current)
        best_ls = current_alpha
        best_ls_triples = current[:]
        
        for iteration in range(30):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            if not current:
                break
            
            idx = rng.randrange(len(current))
            removed = current.pop(idx)
            
            # Try replacements
            candidates = rng.sample(all_triples, min(200, len(all_triples)))
            found_improvement = False
            for t in candidates:
                if t == removed:
                    continue
                if is_linear(current, t):
                    current.append(t)
                    new_alpha = compute_alpha(current)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        found_improvement = True
                        break
                    else:
                        current.pop()
            
            if not found_improvement:
                current.append(removed)
            
            if current_alpha < best_ls:
                best_ls = current_alpha
                best_ls_triples = current[:]
        
        if best_ls < best_score:
            best_score = best_ls
            best_result = best_ls_triples
    
    if best_result is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_result
