import time
import random
from itertools import combinations

def _independence_number_bitmask(triples):
    """Compute independence number using bitmask precomputation."""
    n = 16
    total_masks = 1 << n
    triple_masks = [((1 << t[0]) | (1 << t[1]) | (1 << t[2])) for t in triples]
    
    # Mark all non-independent masks
    is_independent = [True] * total_masks
    for tm in triple_masks:
        remaining = (total_masks - 1) ^ tm
        sub = remaining
        while True:
            is_independent[tm | sub] = False
            if sub == 0:
                break
            sub = (sub - 1) & remaining
    
    # Find max popcount among independent masks
    best = 0
    for mask in range(total_masks):
        if is_independent[mask]:
            c = bin(mask).count('1')
            if c > best:
                best = c
    return best


def _is_linear(triples):
    """Check linearity: any two triples share at most one point."""
    for i in range(len(triples)):
        si = set(triples[i])
        for j in range(i + 1, len(triples)):
            if len(si & set(triples[j])) > 1:
                return False
    return True


def _greedy_construct(seed=None):
    """Greedy construction with randomization."""
    if seed is not None:
        random.seed(seed)
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    triples = []
    for t in all_triples:
        ok = True
        tset = set(t)
        for existing in triples:
            if len(tset & set(existing)) > 1:
                ok = False
                break
        if ok:
            triples.append(list(t))
    return triples


def _local_search(triples, time_limit=60):
    """Local search to minimize independence number."""
    start_time = time.time()
    best_triples = [list(t) for t in triples]
    best_alpha = _independence_number_bitmask(best_triples)
    
    all_triples = list(combinations(range(16), 3))
    
    def can_add(triples_list, new_triple):
        new_set = set(new_triple)
        for t in triples_list:
            if len(new_set & set(t)) > 1:
                return False
        return True
    
    for iteration in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        current_alpha = _independence_number_bitmask(best_triples)
        if current_alpha < best_alpha:
            best_alpha = current_alpha
        
        # Try adding triples
        improved = False
        for _ in range(30):
            t = random.choice(all_triples)
            if can_add(best_triples, t):
                new_triples = best_triples + [list(t)]
                new_alpha = _independence_number_bitmask(new_triples)
                if new_alpha < best_alpha:
                    best_triples = new_triples
                    best_alpha = new_alpha
                    improved = True
                    break
                elif new_alpha == best_alpha and random.random() < 0.2:
                    best_triples = new_triples
                    improved = True
                    break
        
        if not improved and best_triples:
            # Try swap
            for _ in range(15):
                idx = random.randint(0, len(best_triples) - 1)
                temp = best_triples[:idx] + best_triples[idx+1:]
                for _ in range(30):
                    t = random.choice(all_triples)
                    if can_add(temp, t):
                        new_triples = temp + [list(t)]
                        new_alpha = _independence_number_bitmask(new_triples)
                        if new_alpha < best_alpha:
                            best_triples = new_triples
                            best_alpha = new_alpha
                            break
                        elif new_alpha == best_alpha and random.random() < 0.15:
                            best_triples = new_triples
                            break
    
    return best_triples, best_alpha


def construct():
    deadline = time.time() + 140
    best_result = None
    best_alpha = 100
    
    # Initial greedy
    triples = _greedy_construct(seed=42)
    alpha = _independence_number_bitmask(triples)
    best_alpha = alpha
    best_result = [list(t) for t in triples]
    
    # Local search on initial
    remaining = deadline - time.time() - 5
    if remaining > 10:
        triples_ls, alpha_ls = _local_search(triples, time_limit=min(remaining, 30))
        if alpha_ls < best_alpha:
            best_alpha = alpha_ls
            best_result = [list(t) for t in triples_ls]
    
    # Multiple restarts
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        seed = restart * 137 + 3
        triples = _greedy_construct(seed=seed)
        alpha = _independence_number_bitmask(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_result = [list(t) for t in triples]
        
        # Local search
        remaining = deadline - time.time() - 5
        if remaining > 5:
            triples_ls, alpha_ls = _local_search(triples, time_limit=min(remaining, 15))
            if alpha_ls < best_alpha:
                best_alpha = alpha_ls
                best_result = [list(t) for t in triples_ls]
    
    if best_result is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    if not _is_linear(best_result):
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_result
