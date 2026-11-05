"""Max K(2,2,2)-free triple system on 16 points."""
import time
import random
import itertools

def construct():
    deadline = time.time() + 115
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    n = len(all_triples)
    t2i = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflict patterns
    triple_conflicts = [[] for _ in range(n)]
    for i, T in enumerate(all_triples):
        a, b, c = T
        others = [x for x in range(N) if x not in (a, b, c)]
        for d, e, f in itertools.combinations(others, 3):
            for perm in itertools.permutations([d, e, f]):
                d1, d2, d3 = perm
                transversals = []
                for x in (a, d1):
                    for y in (b, d2):
                        for z in (c, d3):
                            t = tuple(sorted((x, y, z)))
                            if t != T:
                                transversals.append(t2i[t])
                triple_conflicts[i].append(tuple(transversals))
    
    best_set = set()
    best_count = 0
    
    for restart in range(1000):
        if time.time() > deadline - 3:
            break
        
        random.seed(restart)
        order = list(range(n))
        random.shuffle(order)
        
        current = set()
        
        for idx in order:
            conflict = False
            for pattern in triple_conflicts[idx]:
                all_in = True
                for j in pattern:
                    if j not in current:
                        all_in = False
                        break
                if all_in:
                    conflict = True
                    break
            if not conflict:
                current.add(idx)
        
        if len(current) > best_count:
            best_count = len(current)
            best_set = set(current)
        
        if time.time() > deadline - 10:
            break
    
    return [list(all_triples[i]) for i in sorted(best_set)]
