import random
from itertools import combinations

def construct():
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    best_family = []
    
    for seed in range(50):
        random.seed(seed)
        family = []
        # Precompute for fast checking
        # For each pair in family, store their intersection
        # When adding C, check all pairs
        
        for s in random.sample(all_sets, len(all_sets)):
            ok = True
            for i in range(len(family)):
                for j in range(i+1, len(family)):
                    a, b = family[i], family[j]
                    if a & b == a & s == b & s:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                family.append(s)
        
        if len(family) > len(best_family):
            best_family = family
    
    return [sorted(s) for s in best_family]
