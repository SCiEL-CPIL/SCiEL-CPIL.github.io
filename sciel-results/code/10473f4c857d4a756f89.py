import time
from itertools import combinations

def construct():
    deadline = time.time() + 140
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    best_result = []
    best_count = 0
    
    # Try multiple orderings
    orderings = [
        all_triples,  # natural order
        sorted(all_triples, key=lambda t: (t[0], t[1], t[2])),
        sorted(all_triples, key=lambda t: (t[0] + t[1] + t[2], t[0], t[1], t[2])),
        sorted(all_triples, key=lambda t: (min(t[0], 8) + min(t[1], 8) + min(t[2], 8), t)),
    ]
    
    for ordering in orderings:
        present = set()
        result = []
        
        def has_triple(a, b, c):
            t = tuple(sorted((a, b, c)))
            return t in present
        
        for t in ordering:
            a, b, c = t
            others = [x for x in range(n) if x != a and x != b and x != c]
            valid = True
            for i in range(len(others)):
                ap = others[i]
                for j in range(len(others)):
                    if j == i:
                        continue
                    bp = others[j]
                    for k in range(len(others)):
                        if k == i or k == j:
                            continue
                        cp = others[k]
                        if has_triple(ap, b, c) and has_triple(a, bp, c) and has_triple(a, b, cp) and \
                           has_triple(ap, bp, c) and has_triple(ap, b, cp) and has_triple(a, bp, cp) and \
                           has_triple(ap, bp, cp):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                present.add((a, b, c))
                result.append([a, b, c])
            
            if time.time() > deadline:
                break
        
        if len(result) > best_count:
            best_count = len(result)
            best_result = result[:]
        
        if time.time() > deadline:
            break
    
    return best_result
