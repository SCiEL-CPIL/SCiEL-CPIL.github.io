import random
import time
from itertools import combinations

def construct():
    def check_full(colors, n, idx_map):
        # Full check for validity
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[idx_map[(a,b,c)]] == 0 and 
                colors[idx_map[(a,b,d)]] == 0 and 
                colors[idx_map[(a,c,d)]] == 0 and 
                colors[idx_map[(b,c,d)]] == 0):
                return quad
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            if (colors[idx_map[(a,b,c)]] == 1 and 
                colors[idx_map[(a,b,d)]] == 1 and 
                colors[idx_map[(a,b,e)]] == 1 and 
                colors[idx_map[(a,c,d)]] == 1 and 
                colors[idx_map[(a,c,e)]] == 1 and 
                colors[idx_map[(a,d,e)]] == 1 and 
                colors[idx_map[(b,c,d)]] == 1 and 
                colors[idx_map[(b,c,e)]] == 1 and 
                colors[idx_map[(b,d,e)]] == 1 and 
                colors[idx_map[(c,d,e)]] == 1):
                return quint
        return None

    def local_search(n, time_limit=5.0):
        random.seed(42 + n)
        triples = list(combinations(range(n), 3))
        idx_map = {}
        for i, (a, b, c) in enumerate(triples):
            idx_map[(a, b, c)] = i
            
        colors = [random.randint(0, 1) for _ in range(len(triples))]
        
        start_time = time.time()
        steps = 0
        while time.time() - start_time < time_limit:
            steps += 1
            # Try to find a violation by sampling
            violation = None
            
            # Sample 4-sets
            for _ in range(50):
                quad = tuple(sorted(random.sample(range(n), 4)))
                a, b, c, d = quad
                if (colors[idx_map[(a,b,c)]] == 0 and 
                    colors[idx_map[(a,b,d)]] == 0 and 
                    colors[idx_map[(a,c,d)]] == 0 and 
                    colors[idx_map[(b,c,d)]] == 0):
                    violation = quad
                    break
            
            # Sample 5-sets if no 4-set violation
            if violation is None:
                for _ in range(50):
                    quint = tuple(sorted(random.sample(range(n), 5)))
                    a, b, c, d, e = quint
                    if (colors[idx_map[(a,b,c)]] == 1 and 
                        colors[idx_map[(a,b,d)]] == 1 and 
                        colors[idx_map[(a,b,e)]] == 1 and 
                        colors[idx_map[(a,c,d)]] == 1 and 
                        colors[idx_map[(a,c,e)]] == 1 and 
                        colors[idx_map[(a,d,e)]] == 1 and 
                        colors[idx_map[(b,c,d)]] == 1 and 
                        colors[idx_map[(b,c,e)]] == 1 and 
                        colors[idx_map[(b,d,e)]] == 1 and 
                        colors[idx_map[(c,d,e)]] == 1):
                        violation = quint
                        break
            
            if violation is not None:
                # Fix violation
                if len(violation) == 4:
                    a, b, c, d = violation
                    triples_to_flip = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                else:
                    a, b, c, d, e = violation
                    triples_to_flip = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
                    
                t = random.choice(triples_to_flip)
                idx = idx_map[t]
                colors[idx] = 1 - colors[idx]
            else:
                # No violation found in samples, do a full check
                full_violation = check_full(colors, n, idx_map)
                if full_violation is None:
                    return colors
                else:
                    # Fix the full violation
                    violation = full_violation
                    if len(violation) == 4:
                        a, b, c, d = violation
                        triples_to_flip = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
                    else:
                        a, b, c, d, e = violation
                        triples_to_flip = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
                    t = random.choice(triples_to_flip)
                    idx = idx_map[t]
                    colors[idx] = 1 - colors[idx]
                    
        return None

    # Try to find valid colorings for decreasing n
    for n in [34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9]:
        res = local_search(n, time_limit=5.0)
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
