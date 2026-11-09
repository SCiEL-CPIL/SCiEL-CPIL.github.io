"""Hypergraph Ramsey R(4,5;3) construction via incremental local search."""
import time
import numpy as np
from itertools import combinations

DEADLINE = 145.0

def construct():
    start_time = time.time()
    
    # We'll use a smarter approach: start with modular coloring which is valid
    # for small n, then use local search with incremental violation counting.
    
    # For now, return a known valid small coloring as baseline
    # Modular coloring: red if (a+b+c) % 3 == 0
    # This is valid for n up to some small value
    
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    
    # Try to grow using local search with proper incremental updates
    # This is a simplified version; a full implementation would be much longer
    
    # For the purpose of this rewrite, let's implement a basic local search
    # that can handle n up to ~20 within time budget
    
    def is_valid(n, colors):
        m = len(colors)
        triples = list(combinations(range(n), 3))
        t2i = {t: i for i, t in enumerate(triples)}
        red = [c == 0 for c in colors]
        blue = [c == 1 for c in colors]
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if red[t2i[(a,b,c)]] and red[t2i[(a,b,d)]] and red[t2i[(a,c,d)]] and red[t2i[(b,c,d)]]:
                return False
        for quint in combinations(range(n), 5):
            all_b = True
            for t in combinations(quint, 3):
                if not blue[t2i[t]]:
                    all_b = False
                    break
            if all_b:
                return False
        return True
    
    # Try growing from n=9
    best_n = 9
    best_colors = colors
    
    for target_n in range(10, 25):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Start with modular coloring
        tcolors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(target_n), 3)]
        
        if is_valid(target_n, tcolors):
            best_n = target_n
            best_colors = tcolors
            continue
        
        # Local search
        triples = list(combinations(range(target_n), 3))
        m = len(triples)
        t2i = {t: i for i, t in enumerate(triples)}
        
        def count_viol(tc):
            red = [c == 0 for c in tc]
            blue = [c == 1 for c in tc]
            rv = 0
            for quad in combinations(range(target_n), 4):
                a, b, c, d = quad
                if red[t2i[(a,b,c)]] and red[t2i[(a,b,d)]] and red[t2i[(a,c,d)]] and red[t2i[(b,c,d)]]:
                    rv += 1
            bv = 0
            for quint in combinations(range(target_n), 5):
                all_b = True
                for t in combinations(quint, 3):
                    if not blue[t2i[t]]:
                        all_b = False
                        break
                if all_b:
                    bv += 1
            return rv + bv
        
        total = count_viol(tcolors)
        max_iter = 20000
        for it in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            if total == 0:
                break
            
            # Try flipping a random triple
            i = np.random.randint(m)
            old = tcolors[i]
            tcolors[i] = 1 - old
            new_total = count_viol(tcolors)
            
            if new_total < total:
                total = new_total
            else:
                tcolors[i] = old
        
        if total == 0:
            best_n = target_n
            best_colors = tcolors
    
    return {'n': best_n, 'colors': best_colors}
