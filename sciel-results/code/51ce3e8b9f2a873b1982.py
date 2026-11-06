"""Hypergraph Ramsey R(4,5;3) construction."""
import time
import random
from itertools import combinations

def _build_index(n):
    triples = list(combinations(range(n), 3))
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx

def _is_valid(colors, n):
    triples, idx = _build_index(n)
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[idx[(a,b,c)]] == 0 and 
            colors[idx[(a,b,d)]] == 0 and 
            colors[idx[(a,c,d)]] == 0 and 
            colors[idx[(b,c,d)]] == 0):
            return False
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    return True

def _count_violations(colors, n, triples, idx):
    r = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[idx[(a,b,c)]] == 0 and 
            colors[idx[(a,b,d)]] == 0 and 
            colors[idx[(a,c,d)]] == 0 and 
            colors[idx[(b,c,d)]] == 0):
            r += 1
            if r > 100:
                return r, 0
    b = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            b += 1
            if b > 100:
                return r, b
    return r, b

def _local_search(n, deadline, seed=42):
    """Targeted local search focusing on violation-reducing flips."""
    triples, idx = _build_index(n)
    m = len(triples)
    rng = random.Random(seed)
    
    # Initial coloring: try sum mod 3
    colors = bytearray(len(triples))
    for i, (a, b, c) in enumerate(triples):
        colors[i] = 0 if (a + b + c) % 3 == 0 else 1
    
    r, b = _count_violations(colors, n, triples, idx)
    best_colors = colors.copy()
    best_viol = r + b
    
    if best_viol == 0:
        return list(colors)
    
    # Tabu-like: track recent flips
    tabu = {}
    tabu_len = 100
    
    for it in range(500000):
        if deadline and time.time() > deadline:
            break
        
        total = r + b
        if total == 0:
            return list(colors)
        
        if total < best_viol:
            best_viol = total
            best_colors = colors.copy()
        
        # Find triples involved in violations
        # For red K4: find all triples in red K4s
        # For blue K5: find all triples in blue K5s
        candidate_flips = set()
        
        # Sample some violations to find good flips
        for _ in range(5):
            # Try flipping a random triple in a red K4
            found = False
            for _ in range(50):
                quad = tuple(sorted(rng.sample(range(n), 4)))
                a, b, c, d = quad
                if (colors[idx[(a,b,c)]] == 0 and 
                    colors[idx[(a,b,d)]] == 0 and 
                    colors[idx[(a,c,d)]] == 0 and 
                    colors[idx[(b,c,d)]] == 0):
                    # Flip one of the four triples
                    tri = rng.choice([(a,b,c), (a,b,d), (a,c,d), (b,c,d)])
                    candidate_flips.add(idx[tri])
                    found = True
                    break
            if not found:
                # Try flipping a random triple in a blue K5
                for _ in range(50):
                    quint = tuple(sorted(rng.sample(range(n), 5)))
                    all_blue = True
                    for tri in combinations(quint, 3):
                        if colors[idx[tri]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        tri = rng.choice(list(combinations(quint, 3)))
                        candidate_flips.add(idx[tri])
                        break
        
        if not candidate_flips:
            candidate_flips.add(rng.randint(0, m - 1))
        
        i = rng.choice(list(candidate_flips))
        
        # Tabu check
        if i in tabu and tabu[i] > it - tabu_len:
            # Allow if it improves
            pass
        
        colors[i] = 1 - colors[i]
        tabu[i] = it
        r, b = _count_violations(colors, n, triples, idx)
    
    if best_viol == 0:
        return list(best_colors)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break
        
        for seed in range(20):
            if time.time() > deadline:
                break
            result = _local_search(n, deadline, seed=seed)
            if result is not None:
                return {'n': n, 'colors': result}
    
    # Fallback
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
