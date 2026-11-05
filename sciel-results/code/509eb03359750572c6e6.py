"""Rewritten program for hypergraph Ramsey R(4,5;3) construction."""
import time
import random
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
DEADLINE = 110  # seconds, safety margin before 120s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _check_validity(n, colors):
    """Check that no 4 vertices have all 4 triples red, no 5 vertices have all 10 triples blue."""
    # Build index for triples
    # colors[i] corresponds to the i-th triple in lexicographic order
    # We need a fast lookup: for vertices a<b<c, find index in combinations list
    # For n up to ~35, C(n,3) can be up to ~6545, so building a dict is feasible
    
    # Precompute triple index
    triple_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_idx[(a,b,c)] = idx
                idx += 1
    
    # Check red K4: all 4 triples red among any 4 vertices
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[triple_idx[(a,b,c)]] == 0 and
            colors[triple_idx[(a,b,d)]] == 0 and
            colors[triple_idx[(a,c,d)]] == 0 and
            colors[triple_idx[(b,c,d)]] == 0):
            return False
    
    # Check blue K5: all 10 triples blue among any 5 vertices
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[triple_idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True

def _count_violations(n, colors):
    """Count number of red K4 and blue K5 violations (0 means valid)."""
    triple_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_idx[(a,b,c)] = idx
                idx += 1
    
    red_k4 = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[triple_idx[(a,b,c)]] == 0 and
            colors[triple_idx[(a,b,d)]] == 0 and
            colors[triple_idx[(a,c,d)]] == 0 and
            colors[triple_idx[(b,c,d)]] == 0):
            red_k4 += 1
    
    blue_k5 = 0
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[triple_idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            blue_k5 += 1
    
    return red_k4 + blue_k5

def _local_search(n, colors, max_iters=5000):
    """Hill-climbing local search to fix violations."""
    colors = list(colors)
    
    triple_idx = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_idx[(a,b,c)] = idx
                idx += 1
    
    # Precompute which triples involve each vertex
    vertex_triples = [[] for _ in range(n)]
    for (a,b,c), idx in triple_idx.items():
        vertex_triples[a].append(idx)
        vertex_triples[b].append(idx)
        vertex_triples[c].append(idx)
    
    def violations():
        red_k4 = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_idx[(a,b,c)]] == 0 and
                colors[triple_idx[(a,b,d)]] == 0 and
                colors[triple_idx[(a,c,d)]] == 0 and
                colors[triple_idx[(b,c,d)]] == 0):
                red_k4 += 1
        blue_k5 = 0
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[triple_idx[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5 += 1
        return red_k4 + blue_k5
    
    best = violations()
    if best == 0:
        return colors
    
    iters = 0
    stuck = 0
    while iters < max_iters and _time_left() > 10:
        # Pick a random triple to flip
        t = random.randint(0, len(colors)-1)
        old_color = colors[t]
        colors[t] = 1 - old_color
        new_v = violations()
        if new_v < best:
            best = new_v
            stuck = 0
            if best == 0:
                return colors
        elif new_v == best and random.random() < 0.1:
            stuck = 0
        else:
            colors[t] = old_color
            stuck += 1
        iters += 1
        if stuck > 200:
            # Random restart within local search
            stuck = 0
            for _ in range(min(20, len(colors))):
                t = random.randint(0, len(colors)-1)
                colors[t] = 1 - colors[t]
    
    return colors

def _try_cyclic_construction(n):
    """Try cyclic constructions based on quadratic residues and other algebraic rules."""
    from itertools import combinations
    
    # Try various algebraic colorings
    best_n = 0
    best_colors = None
    
    # Rule 1: quadratic residue of (b-a)*(c-b)*(c-a) mod n
    # Rule 2: sum mod 2
    # Rule 3: product of differences mod n
    # Rule 4: various hash-like functions
    
    for n_try in range(30, min(n+1, 36)):
        if _time_left() < 20:
            break
        
        found = False
        # Try quadratic residue coloring
        # For each triple (a,b,c), compute (b-a) mod n, (c-b) mod n, (c-a) mod n
        # Color based on whether these are quadratic residues
        
        # Simple: color = 0 if ((b-a)*(c-b)*(c-a)) is a QR mod n, else 1
        # But we need to be careful with n not being prime
        
        # Try: color based on (a+b+c) mod 2, (a+b+c) mod 3, etc.
        for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
            if _time_left() < 15:
                break
            colors = []
            for a, b, c in combinations(range(n_try), 3):
                s = (a + b + c) % mod
                # Use a threshold: if s < mod//2, red else blue
                colors.append(0 if s < mod//2 else 1)
            
            if _check_validity(n_try, colors):
                return {'n': n_try, 'colors': colors}
        
        # Try: color = 0 if (b-a) is a quadratic residue mod n_try (for prime n_try)
        if n_try > 2:
            # Check if n_try is prime
            is_prime = True
            for p in range(2, int(n_try**0.5)+1):
                if n_try % p == 0:
                    is_prime = False
                    break
            if is_prime:
                # Quadratic residues mod p
                qr = set((x*x) % n_try for x in range(n_try))
                colors = []
                for a, b, c in combinations(range(n_try), 3):
                    d1 = (b - a) % n_try
                    d2 = (c - b) % n_try
                    d3 = (c - a) % n_try
                    # Color red if all three differences are QR
                    if d1 in qr and d2 in qr and d3 in qr:
                        colors.append(0)
                    else:
                        colors.append(1)
                
                if _check_validity(n_try, colors):
                    return {'n': n_try, 'colors': colors}
                
                # Try: red if exactly 2 are QR
                colors = []
                for a, b, c in combinations(range(n_try), 3):
                    d1 = (b - a) % n_try
                    d2 = (c - b) % n_try
                    d3 = (c - a) % n_try
                    cnt = (1 if d1 in qr else 0) + (1 if d2 in qr else 0) + (1 if d3 in qr else 0)
                    colors.append(0 if cnt == 2 else 1)
                
                if _check_validity(n_try, colors):
                    return {'n': n_try, 'colors': colors}
    
    return None

def _backtracking_search(n, max_nodes=100000):
    """Backtracking with constraint propagation to find valid coloring."""
    from itertools import combinations
    
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Build adjacency: for each vertex, which triples contain it
    vertex_triples = [[] for _ in range(n)]
    for i, (a,b,c) in enumerate(triples):
        vertex_triples[a].append(i)
        vertex_triples[b].append(i)
        vertex_triples[c].append(i)
    
    # For each 4-set, the 4 triples that form it
    red_k4_sets = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        red_k4_sets.append([(a,b,c), (a,b,d), (a,c,d), (b,c,d)])
    
    # For each 5-set, the 10 triples
    blue_k5_sets = []
    for quint in combinations(range(n), 5):
        blue_k5_sets.append(list(combinations(quint, 3)))
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Convert red_k4_sets to index lists
    red_k4_idx = [[triple_to_idx[t] for t in s] for s in red_k4_sets]
    blue_k5_idx = [[triple_to_idx[t] for t in s] for s in blue_k5_sets]
    
    colors = [-1] * num_triples
    nodes = [0]
    
    def backtrack(pos):
        if _time_left() < 5:
            return False
        if pos == num_triples:
            return True
        
        nodes[0] += 1
        if nodes[0] % 10000 == 0 and _time_left() < 5:
            return False
        
        # Try color 0 (red)
        colors[pos] = 0
        valid = True
        # Check if this creates a red K4
        for k4 in red_k4_idx:
            if pos in k4:
                all_red = True
                for t in k4:
                    if colors[t] != 0:
                        all_red = False
                        break
                if all_red:
                    valid = False
                    break
        if valid:
            if backtrack(pos + 1):
                return True
        
        # Try color 1 (blue)
        colors[pos] = 1
        valid = True
        # Check if this creates a blue K5
        for k5 in blue_k5_idx:
            if pos in k5:
                all_blue = True
                for t in k5:
                    if colors[t] != 1:
                        all_blue = False
                        break
                if all_blue:
                    valid = False
                    break
        if valid:
            if backtrack(pos + 1):
                return True
        
        colors[pos] = -1
        return False
    
    if backtrack(0):
        return colors
    
    return None

def construct():
    """Construct a valid coloring maximizing n."""
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with a known small valid coloring and try to grow
    
    # Strategy: 
    # 1. Try cyclic/algebraic constructions for n=30..35
    # 2. If found, return it
    # 3. Otherwise, use local search from random initializations
    # 4. Use backtracking for smaller n if time permits
    
    # First try algebraic constructions
    result = _try_cyclic_construction(35)
    if result:
        return result
    
    # Try local search from various starting points
    best_n = 0
    best_colors = None
    
    # Start from n=20 with a random coloring and try to grow
    for target_n in [34, 33, 32, 31, 30, 28, 26, 24, 22, 20]:
        if _time_left() < 15:
            break
        
        found = False
        for attempt in range(5):
            if _time_left() < 10:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 1) for _ in range(len(list(combinations(range(target_n), 3))))]
            
            # Local search
            colors = _local_search(target_n, colors, max_iters=3000)
            
            if _check_validity(target_n, colors):
                best_n = target_n
                best_colors = colors
                found = True
                break
        
        if found:
            break
    
    if best_n > 0:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: try backtracking for small n
    for n in [12, 10, 8]:
        if _time_left() < 5:
            break
        colors = _backtracking_search(n)
        if colors is not None:
            return {'n': n, 'colors': colors}
    
    # Last resort: return a trivial valid coloring
    n = 6
    colors = [0] * len(list(combinations(range(n), 3)))
    # Check validity
    if not _check_validity(n, colors):
        # Try random
        for _ in range(100):
            colors = [random.randint(0, 1) for _ in range(len(list(combinations(range(n), 3))))]
            if _check_validity(n, colors):
                break
    
    return {'n': n, 'colors': colors}
