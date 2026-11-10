import itertools
import time

def construct():
    # We aim to maximize n.
    # Strategy: 
    # 1. Try algebraic constructions based on Z_p for n from 34 down to 10.
    # 2. If found, return.
    # 3. If not found, fallback to a small valid n (e.g., n=9) to ensure non-zero score.
    
    def check_validity(n, colors):
        """
        Check if the coloring is valid.
        colors: list of 0/1 for all triples in lex order.
        """
        triple_to_idx = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    idx += 1
        
        # Check Red K4 (all 4 triples color 0)
        for quad in itertools.combinations(range(n), 4):
            is_red = True
            for triple in itertools.combinations(quad, 3):
                if colors[triple_to_idx[triple]] != 0:
                    is_red = False
                    break
            if is_red:
                return False
                
        # Check Blue K5 (all 10 triples color 1)
        for quint in itertools.combinations(range(n), 5):
            is_blue = True
            for triple in itertools.combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    is_blue = False
                    break
            if is_blue:
                return False
                
        return True

    def generate_algebraic_coloring(n, p, func_id):
        """
        Generate a coloring based on differences in Z_p.
        func_id selects the function f(x,y).
        """
        colors = [0] * (n * (n-1) * (n-2) // 6)
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    x = (b - a) % p
                    y = (c - b) % p
                    # Apply function
                    if func_id == 0:
                        # f(x,y) = (x*y) % 2
                        col = (x * y) % 2
                    elif func_id == 1:
                        # f(x,y) = (x+y) % 2
                        col = (x + y) % 2
                    elif func_id == 2:
                        # f(x,y) = (x^2 + y^2) % 2
                        col = ((x*x + y*y) % 2)
                    elif func_id == 3:
                        # f(x,y) = 1 if x < y else 0 (asymmetric, but let's try)
                        col = 1 if x < y else 0
                    elif func_id == 4:
                        # f(x,y) = (x + 2*y) % 2
                        col = (x + 2*y) % 2
                    elif func_id == 5:
                        # f(x,y) = (x*y + x + y) % 2
                        col = (x*y + x + y) % 2
                    else:
                        col = 0
                    colors[idx] = col
                    idx += 1
        return colors

    # Primes to try for p
    primes = [37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    # Function IDs to try
    func_ids = [0, 1, 2, 3, 4, 5]
    
    # Try to find a valid n starting from 34 downwards
    for n in range(34, 9, -1):
        for p in primes:
            if p < n:
                continue
            for f_id in func_ids:
                colors = generate_algebraic_coloring(n, p, f_id)
                if check_validity(n, colors):
                    return {'n': n, 'colors': colors}
                    
    # Fallback: If algebraic search fails for n >= 10, 
    # we need a valid coloring for a smaller n to get a score.
    # n=9 is a known small valid size for many Ramsey-type problems or can be easily constructed.
    # Let's construct a valid n=9 coloring.
    # A simple valid coloring for n=9 can be found by random search or known small cases.
    # For R(4,5;3), n=9 is likely valid.
    # We will use a deterministic random seed to find a valid n=9 coloring quickly.
    
    import random
    random.seed(42)
    
    def find_small_valid(n, time_limit=10):
        start = time.time()
        num_triples = n * (n-1) * (n-2) // 6
        triple_to_idx = {}
        idx = 0
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    triples.append((a,b,c))
                    idx += 1
        
        # Precompute K4 and K5 edges
        k4_edges = []
        for quad in itertools.combinations(range(n), 4):
            k4_edges.append([triple_to_idx[t] for t in itertools.combinations(quad, 3)])
        k5_edges = []
        for quint in itertools.combinations(range(n), 5):
            k5_edges.append([triple_to_idx[t] for t in itertools.combinations(quint, 3)])
        
        while time.time() - start < time_limit:
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            # Quick check
            valid = True
            for edges in k4_edges:
                if all(colors[e] == 0 for e in edges):
                    valid = False
                    break
            if not valid: continue
            for edges in k5_edges:
                if all(colors[e] == 1 for e in edges):
                    valid = False
                    break
            if valid:
                return colors
        return None

    # Try n=9
    colors_9 = find_small_valid(9)
    if colors_9:
        return {'n': 9, 'colors': colors_9}
        
    # Ultimate fallback: n=1 is always valid
    return {'n': 1, 'colors': [0]}
