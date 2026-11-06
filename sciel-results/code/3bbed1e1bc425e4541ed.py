import math
import random
import time

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Moduli to consider: all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    n = len(moduli)
    
    # Precompute gcd matrix for speed
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    def try_greedy(order):
        """
        order: list of indices into moduli
        returns: dict {modulus: residue} or None
        """
        residues = {}
        assigned_indices = []
        
        for idx in order:
            m = moduli[idx]
            # Find a valid residue r for m
            found = False
            for r in range(m):
                valid = True
                # Check against all already assigned
                for j_idx in assigned_indices:
                    m_j = moduli[j_idx]
                    r_j = residues[m_j]
                    g = gcds[idx][j_idx]
                    if (r - r_j) % g == 0:
                        valid = False
                        break
                if valid:
                    residues[m] = r
                    assigned_indices.append(idx)
                    found = True
                    break
            if not found:
                return None
        
        return residues

    # Strategy 1: Deterministic orderings
    # Order by increasing modulus
    order1 = list(range(n))
    res = try_greedy(order1)
    if res is not None:
        return [[res[m], m] for m in res]
        
    # Order by decreasing modulus
    order2 = list(range(n-1, -1, -1))
    res = try_greedy(order2)
    if res is not None:
        return [[res[m], m] for m in res]
        
    # Order by number of divisors (more constrained first?)
    def num_divs(x):
        c = 0
        for d in range(1, int(x**0.5)+1):
            if x % d == 0:
                c += 1
                if d != x//d:
                    c += 1
        return c
    order3 = list(range(n))
    order3.sort(key=lambda i: (-num_divs(moduli[i]), moduli[i]))
    res = try_greedy(order3)
    if res is not None:
        return [[res[m], m] for m in res]

    # Strategy 2: Random restarts
    # If deterministic fails, try random orders
    for _ in range(2000):
        if time.time() > deadline:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if res is not None:
            return [[res[m], m] for m in res]
            
    # Fallback: Return a small valid set if all else fails
    # e.g. [0, 2], [1, 4]
    return [[0, 2], [1, 4]]
