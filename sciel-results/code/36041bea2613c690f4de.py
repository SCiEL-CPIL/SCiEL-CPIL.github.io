"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 80
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    pf = {m: prime_factors(m) for m in moduli}
    
    def greedy_with_constraints(order):
        selected = []
        prime_count = defaultdict(int)
        for m in order:
            can_add = True
            for p in pf[m]:
                if prime_count[p] >= p:
                    can_add = False
                    break
            if can_add:
                selected.append(m)
                for p in pf[m]:
                    prime_count[p] += 1
        return selected
    
    best_set = []
    
    orderings = [
        sorted(moduli, key=lambda m: (len(pf[m]), m)),
        sorted(moduli, key=lambda m: (-len(pf[m]), m)),
        sorted(moduli, key=lambda m: m),
        sorted(moduli, key=lambda m: -m),
        sorted(moduli, key=lambda m: (max(pf[m], default=1), m)),
    ]
    
    for order in orderings:
        s = greedy_with_constraints(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try to add any modulus not in best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best_set)
        prime_count = defaultdict(int)
        for m in current:
            for p in pf[m]:
                prime_count[p] += 1
        for m in moduli:
            if m in current:
                continue
            can_add = True
            for p in pf[m]:
                if prime_count[p] >= p:
                    can_add = False
                    break
            if can_add:
                current.append(m)
                for p in pf[m]:
                    prime_count[p] += 1
                improved = True
        best_set = current
    
    def assign_residues(selected):
        order = sorted(selected, key=lambda m: (len(pf[m]), m))
        assignment = {}
        
        def backtrack(idx):
            if idx == len(order):
                return True
            m = order[idx]
            for r in range(m):
                ok = True
                for m2, r2 in assignment.items():
                    g = math.gcd(m, m2)
                    if r % g != r2 % g:
                        continue
                    else:
                        ok = False
                        break
                if ok:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    assignment = assign_residues(best_set)
    
    if assignment is None:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return [[assignment[m], m] for m in best_set]
# EVOLVE-BLOCK-END
