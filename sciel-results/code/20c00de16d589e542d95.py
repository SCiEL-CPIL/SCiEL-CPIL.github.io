"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations
from functools import lru_cache

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs
    lcm_table = {}
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            lcm_table[(a, b)] = a * b // math.gcd(a, b)
    
    def get_lcm(a, b):
        if a > b:
            a, b = b, a
        return lcm_table[(a, b)]
    
    # Build hyperedges: sets of 3 distinct elements {a,b,c} such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # For three numbers a,b,c, this happens when all three pairwise LCMs are equal.
    
    # Let's find all such triples
    bad_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                l_ab = get_lcm(a, b)
                l_ac = get_lcm(a, c)
                if l_ab == l_ac:
                    l_bc = get_lcm(b, c)
                    if l_ab == l_bc:
                        bad_triples.add((a, b, c))
    
    # Build conflict graph: vertices are elements, edge between x,y if there exists z
    # such that {x,y,z} is a bad triple
    # Actually, for maximum independent set on hypergraph, we can use a different approach.
    # Let's use local search with kicks.
    
    # Greedy initialization: start with a random seed set
    def greedy_solve(order):
        selected = []
        for x in order:
            can_add = True
            for a, b in combinations(selected, 2):
                l_ab = get_lcm(a, b)
                l_ax = get_lcm(a, x)
                l_bx = get_lcm(b, x)
                if l_ab == l_ax == l_bx:
                    can_add = False
                    break
            if can_add:
                selected.append(x)
        return selected
    
    def is_valid(selected):
        for a, b, c in combinations(selected, 3):
            l_ab = get_lcm(a, b)
            l_ac = get_lcm(a, c)
            l_bc = get_lcm(b, c)
            if l_ab == l_ac == l_bc:
                return False
        return True
    
    def local_search(selected):
        selected = set(selected)
        improved = True
        while improved:
            improved = False
            # Try adding elements
            for x in elements:
                if x not in selected:
                    trial = selected | {x}
                    if is_valid(trial):
                        selected = trial
                        improved = True
            # Try removing and re-adding
            for x in list(selected):
                for y in elements:
                    if y not in selected:
                        trial = (selected - {x}) | {y}
                        if len(trial) > len(selected) and is_valid(trial):
                            selected = trial
                            improved = True
        return list(selected)
    
    def local_search_with_kicks(selected):
        selected = set(selected)
        best = set(selected)
        best_size = len(selected)
        
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            # Random kick: remove k random elements and re-add
            k = random.randint(2, 5)
            to_remove = random.sample(list(selected), min(k, len(selected)))
            kicked = selected - set(to_remove)
            
            # Try to add back from remaining elements
            for x in elements:
                if x not in kicked:
                    trial = kicked | {x}
                    if is_valid(trial):
                        kicked = trial
            
            # Local improvement
            improved = True
            while improved:
                improved = False
                for x in elements:
                    if x not in kicked:
                        trial = kicked | {x}
                        if is_valid(trial):
                            kicked = trial
                            improved = True
            
            if len(kicked) > best_size:
                best = kicked
                best_size = len(kicked)
        
        return list(best)
    
    # Try multiple strategies
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            order = list(elements)
        elif attempt == 1:
            order = list(range(65, 129)) + list(range(1, 65))
        elif attempt == 2:
            # Primes + 1
            primes = [x for x in range(2, 129) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
            order = [p + 1 for p in primes if p + 1 <= 128] + [x for x in elements if x not in [p + 1 for p in primes if p + 1 <= 128]]
        elif attempt == 3:
            order = list(range(128, 0, -1))
        else:
            order = elements[:]
            random.shuffle(order)
        
        result = greedy_solve(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
        
        # Local search
        if time.time() < deadline - 5:
            improved = local_search(result)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = set(improved)
    
    # Strategy 2: Local search with kicks from best found
    if time.time() < deadline - 10:
        kicked_result = local_search_with_kicks(list(best_set))
        if len(kicked_result) > best_size:
            best_size = len(kicked_result)
            best_set = set(kicked_result)
    
    # Additional kicks
    for _ in range(20):
        if time.time() > deadline:
            break
        kicked_result = local_search_with_kicks(list(best_set))
        if len(kicked_result) > best_size:
            best_size = len(kicked_result)
            best_set = set(kicked_result)
    
    return list(best_set) if best_set else [1]
# EVOLVE-BLOCK-END
