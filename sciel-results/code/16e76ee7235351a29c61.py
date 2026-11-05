import random
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute prime factors
    prime_factors = {}
    for m in moduli:
        factors = set()
        temp = m
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        prime_factors[m] = factors
        
    def omega(m):
        return len(prime_factors[m])
        
    # Generate orders
    order_asc = sorted(moduli, key=lambda x: omega(x))
    order_desc = sorted(moduli, key=lambda x: -omega(x))
    order_natural = moduli
    order_random = moduli.copy()
    random.shuffle(order_random)
    
    orders = [order_asc, order_desc, order_natural, order_random]
    
    best_count = 0
    best_solution = []
    
    def is_compatible(r_new, m_new, assignment):
        pf_new = prime_factors[m_new]
        for m_old, r_old in assignment.items():
            pf_old = prime_factors[m_old]
            common = pf_new.intersection(pf_old)
            if not common:
                return False
            diff = r_new - r_old
            compatible = False
            for p in common:
                if diff % p == 0:
                    compatible = True
                    break
            if not compatible:
                return False
        return True

    def greedy_search(order, time_limit):
        assignment = {}
        for m in order:
            if time.time() > time_limit:
                break
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(r, m, assignment):
                    assignment[m] = r
                    break
        return assignment

    # Run searches
    # We have ~85s.
    # Let's do 1000 iterations total, distributed among orders.
    total_iterations = 1000
    it_per_order = total_iterations // len(orders)
    
    for order in orders:
        for _ in range(it_per_order):
            if time.time() > deadline:
                break
            current_time_limit = min(deadline, time.time() + 0.05)
            assignment = greedy_search(order, current_time_limit)
            if len(assignment) > best_count:
                best_count = len(assignment)
                best_solution = [[r, m] for m, r in assignment.items()]
                
    if not best_solution:
        best_solution = [[0, 1<<k] for k in range(1, 8)]
        
    return best_solution
