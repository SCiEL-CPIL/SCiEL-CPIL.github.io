import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    moduli = list(range(2, 129))
    best_solution = []
    
    def count_divisors(n):
        count = 0
        i = 2
        while i * i <= n:
            while n % i == 0:
                count += 1
                n //= i
            i += 1
        if n > 1:
            count += 1
        return count
            
    div_counts = {m: count_divisors(m) for m in moduli}
    
    def greedy_trial(order):
        current_solution = []
        for m in order:
            # Create a boolean array to mark forbidden residues
            forbidden = [False] * m
            
            for (r_old, m_old) in current_solution:
                g = _gcd(m, m_old)
                # r must not be congruent to r_old mod g
                # So r = r_old + k*g is forbidden
                # We need to mark all such r in [0, m)
                # The step is g.
                # The residues are r_old % m, (r_old + g) % m, ...
                # Actually, the set of forbidden residues is exactly the residue class of r_old modulo g, mapped to [0, m).
                # Since g divides m, the residues are r_old % g, r_old % g + g, ...
                # Wait, r_old is in [0, m_old).
                # The condition is (r - r_old) % g != 0.
                # So r % g != r_old % g.
                # Let rem = r_old % g.
                # Then r is forbidden if r % g == rem.
                
                rem = r_old % g
                for k in range(rem, m, g):
                    forbidden[k] = True
            
            # Find a non-forbidden residue
            chosen_r = -1
            for r in range(m):
                if not forbidden[r]:
                    chosen_r = r
                    break
            
            if chosen_r != -1:
                current_solution.append([chosen_r, m])
                
        return current_solution

    # Try deterministic orders
    orders = [
        sorted(moduli),
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda x: -div_counts[x]),
        sorted(moduli, key=lambda x: div_counts[x])
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy_trial(order)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # Random restarts
    while time.time() < deadline:
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = greedy_trial(random_order)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    if not best_solution:
        return [[1, 2]]
        
    return best_solution
