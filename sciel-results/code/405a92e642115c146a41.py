import time
from math import gcd

def _find_max_classes(max_mod=128, deadline=None):
    if deadline is None:
        deadline = time.time() + 85  # Safety margin for 90s limit
    
    all_moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcd_table = {}
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = gcd(m1, m2)
            else:
                gcd_table[(m1, m2)] = m1

    best_set = []
    best_size = 0
    search_state = {'time_exceeded': False}
    
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def greedy_construct(order):
        current_set = []
        for m in order:
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in current_set:
                    g = gcd_table[(m, m_old)]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            if found_r != -1:
                current_set.append([found_r, m])
        return current_set

    # Initial greedy to get a baseline
    orders = [
        sorted(all_moduli, key=lambda m: count_prime_factors(m)),
        sorted(all_moduli, key=lambda m: -count_prime_factors(m)),
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        res = greedy_construct(order)
        if len(res) > best_size:
            best_size = len(res)
            best_set = [list(c) for c in res]
            
    def backtrack(remaining, assignments):
        if time.time() > deadline:
            search_state['time_exceeded'] = True
            return
        
        if not remaining:
            if len(assignments) > best_size:
                nonlocal best_size, best_set
                best_size = len(assignments)
                best_set = [[r, m] for m, r in assignments.items()]
            return
        
        # Pruning
        if len(assignments) + len(remaining) <= best_size:
            return
            
        # Select Most Constrained Variable (MRV)
        # Count conflicts with currently assigned moduli
        max_conflicts = -1
        selected_m = None
        # Tie-breaker: Smallest modulus to reduce residue search space?
        # Or largest? Let's try smallest.
        for m in remaining:
            c = 0
            for m_old in assignments:
                if gcd_table[(m, m_old)] > 1:
                    c += 1
            if c > max_conflicts:
                max_conflicts = c
                selected_m = m
            elif c == max_conflicts and selected_m is not None:
                if m < selected_m:
                    selected_m = m

        if selected_m is None:
            return

        new_remaining = [m for m in remaining if m != selected_m]
        
        # Try residues
        # Randomize order to avoid bias?
        # For determinism in finding the max, it doesn't matter.
        # But for speed, maybe try residues that are "popular" or "unpopular"?
        # Just 0..m-1.
        
        for r in range(selected_m):
            ok = True
            for m_old, r_old in assignments.items():
                g = gcd_table[(selected_m, m_old)]
                if (r - r_old) % g == 0:
                    ok = False
                    break
            
            if ok:
                assignments[selected_m] = r
                backtrack(new_remaining, assignments)
                del assignments[selected_m]
                
                if search_state['time_exceeded']:
                    return

    backtrack(all_moduli, {})
    
    if not best_set:
        best_set = greedy_construct(sorted(all_moduli))
        
    return best_set

def construct():
    return _find_max_classes()
