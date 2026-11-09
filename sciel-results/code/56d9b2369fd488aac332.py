import time
import random
from math import gcd

def construct():
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for m1 in range(2, max_mod + 1):
        for m2 in range(2, max_mod + 1):
            gcd_table[m1][m2] = gcd(m1, m2)
    
    deadline = time.time() + 80
    
    def try_insert_modulus(assignment, m_new):
        """
        Try to insert m_new into assignment by possibly reassigning one existing modulus's residue.
        Returns new assignment if successful, None otherwise.
        """
        # First, try to find a residue for m_new that works with current assignment
        for r in range(m_new):
            ok = True
            for m, r_old in assignment.items():
                g = gcd_table[m_new][m]
                if (r - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                new_assignment = dict(assignment)
                new_assignment[m_new] = r
                return new_assignment
        
        # Can't insert directly. Try reassigning one existing modulus.
        keys = list(assignment.keys())
        random.shuffle(keys)
        for m_old in keys:
            old_r = assignment[m_old]
            for r_new in range(m_old):
                if r_new == old_r:
                    continue
                # Check if r_new works for m_old with all other moduli
                ok_old = True
                for m_other, r_other in assignment.items():
                    if m_other == m_old:
                        continue
                    g = gcd_table[m_old][m_other]
                    if (r_new - r_other) % g == 0:
                        ok_old = False
                        break
                if not ok_old:
                    continue
                
                # Now try to insert m_new with m_old having r_new
                for r in range(m_new):
                    ok_new = True
                    for m_other, r_other in assignment.items():
                        if m_other == m_old:
                            if (r - r_new) % gcd_table[m_new][m_old] == 0:
                                ok_new = False
                                break
                        else:
                            if (r - r_other) % gcd_table[m_new][m_other] == 0:
                                ok_new = False
                                break
                    if ok_new:
                        new_assignment = dict(assignment)
                        new_assignment[m_old] = r_new
                        new_assignment[m_new] = r
                        return new_assignment
        
        return None
    
    def greedy_random_order():
        order = all_moduli[:]
        random.shuffle(order)
        current = {}
        for m in order:
            valid = []
            for r in range(m):
                ok = True
                for m2, r2 in current.items():
                    g = gcd_table[m][m2]
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    valid.append(r)
            if valid:
                current[m] = random.choice(valid)
        return current
    
    def greedy_least_constrained():
        remaining = set(all_moduli)
        current = {}
        while remaining:
            best_m = None
            best_valid = None
            best_count = -1
            for m in remaining:
                valid = []
                for r in range(m):
                    ok = True
                    for m2, r2 in current.items():
                        g = gcd_table[m][m2]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid.append(r)
                if len(valid) > best_count:
                    best_count = len(valid)
                    best_m = m
                    best_valid = valid
            if best_m is None or not best_valid:
                break
            current[best_m] = random.choice(best_valid)
            remaining.remove(best_m)
        return current
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    best_count = 0
    best_result = {}
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        if iteration % 4 == 0:
            initial = greedy_least_constrained()
        elif iteration % 4 == 1:
            initial = greedy_random_order()
        elif iteration % 4 == 2:
            primes = [m for m in all_moduli if is_prime(m)]
            composites = [m for m in all_moduli if not is_prime(m)]
            random.shuffle(primes)
            random.shuffle(composites)
            order = primes + composites
            current = {}
            for m in order:
                valid = []
                for r in range(m):
                    ok = True
                    for m2, r2 in current.items():
                        g = gcd_table[m][m2]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid.append(r)
                if valid:
                    current[m] = random.choice(valid)
            initial = current
        else:
            order = sorted(all_moduli, reverse=True)
            current = {}
            for m in order:
                valid = []
                for r in range(m):
                    ok = True
                    for m2, r2 in current.items():
                        g = gcd_table[m][m2]
                        if (r - r2) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid.append(r)
                if valid:
                    current[m] = random.choice(valid)
            initial = current
        
        # Local search: try to insert missing moduli
        current = dict(initial)
        missing = [m for m in all_moduli if m not in current]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for m in missing[:]:
                if m in current:
                    missing.remove(m)
                    continue
                new_assignment = try_insert_modulus(current, m)
                if new_assignment is not None:
                    current = new_assignment
                    missing.remove(m)
                    improved = True
        
        if len(current) > best_count:
            best_count = len(current)
            best_result = dict(current)
    
    if best_result:
        result = [[r, m] for m, r in best_result.items()]
        result.sort(key=lambda x: x[1])
        return result
    else:
        return [[0, 2]]
