import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)

    def sum_recip(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    def evaluate_L(assignment, max_check=10000):
        """Find largest L such that 1..L all covered."""
        if not assignment:
            return 0
        # Use a boolean array for speed
        covered = bytearray(max_check + 1)
        for p, r in assignment.items():
            first = r if r > 0 else p
            for x in range(first, max_check + 1, p):
                covered[x] = 1
        L = 0
        for x in range(1, max_check + 1):
            if covered[x]:
                L = x
            else:
                break
        return L

    # Strategy: Start with a good set of primes, then optimize residues
    # Key insight: small primes (2,3,5,7,11) are very valuable for coverage
    # but cost more budget. We need to find the right mix.
    
    # Precompute useful prime groups
    small_primes = [2, 3, 5, 7, 11]
    medium_primes = [p for p in primes if 13 <= p <= 100]
    large_primes = [p for p in primes if p > 100]
    
    all_primes = primes[:]
    
    # Initial assignment: use all primes >= 13 with random residues
    # Then try to add small primes if budget allows
    def initial_assignment():
        # Start with medium and large primes
        used_primes = medium_primes + large_primes
        assign = {}
        for p in used_primes:
            assign[p] = random.randint(0, p - 1)
        
        # Try to add small primes if budget allows
        s = sum_recip(list(assign.keys()))
        for p in sorted(small_primes, reverse=True):  # Add larger small primes first
            if s + Fraction(1, p) <= 1:
                assign[p] = random.randint(0, p - 1)
                s += Fraction(1, p)
        
        return assign

    best_assign = initial_assignment()
    best_L = evaluate_L(best_assign)
    
    # Local search: optimize residues to maximize coverage
    def local_search(assignment, time_budget):
        t0 = time.time()
        current = assignment.copy()
        current_L = evaluate_L(current)
        best = current.copy()
        best_L = current_L
        temperature = 5.0
        cooling = 0.999
        
        while time.time() - t0 < time_budget:
            # Pick a random prime to change residue
            if not current:
                break
            p = random.choice(list(current.keys()))
            old_r = current[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            current[p] = new_r
            new_L = evaluate_L(current)
            delta = new_L - current_L
            if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.01)):
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best = current.copy()
            else:
                current[p] = old_r
            
            # Occasionally try swapping primes
            if random.random() < 0.1:
                available = [p for p in all_primes if p not in current]
                if available and len(current) > 5:
                    p_remove = random.choice(list(current.keys()))
                    s_after = sum_recip([p for p in current if p != p_remove])
                    candidates = [p for p in available if s_after + Fraction(1, p) <= 1]
                    if candidates:
                        p_add = random.choice(candidates)
                        r_remove = current[p_remove]
                        del current[p_remove]
                        current[p_add] = random.randint(0, p_add - 1)
                        new_L = evaluate_L(current)
                        delta = new_L - current_L
                        if delta > 0 or random.random() < pow(2.718, delta / max(temperature, 0.01)):
                            current_L = new_L
                            if new_L > best_L:
                                best_L = new_L
                                best = current.copy()
                        else:
                            del current[p_add]
                            current[p_remove] = r_remove
            
            temperature *= cooling
        
        return best, best_L

    # Run local search
    remaining = deadline - time.time() - 5
    if remaining > 10:
        best_assign, best_L = local_search(best_assign, remaining * 0.7)
    
    # Final polish: try targeted residue changes for gaps
    if time.time() < deadline - 3:
        L_check = best_L + 1
        # Find first gap
        covered = bytearray(L_check + 1)
        for p, r in best_assign.items():
            first = r if r > 0 else p
            for x in range(first, L_check + 1, p):
                covered[x] = 1
        
        if not covered[L_check]:
            # Try to cover L_check by adjusting residues
            for p in list(best_assign.keys()):
                needed_r = L_check % p
                if best_assign[p] != needed_r:
                    old_r = best_assign[p]
                    best_assign[p] = needed_r
                    new_L = evaluate_L(best_assign)
                    if new_L > best_L:
                        best_L = new_L
                    else:
                        best_assign[p] = old_r
                        break
    
    # Format output
    result = []
    for p in sorted(best_assign.keys()):
        result.append([p, best_assign[p]])
    
    return result
