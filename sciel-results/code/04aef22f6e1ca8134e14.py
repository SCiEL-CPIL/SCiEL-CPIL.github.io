import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def verify(result):
        """Verify all pairs are disjoint and moduli are distinct."""
        if not result:
            return False
        moduli_used = set()
        for r, m in result:
            if m < 2 or m > MAX_MOD:
                return False
            if r < 0 or r >= m:
                return False
            if m in moduli_used:
                return False
            moduli_used.add(m)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                r1, m1 = result[i]
                r2, m2 = result[j]
                if not is_disjoint(r1, m1, r2, m2):
                    return False
        return True
    
    def try_greedy(moduli_order):
        selected = []
        for m in moduli_order:
            if time.time() > deadline:
                break
            for r in range(m):
                ok = True
                for (r2, m2) in selected:
                    if not is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    break
        return [[r, m] for r, m in selected]
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    best_result = []
    
    # Try multiple orderings
    orderings = [
        sorted(all_moduli, key=num_divisors, reverse=True),
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
    ]
    
    random.seed(42)
    for _ in range(10):
        rand_order = all_moduli.copy()
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = try_greedy(order)
        if verify(result) and len(result) > len(best_result):
            best_result = result
    
    # Try more random orderings if time permits
    attempts = 10
    while time.time() < deadline - 2 and attempts < 200:
        rand_order = all_moduli.copy()
        random.shuffle(rand_order)
        result = try_greedy(rand_order)
        if verify(result) and len(result) > len(best_result):
            best_result = result
        attempts += 1
    
    if not best_result:
        # Fallback
        best_result = [[1, 2], [2, 4]]
    
    return best_result
