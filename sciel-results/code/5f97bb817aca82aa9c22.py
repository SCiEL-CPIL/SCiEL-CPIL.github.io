import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    def build_coverage(classes, L_max):
        covered = np.zeros(L_max + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= L_max:
                covered[start::p] = True
        return covered
    
    def calc_L_from_bitset(covered, L_max):
        if not np.all(covered[1:L_max+1]):
            uncov = np.where(~covered[1:L_max+1])[0]
            if len(uncov) == 0:
                return L_max
            return int(uncov[0])
        return L_max
    
    def greedy_with_lookahead(prime_list, L_max=10000):
        uncovered_count = np.ones(L_max + 1, dtype=np.int32)
        uncovered_count[0] = 0
        classes = []
        budget_used = Fraction(0)
        used_primes = set()
        
        for p in sorted(prime_list):
            if p in used_primes:
                continue
            if budget_used + Fraction(1, p) > 1:
                continue
            
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = p if r == 0 else r
                if start > L_max:
                    continue
                mask = uncovered_count[start::p] > 0
                gain = np.sum(mask)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                classes.append([p, best_r])
                budget_used += Fraction(1, p)
                used_primes.add(p)
                start = p if best_r == 0 else best_r
                uncovered_count[start::p] -= 1
                uncovered_count[start::p] = np.maximum(uncovered_count[start::p], 0)
            
            if np.all(uncovered_count[1:L_max+1] == 0):
                break
        
        return classes
    
    def local_search_bitset(classes, L_max=10000, max_iter=100):
        current = [list(c) for c in classes]
        covered = build_coverage(current, L_max)
        current_L = calc_L_from_bitset(covered, L_max)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, (p, r) in enumerate(current):
                best_r = r
                best_L = current_L
                for r_new in range(p):
                    if r_new == r:
                        continue
                    old_start = p if r == 0 else r
                    new_start = p if r_new == 0 else r_new
                    
                    if old_start <= L_max:
                        covered[old_start::p] = False
                    if new_start <= L_max:
                        covered[new_start::p] = True
                    
                    new_L = calc_L_from_bitset(covered, L_max)
                    
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r_new
                    else:
                        if new_start <= L_max:
                            covered[new_start::p] = False
                        if old_start <= L_max:
                            covered[old_start::p] = True
                
                if best_r != r:
                    current[i][1] = best_r
                    current_L = best_L
                    improved = True
            
            if not improved:
                break
        
        return current, current_L
    
    best_classes = None
    best_L = 0
    
    prime_subsets = [
        primes[:60],
        primes[:40],
        [p for p in primes if p >= 5][:50],
        [p for p in primes if p >= 7][:40],
        [p for p in primes if p >= 11][:30],
        [p for p in primes if p >= 13][:25],
        [p for p in primes if p >= 17][:20],
    ]
    
    L_max = 10000
    
    for ps in prime_subsets:
        if time.time() > deadline - 20:
            break
        
        classes = greedy_with_lookahead(ps, L_max)
        if not classes:
            continue
        
        classes, L = local_search_bitset(classes, L_max, max_iter=50)
        
        if L > best_L:
            best_L = L
            best_classes = classes
    
    if best_L < 500:
        test_configs = [
            [[2, 1], [3, 0], [5, 1], [7, 3], [11, 5], [13, 6]],
            [[2, 0], [3, 1], [5, 2], [7, 4], [11, 6], [13, 7]],
            [[2, 1], [3, 2], [5, 3], [7, 5], [11, 7], [13, 9]],
        ]
        for tc in test_configs:
            if time.time() > deadline - 10:
                break
            covered = build_coverage(tc, L_max)
            L = calc_L_from_bitset(covered, L_max)
            s = Fraction(0)
            for p, r in tc:
                s += Fraction(1, p)
            if s <= 1 and L > best_L:
                best_L = L
                best_classes = [list(c) for c in tc]
                classes, L2 = local_search_bitset(best_classes, L_max, max_iter=30)
                if L2 > best_L:
                    best_L = L2
                    best_classes = classes
    
    if best_classes is None:
        best_classes = [[2, 1], [3, 0]]
        best_L = 1
    
    L_check = min(best_L, 100000)
    covered = build_coverage(best_classes, L_check)
    if not np.all(covered[1:L_check+1]):
        uncov = np.where(~covered[1:L_check+1])[0]
        best_L = int(uncov[0]) if len(uncov) > 0 else 0
    
    return best_classes
