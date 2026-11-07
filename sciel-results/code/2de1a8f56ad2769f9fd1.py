import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = 256
    
    # 1. Generate primes up to N
    def _primes_up_to(n):
        if n < 2:
            return []
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:n + 1:i] = bytearray(len(sieve[i * i:n + 1:i]))
        return [i for i in range(2, n + 1) if sieve[i]]
        
    primes = _primes_up_to(N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Precompute prime factor bitmasks for each number 1..N
    # mask[x] = bitmask where bit i is set if prime i divides x
    mask = [0] * (N + 1)
    for x in range(2, N + 1):
        temp = x
        m = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= (1 << prime_to_idx[temp])
        mask[x] = m
    
    # 3. Attempt to solve using ILP (scipy.optimize.milp)
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        c = np.array([-float(i) for i in range(1, N + 1)])
        
        A = np.zeros((num_primes, N), dtype=float)
        for x in range(1, N + 1):
            idx = x - 1
            m = mask[x]
            for p_idx in range(num_primes):
                if (m >> p_idx) & 1:
                    A[p_idx, idx] = 1.0
                    
        b_lower = np.zeros(num_primes)
        b_upper = np.ones(num_primes)
        
        constraints = LinearConstraint(A, b_lower, b_upper)
        
        lb = np.zeros(N)
        ub = np.ones(N)
        bounds = Bounds(lb, ub)
        
        integrality = np.ones(N)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            sol = res.x
            result = []
            for i in range(N):
                if sol[i] > 0.5:
                    result.append(i + 1)
            
            if _verify(result, N):
                return result
    except Exception:
        pass
        
    # 4. Fallback: Bitmask-based Greedy + Comprehensive Local Search
    
    def greedy_init(order):
        """Greedy initialization with given ordering of candidates."""
        selected = set()
        used_mask = 0
        for x in order:
            if time.time() > deadline:
                break
            if mask[x] & used_mask == 0:
                selected.add(x)
                used_mask |= mask[x]
        return selected
    
    # Try multiple greedy orderings and keep the best
    best_selected = None
    best_sum = 0
    
    # Ordering 1: descending by value
    order1 = list(range(N, 0, -1))
    sel1 = greedy_init(order1)
    sum1 = sum(sel1)
    if sum1 > best_sum:
        best_sum = sum1
        best_selected = sel1
    
    # Ordering 2: descending by value, but skip numbers with many prime factors first
    # Actually, let's try ascending by number of prime factors, then descending by value
    def num_prime_factors(x):
        return bin(mask[x]).count('1')
    
    order2 = sorted(range(N, 0, -1), key=lambda x: (num_prime_factors(x), -x))
    sel2 = greedy_init(order2)
    sum2 = sum(sel2)
    if sum2 > best_sum:
        best_sum = sum2
        best_selected = sel2
    
    # Ordering 3: descending by value / number of prime factors (efficiency)
    order3 = sorted(range(N, 0, -1), key=lambda x: -x / max(1, num_prime_factors(x)))
    sel3 = greedy_init(order3)
    sum3 = sum(sel3)
    if sum3 > best_sum:
        best_sum = sum3
        best_selected = sel3
    
    # Ordering 4: Primorial-friendly numbers first (numbers with fewer/smaller prime factors)
    # Sort by: (num_prime_factors, sum of prime factors, -value)
    def prime_factor_sum(x):
        s = 0
        m = mask[x]
        for p_idx in range(num_primes):
            if (m >> p_idx) & 1:
                s += primes[p_idx]
        return s
    
    order4 = sorted(range(N, 0, -1), key=lambda x: (num_prime_factors(x), prime_factor_sum(x), -x))
    sel4 = greedy_init(order4)
    sum4 = sum(sel4)
    if sum4 > best_sum:
        best_sum = sum4
        best_selected = sel4
    
    selected = best_selected
    
    # Comprehensive Local Search with bitmask conflict checking
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_sum = sum(selected)
        
        # Move 1: 1-for-1 swap (remove one selected, add one unselected)
        for u in range(N, 0, -1):
            if time.time() > deadline:
                break
            if u in selected:
                continue
            
            # Find which selected numbers conflict with u
            to_remove = []
            for s in selected:
                if mask[s] & mask[u] != 0:
                    to_remove.append(s)
            
            gain = u - sum(to_remove)
            if gain > 0:
                for s in to_remove:
                    selected.remove(s)
                selected.add(u)
                improved = True
                break
        
        if improved:
            continue
        
        # Move 2: 1-for-2 (remove two selected, add one unselected)
        for u in range(N, 0, -1):
            if time.time() > deadline:
                break
            if u in selected:
                continue
            
            # Find all selected numbers that conflict with u
            conflicting = [s for s in selected if mask[s] & mask[u] != 0]
            
            # Try removing subsets of size 1 and 2 from conflicting
            # Size 1 is already covered by Move 1, so try size 2
            if len(conflicting) >= 2:
                # Try all pairs of conflicting numbers
                for i in range(len(conflicting)):
                    for j in range(i + 1, len(conflicting)):
                        s1 = conflicting[i]
                        s2 = conflicting[j]
                        # Check if removing s1 and s2 frees up all conflicts with u
                        # i.e., no other selected number conflicts with u
                        remaining_conflicts = False
                        for s in selected:
                            if s == s1 or s == s2:
                                continue
                            if mask[s] & mask[u] != 0:
                                remaining_conflicts = True
                                break
                        
                        if not remaining_conflicts:
                            gain = u - s1 - s2
                            if gain > 0:
                                selected.remove(s1)
                                selected.remove(s2)
                                selected.add(u)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue
        
        # Move 3: 2-for-1 (remove one selected, add two unselected)
        for s in list(selected):
            if time.time() > deadline:
                break
            
            # Find unselected numbers that don't conflict with each other or with (selected - {s})
            remaining_mask = 0
            for t in selected:
                if t != s:
                    remaining_mask |= mask[t]
            
            # Find candidates that don't conflict with remaining selected
            candidates = []
            for u in range(N, 0, -1):
                if u in selected:
                    continue
                if mask[u] & remaining_mask == 0:
                    candidates.append(u)
            
            # Try pairs of candidates that don't conflict with each other
            for i in range(len(candidates)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(candidates)):
                    u1 = candidates[i]
                    u2 = candidates[j]
                    if mask[u1] & mask[u2] != 0:
                        continue
                    gain = u1 + u2 - s
                    if gain > 0:
                        selected.remove(s)
                        selected.add(u1)
                        selected.add(u2)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
        
        # Move 4: 2-for-2 swap
        # This is more expensive, so only do it if we have time
        if time.time() < deadline - 5:
            for s1 in list(selected):
                if time.time() > deadline - 3:
                    break
                for s2 in list(selected):
                    if s2 <= s1:
                        continue
                    if time.time() > deadline - 3:
                        break
                    
                    # Find unselected numbers that don't conflict with (selected - {s1, s2})
                    remaining_mask = 0
                    for t in selected:
                        if t != s1 and t != s2:
                            remaining_mask |= mask[t]
                    
                    candidates = []
                    for u in range(N, 0, -1):
                        if u in selected:
                            continue
                        if mask[u] & remaining_mask == 0:
                            candidates.append(u)
                    
                    # Try pairs of candidates that don't conflict with each other
                    for i in range(len(candidates)):
                        if time.time() > deadline - 3:
                            break
                        for j in range(i + 1, len(candidates)):
                            u1 = candidates[i]
                            u2 = candidates[j]
                            if mask[u1] & mask[u2] != 0:
                                continue
                            gain = u1 + u2 - s1 - s2
                            if gain > 0:
                                selected.remove(s1)
                                selected.remove(s2)
                                selected.add(u1)
                                selected.add(u2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
    
    result = list(selected)
    
    if not result:
        result = [1]
        
    return [int(x) for x in result]

def _verify(result, N):
    if not result:
        return False
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool) or x < 1 or x > N:
            return False
    if len(result) != len(set(result)):
        return False
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                return False
    return True
