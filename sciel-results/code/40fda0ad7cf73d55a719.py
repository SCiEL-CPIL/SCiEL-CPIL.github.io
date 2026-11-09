"""Rewritten program: systematic search to pack distinct multiples of fixed divisors."""
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0
    
    # Initial solution: the divisors themselves
    best_assign = divisors[:]
    best_val = max(best_assign) - min(best_assign)  # 33
    
    # Search for better solutions by trying smaller spans
    for W in range(9, best_val + 1):
        if time.time() > deadline:
            break
        
        # Number of attempts: more for smaller W
        if W <= 12:
            num_attempts = 1000
        elif W <= 18:
            num_attempts = 500
        elif W <= 25:
            num_attempts = 200
        else:
            num_attempts = 50
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            L = random.randint(1, MAX_V - W)
            
            # Check if each divisor has at least one multiple in [L, L+W]
            candidates = []
            valid = True
            for d in divisors:
                k1 = (L + d - 1) // d  # ceil(L/d)
                k2 = (L + W) // d       # floor((L+W)/d)
                if k1 > k2:
                    valid = False
                    break
                candidates.append((k1, k2))
            
            if not valid:
                continue
            
            # Try to assign distinct values using backtracking
            # Sort by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: candidates[i][1] - candidates[i][0])
            used = set()
            assignment = [0] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return None
                if idx == n:
                    return assignment[:]
                i = order[idx]
                d = divisors[i]
                k1, k2 = candidates[i]
                for k in range(k1, k2 + 1):
                    val = k * d
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        result = backtrack(idx + 1)
                        if result is not None:
                            return result
                        used.remove(val)
                return None
            
            result = backtrack(0)
            if result is not None:
                actual_span = max(result) - min(result)
                if actual_span < best_val:
                    best_val = actual_span
                    best_assign = result[:]
                if best_val == W:
                    break
        
        if best_val == W:
            break
    
    # Local search: try to improve the current best solution
    if best_assign is not None and time.time() < deadline:
        for _ in range(500):
            if time.time() > deadline:
                break
            candidate = best_assign[:]
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]
            for delta_k in [-3, -2, -1, 1, 2, 3]:
                new_val = current + delta_k * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                if new_val in candidate:
                    continue
                candidate[idx] = new_val
                if len(set(candidate)) == n:
                    span = max(candidate) - min(candidate)
                    if span < best_val:
                        best_val = span
                        best_assign = candidate[:]
                        break
                candidate[idx] = current
    
    # Final validation
    result = [int(v) for v in best_assign]
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int)
        assert not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result
