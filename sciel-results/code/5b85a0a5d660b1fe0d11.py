import time
import sys
from itertools import combinations

_N = 5040
_TARGET = 4979
_DEADLINE = 85.0

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def _check_k1(divisors, target):
    if target in divisors:
        return [target]
    return None

def _check_k2(divisors, target):
    div_set = set(divisors)
    for d in divisors:
        rem = target - d
        if rem in div_set and rem > d:
            return [d, rem]
    return None

def _check_k3(divisors, target):
    div_set = set(divisors)
    n = len(divisors)
    for i in range(n):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem1 = target - d1
        for j in range(i + 1, n):
            d2 = divisors[j]
            if d1 + d2 >= target:
                break
            rem2 = target - d1 - d2
            if rem2 in div_set and rem2 > d2:
                return [d1, d2, rem2]
    return None

def _check_k4(divisors, target):
    n = len(divisors)
    half = n // 2
    left = divisors[:half]
    right = divisors[half:]
    
    # Generate all pair sums from left
    left_sums = {}
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            s = left[i] + left[j]
            if s > target:
                continue
            if s not in left_sums:
                left_sums[s] = (left[i], left[j])
    
    # Generate all pair sums from right and check
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            s = right[i] + right[j]
            if s > target:
                continue
            rem = target - s
            if rem in left_sums:
                l1, l2 = left_sums[rem]
                # Ensure all 4 are distinct
                if len({l1, l2, right[i], right[j]}) == 4:
                    return [l1, l2, right[i], right[j]]
    return None

def _check_k5(divisors, target):
    n = len(divisors)
    half = n // 2
    left = divisors[:half]
    right = divisors[half:]
    
    # Generate all pair sums from left
    left_pairs = {}
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            s = left[i] + left[j]
            if s > target:
                continue
            if s not in left_pairs:
                left_pairs[s] = (left[i], left[j])
    
    # Generate all triple sums from right and check
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            for k in range(j + 1, len(right)):
                s = right[i] + right[j] + right[k]
                if s > target:
                    break
                rem = target - s
                if rem in left_pairs:
                    l1, l2 = left_pairs[rem]
                    if len({l1, l2, right[i], right[j], right[k]}) == 5:
                        return [l1, l2, right[i], right[j], right[k]]
    return None

def _check_k6(divisors, target):
    n = len(divisors)
    half = n // 2
    left = divisors[:half]
    right = divisors[half:]
    
    # Generate all triple sums from left
    left_triples = {}
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            for k in range(j + 1, len(left)):
                s = left[i] + left[j] + left[k]
                if s > target:
                    break
                if s not in left_triples:
                    left_triples[s] = (left[i], left[j], left[k])
    
    # Generate all triple sums from right and check
    for i in range(len(right)):
        for j in range(i + 1, len(right)):
            for k in range(j + 1, len(right)):
                s = right[i] + right[j] + right[k]
                if s > target:
                    break
                rem = target - s
                if rem in left_triples:
                    l1, l2, l3 = left_triples[rem]
                    if len({l1, l2, l3, right[i], right[j], right[k]}) == 6:
                        return [l1, l2, l3, right[i], right[j], right[k]]
    return None

def construct():
    start = time.time()
    divisors = _divisors_of(_N)
    # Filter divisors that are too large
    divisors = [d for d in divisors if d <= _TARGET]
    
    # Check k=1 to k=6
    for k in range(1, 7):
        if time.time() > _DEADLINE - 5:
            break
        if k == 1:
            res = _check_k1(divisors, _TARGET)
        elif k == 2:
            res = _check_k2(divisors, _TARGET)
        elif k == 3:
            res = _check_k3(divisors, _TARGET)
        elif k == 4:
            res = _check_k4(divisors, _TARGET)
        elif k == 5:
            res = _check_k5(divisors, _TARGET)
        elif k == 6:
            res = _check_k6(divisors, _TARGET)
        
        if res is not None:
            return res
            
    # Fallback: DP for min cardinality subset sum
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    dp[0] = 0
    
    for d in divisors:
        if time.time() > _DEADLINE - 2:
            break
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                
    if dp[_TARGET] != INF:
        result = []
        s = _TARGET
        valid = True
        while s > 0:
            d = parent[s]
            if d == -1:
                valid = False
                break
            result.append(d)
            s -= d
        if valid and sum(result) == _TARGET and len(set(result)) == len(result):
            return result
            
    return []
