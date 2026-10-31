import math
import time
from collections import defaultdict

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 5.0

# Global state for dynamic feedback
_sym_count = [0, 0, 0]  # counts of 0, 1, 2 across all selected vectors
_total_selected = 0
_priority_cache = {}

def _time_check():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def priority(vector, n):
    global _sym_count, _total_selected
    
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    if _time_check():
        # Fallback: simple deterministic score
        h = 0
        for val in vector:
            h = (h * 31 + val) % (10**9 + 7)
        result = float(h / (10**9 + 7))
        _priority_cache[key] = result
        return result
    
    # Count symbols in this vector
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Current distribution (before adding this vector)
    if _total_selected == 0:
        cur_dist = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    else:
        cur_dist = [c / _total_selected for c in _sym_count]
    
    # Distribution after hypothetically adding this vector
    new_sym_count = [_sym_count[0] + c0, _sym_count[1] + c1, _sym_count[2] + c2]
    new_total = _total_selected + 1
    new_dist = [c / new_total for c in new_sym_count]
    
    # Ideal asymptotic distribution for cap sets is uniform
    ideal = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    
    # Marginal skew risk: how much does adding this vector move away from ideal?
    # Use squared L2 distance change
    risk = 0.0
    for i in range(3):
        risk += (new_dist[i] - ideal[i]) ** 2 - (cur_dist[i] - ideal[i]) ** 2
    
    # We want to minimize risk. If risk is negative, it moves toward ideal (good)
    # If positive, it moves away (bad)
    
    # Additionally, consider the absolute level of skew
    abs_skew = 0.0
    for i in range(3):
        abs_skew += abs(new_dist[i] - ideal[i])
    
    # Combine: prefer vectors that reduce or don't increase skew
    # Use a weighted combination
    balance_score = -risk * 10000.0 - abs_skew * 1000.0
    
    # Deterministic tie-breaker using a polynomial hash
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    result = float(balance_score + hash_score)
    _priority_cache[key] = result
    
    # Update global state: assume this vector will be selected
    # In a greedy algorithm, the highest-priority vector is added next
    # We don't know for sure if this exact vector will be added, but we update
    # based on the assumption that the evaluator adds vectors in descending priority
    # For now, we update with this vector's contribution
    _sym_count[0] += c0
    _sym_count[1] += c1
    _sym_count[2] += c2
    _total_selected += 1
    
    return result
