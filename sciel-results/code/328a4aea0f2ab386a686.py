"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction

N = 512
WEIGHTS = [0.0] * (N + 1)
EXACT_WEIGHTS = [Fraction(0, 1)] * (N + 1)
for i in range(1, N):
    WEIGHTS[i] = 1.0 / (N - i)
    EXACT_WEIGHTS[i] = Fraction(1, N - i)

# Precompute prime factors for each number 1..511 to speed up coprimality checks
def _factorize(x):
    factors = set()
    d = 2
    while d * d <= x:
        if x % d == 0:
            factors.add(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        factors.add(x)
    return factors

FACTORS = [set() for _ in range(N)]
for i in range(1, N):
    FACTORS[i] = _factorize(i)

def _is_coprime_to_set(a, current_set):
    """Check if a is coprime to all elements in current_set using precomputed factors."""
    fa = FACTORS[a]
    if not fa:
        return True
    for x in current_set:
        if fa & FACTORS[x]:
            return False
    return True

def _score_set_float(current_set):
    return sum(WEIGHTS[a] for a in current_set)

def _score_set_exact(current_set):
    return sum(EXACT_WEIGHTS[a] for a in current_set)

def _local_search(initial_set, n=N, time_limit=10.0):
    """Perform local search (hill climbing) starting from initial_set."""
    start_time = time.time()
    current_set = list(initial_set)
    current_score = _score_set_float(current_set)
    
    # Candidates: all numbers 1..n-1
    all_candidates = list(range(1, n))
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        
        # 1. Try adding a single element (highest weight first)
        # We maintain a sorted list of candidates by weight descending for efficiency
        # But since weights are fixed, we can just iterate 511 down to 1
        best_add_candidate = None
        best_add_gain = 0.0
        for a in range(n - 1, 0, -1):
            if a not in current_set and _is_coprime_to_set(a, current_set):
                gain = WEIGHTS[a]
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_candidate = a
                else:
                    # Since we are iterating descending, if gain <= best_add_gain, 
                    # and weights are strictly decreasing with a, 
                    # wait, weights are 1/(512-a). 
                    # a=511 -> 1/1 = 1.0
                    # a=510 -> 1/2 = 0.5
                    # So weights decrease as a decreases.
                    # If we iterate from 511 down to 1, the first valid one is the best.
                    # So we can just break?
                    # No, we need to check if it's valid. If valid, it's the best possible add.
                    # So we can just return the first valid one found in descending order.
                    pass
        
        if best_add_candidate is not None:
            current_set.append(best_add_candidate)
            current_score += best_add_gain
            improved = True
            continue
        
        # 2. Try swapping one element
        # Sort current set by weight ascending (remove low weight first)
        # Sort candidates not in set by weight descending (add high weight first)
        current_sorted = sorted(current_set, key=lambda x: WEIGHTS[x])
        candidates_not_in_set = [a for a in range(n - 1, 0, -1) if a not in current_set]
        
        swap_found = False
        for a_remove in current_sorted:
            if swap_found:
                break
            weight_remove = WEIGHTS[a_remove]
            temp_set = [x for x in current_set if x != a_remove]
            
            for a_add in candidates_not_in_set:
                weight_add = WEIGHTS[a_add]
                if weight_add <= weight_remove:
                    # Since candidates are sorted descending, if this one is not better, none after are.
                    break
                
                if _is_coprime_to_set(a_add, temp_set):
                    # Valid swap
                    new_score = current_score - weight_remove + weight_add
                    current_set = temp_set
                    current_set.append(a_add)
                    current_score = new_score
                    improved = True
                    swap_found = True
                    break
    
    return current_set, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_set = []
    best_score = 0.0
    
    # Strategy: Multiple restarts with different greedy orderings
    # 1. Descending a (original greedy)
    # 2. Ascending a
    # 3. Randomized
    # 4. Priority based on "flexibility" (number of coprime partners)
    
    # Precompute coprime counts for flexibility heuristic
    coprime_count = [0] * N
    for a in range(1, N):
        for b in range(1, N):
            if a != b and _is_coprime_to_set(a, [b]):
                coprime_count[a] += 1
    
    orderings = []
    
    # Ordering 1: Descending a (highest weight first)
    orderings.append(list(range(N - 1, 0, -1)))
    
    # Ordering 2: Ascending a (lowest weight first, but might allow more elements?)
    orderings.append(list(range(1, N)))
    
    # Ordering 3: Sorted by flexibility (most coprime partners first)
    orderings.append(sorted(range(1, N), key=lambda x: -coprime_count[x]))
    
    # Ordering 4: Hybrid - high weight, but broken by flexibility
    orderings.append(sorted(range(1, N), key=lambda x: (-WEIGHTS[x], -coprime_count[x])))
    
    # Ordering 5: Randomized
    for _ in range(5):
        order = list(range(1, N))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = []
        for a in order:
            if _is_coprime_to_set(a, current_set):
                current_set.append(a)
        
        # Local search refinement
        refined_set, refined_score = _local_search(current_set, n=N, time_limit=5.0)
        
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    # Final polish: Try to improve the best set found with more aggressive local search
    if best_set:
        refined_set, refined_score = _local_search(best_set, n=N, time_limit=10.0)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    return best_set
# EVOLVE-BLOCK-END
