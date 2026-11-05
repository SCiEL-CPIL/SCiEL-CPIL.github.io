"""Optimized program for sum-product extremal sets using geometric progressions."""
import time
from math import log, exp, gcd
from itertools import combinations
import random

# EVOLVE-BLOCK-START
def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sum_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sum_set.add(ai + A[j])
    prod_set = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prod_set.add(ai * A[j])
    return max(len(sum_set), len(prod_set))

def _geometric_progression(a, r, n=20, max_val=10000):
    """Generate geometric progression a, a*r, a*r^2, ..., a*r^(n-1)."""
    A = []
    val = a
    for _ in range(n):
        if val > max_val:
            return None
        A.append(val)
        val *= r
    return A

def _search_geometric_progressions(time_budget):
    """Search over geometric progressions to minimize objective."""
    start_time = time.time()
    best_obj = float('inf')
    best_A = list(range(1, 21))
    
    for r in range(2, 100):
        if time.time() - start_time > time_budget - 5:
            break
        
        max_a = 10000 // (r ** 19)
        if max_a < 1:
            continue
        
        a_values = range(1, max_a + 1)
        
        for a in a_values:
            if time.time() - start_time > time_budget - 5:
                break
            A = _geometric_progression(a, r, 20)
            if A is None or len(set(A)) != 20:
                continue
            obj = _compute_objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = A
    
    return best_A, best_obj

def _simulated_annealing(A, time_budget):
    """Perform simulated annealing to improve the solution."""
    start_time = time.time()
    current_obj = _compute_objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    n = len(A)
    max_val = 10000
    
    temperature = 10.0
    cooling_rate = 0.995
    
    while temperature > 0.1 and (time.time() - start_time < time_budget - 5):
        i = random.randint(0, n - 1)
        old_val = A[i]
        
        # Try a random perturbation
        if random.random() < 0.3:
            # Large jump
            delta = random.randint(-200, 200)
        else:
            # Small jump
            delta = random.randint(-30, 30)
        
        new_val = old_val + delta
        if new_val < 1 or new_val > max_val:
            temperature *= cooling_rate
            continue
        if new_val in A:
            temperature *= cooling_rate
            continue
        
        new_A = list(A)
        new_A[i] = new_val
        new_A = sorted(new_A)
        
        new_obj = _compute_objective(new_A)
        delta_obj = new_obj - current_obj
        
        if delta_obj < 0 or random.random() < exp(-delta_obj / temperature):
            A = new_A
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = list(A)
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def construct():
    """Construct 20 distinct positive integers <= 10000 minimizing max(|A+A|, |A*A|)."""
    start_time = time.time()
    deadline = start_time + 50
    
    # Start with geometric progression search
    best_A, best_obj = _search_geometric_progressions(15)
    
    # Try simulated annealing if time permits
    if time.time() < deadline - 5:
        sa_budget = min(30, deadline - start_time - 5)
        best_A, best_obj = _simulated_annealing(best_A, sa_budget)
    
    # Verify validity
    if len(best_A) != 20 or len(set(best_A)) != 20 or any(x < 1 or x > 10000 for x in best_A):
        best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
