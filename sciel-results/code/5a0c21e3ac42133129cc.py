"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _objective_np(A):
    """
    Compute max(|A+A|, |A*A|) using NumPy for speed.
    A is a list of 20 distinct positive integers.
    """
    A_np = np.array(A, dtype=np.int64)
    sums = A_np[:, None] + A_np[None, :]
    sum_size = len(np.unique(sums))
    
    prods = A_np[:, None] * A_np[None, :]
    prod_size = len(np.unique(prods))
    
    return max(sum_size, prod_size)

def _is_valid(A):
    """Check 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_structured_candidates():
    """
    Generate initial candidates from structured sets to exploit collisions.
    Uses union of arithmetic and geometric progressions.
    """
    candidates = []
    
    # Strategy: Union of two structured sub-sets
    # Sub-set 1: Arithmetic progression {a + i*d} for i in 0..n1-1
    # Sub-set 2: Geometric progression {b * r^j} for j in 0..n2-1
    # where n1 + n2 = 20, and the two sets are disjoint
    
    # Try various splits
    for n1 in range(5, 16):  # Arithmetic part length
        n2 = 20 - n1          # Geometric part length
        
        # Enumerate arithmetic parameters
        for a in range(1, 11):  # Start
            for d in range(1, 6):  # Step
                arith = [a + i*d for i in range(n1)]
                if max(arith) > 10000:
                    continue
                
                # Enumerate geometric parameters
                for b in range(1, 11):  # Start
                    for r in [2, 3, 4, 5]:  # Ratio
                        geo = []
                        val = b
                        for j in range(n2):
                            if val > 10000:
                                break
                            geo.append(val)
                            val *= r
                        
                        if len(geo) != n2:
                            continue
                        
                        # Check disjointness and validity
                        combined = sorted(arith + geo)
                        if len(set(combined)) != 20:
                            continue
                        if any(x < 1 or x > 10000 for x in combined):
                            continue
                        
                        candidates.append(combined)
    
    # Also try pure geometric and pure arithmetic as baselines
    for a in [1, 2, 3]:
        for r in [2, 3]:
            seq = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                seq.append(val)
                val *= r
            if len(seq) == 20:
                candidates.append(sorted(seq))
    
    for a in [1, 2, 3]:
        for d in [1, 2, 3]:
            seq = [a + i*d for i in range(20)]
            if max(seq) <= 10000:
                candidates.append(seq)
    
    # Add some random structured candidates
    for _ in range(10):
        n1 = random.randint(5, 15)
        n2 = 20 - n1
        a = random.randint(1, 10)
        d = random.randint(1, 5)
        arith = [a + i*d for i in range(n1)]
        b = random.randint(1, 10)
        r = random.choice([2, 3, 4])
        geo = []
        val = b
        for j in range(n2):
            if val > 10000:
                break
            geo.append(val)
            val *= r
        if len(geo) == n2:
            combined = sorted(arith + geo)
            if len(set(combined)) == 20 and max(combined) <= 10000:
                candidates.append(combined)
    
    return candidates

def _nudge_move(A):
    """Nudge one element by a small amount."""
    idx = random.randrange(20)
    delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
    new_val = A[idx] + delta
    if new_val < 1 or new_val > 10000:
        return None
    if new_val in A:
        return None
    A[idx] = new_val
    return sorted(A)

def _swap_move(A):
    """Swap one element with a random new value, biased towards small numbers."""
    idx = random.randrange(20)
    old = A[idx]
    while True:
        if random.random() < 0.7:
            new = random.randint(1, 50)
        else:
            new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves using vectorized objective."""
    current = list(A)
    current_cost = _objective_np(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.6:
            new_A = _nudge_move(list(current))
            if new_A is None:
                continue
        else:
            new_A = _swap_move(list(current))
        
        new_cost = _objective_np(new_A)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
        else:
            if temp > 1e-9:
                prob = math.exp(-delta / temp)
            else:
                prob = 0.0
            accept = random.random() < prob
            
        if accept:
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    random.seed(42)
    
    # Phase 1: Generate structured initial solutions
    best_set = None
    best_cost = float('inf')
    
    candidates = _generate_structured_candidates()
    for A in candidates:
        if time.time() > deadline:
            break
        if not _is_valid(A):
            continue
        cost = _objective_np(A)
        if cost < best_cost:
            best_cost = cost
            best_set = list(A)
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective_np(best_set)

    # Phase 2: Simulated annealing
    start_temp = 10.0
    end_temp = 0.01
    steps = 8000
    
    result, cost = _anneal(best_set, start_temp, end_temp, steps, deadline)
    if cost < best_cost:
        best_cost = cost
        best_set = list(result)
        
    # Phase 3: Local refinement (Hill Climbing)
    if time.time() < deadline - 5:
        current = list(best_set)
        current_cost = best_cost
        improved = True
        max_iters = 1500
        iter_count = 0
        while improved and time.time() < deadline - 2 and iter_count < max_iters:
            improved = False
            iter_count += 1
            indices = list(range(20))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline - 2:
                    break
                old = current[idx]
                deltas = [-10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, -20]
                random.shuffle(deltas)
                for delta in deltas:
                    new_val = old + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current:
                        continue
                    current[idx] = new_val
                    new_cost = _objective_np(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_set = list(current)
                        break
                    else:
                        current[idx] = old

    return best_set

# EVOLVE-BLOCK-END
