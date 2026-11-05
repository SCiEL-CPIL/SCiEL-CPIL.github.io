# EVOLVE-BLOCK-START
import time
import random
import itertools

def _set_sizes(A):
    """
    Compute |A+A| and |A*A| for a list A of distinct integers.
    Returns (sum_count, prod_count).
    """
    n = len(A)
    sums = set()
    prods = set()
    # Exploit symmetry i <= j
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _objective(A):
    s, p = _set_sizes(A)
    return max(s, p)

def _validate(A, n, max_val):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > max_val for x in A):
        return False
    return True

def _generate_ap(start, step, n, max_val):
    A = [start + i * step for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_gp(n, max_val):
    # Powers of 2
    A = [2**i for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_hybrid(n, max_val):
    # Small AP + Geometric tail
    A = set()
    # 1..10
    for i in range(1, 11):
        A.add(i)
    # Powers of 2 starting from 16
    v = 16
    while len(A) < n:
        if v <= max_val:
            A.add(v)
        v *= 2
    if len(A) >= n:
        return sorted(list(A)[:n])
    # Fallback fill
    c = 1
    while len(A) < n:
        if c not in A:
            A.add(c)
        c += 1
    return sorted(A)

def _generate_random(n, max_val):
    return sorted(random.sample(range(1, max_val + 1), n))

def _generate_log_spaced(n, max_val):
    return None

def _local_search_ap(A_init, n, max_val, deadline):
    """
    Perform local search starting from an AP, trying to reduce product set size.
    """
    current_A = A_init[:]
    current_cost = _objective(current_A)
    best_A = current_A[:]
    best_cost = current_cost
    
    # We want to reduce max(s, p). Usually p is the bottleneck for APs.
    # We try to replace elements with values that create more product collisions.
    # However, changing an element in an AP breaks the sumset structure, increasing s.
    # So we must be careful.
    
    # Strategy: Try swapping pairs or replacing single elements with nearby values.
    # Given time constraints, we will do a limited number of steps.
    
    steps = 0
    max_steps = 500
    
    while steps < max_steps and time.time() < deadline:
        steps += 1
        # Pick a random element to modify
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        # Generate a few candidate new values
        candidates = set()
        # Nearby values
        for delta in range(-10, 11):
            if delta == 0: continue
            nv = old_val + delta
            if 1 <= nv <= max_val:
                candidates.add(nv)
        
        # Random values
        for _ in range(5):
            candidates.add(random.randint(1, max_val))
            
        best_move_val = None
        best_move_cost = current_cost
        
        for nv in candidates:
            if nv == old_val:
                continue
            # Check distinctness
            if nv in current_A:
                continue
                
            current_A[idx] = nv
            new_cost = _objective(current_A)
            
            if new_cost < best_move_cost:
                best_move_cost = new_cost
                best_move_val = nv
            current_A[idx] = old_val # Revert for next candidate
            
        if best_move_val is not None:
            current_A[idx] = best_move_val
            current_cost = best_move_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            # If no improvement, maybe do a random jump to escape local minima?
            # But for now, just break or continue with random restarts in main loop.
            pass
            
    return best_A, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    n = 20
    max_val = 10000
    
    best_A = None
    best_cost = float('inf')
    
    # 1. Search over Arithmetic Progressions
    # A = a + i*d
    # We want a >= 1, a + 19*d <= 10000
    # d >= 1
    # a <= 10000 - 19*d
    
    # Iterate over small d and a
    # We prioritize small d as they keep the set compact, potentially allowing
    # more product collisions? Actually, for APs, the product set size is
    # determined by the number of distinct values of (a+id)(a+jd).
    # For small a, d, collisions are more likely?
    
    d_max = 500 # Limit search space
    for d in range(1, d_max + 1):
        a_max = max_val - 19 * d
        if a_max < 1:
            continue
            
        # Check a few values of a for this d
        # We can't check all a, so we sample
        # Or we can just pick a=1 for each d first
        A = _generate_ap(1, d, n, max_val)
        if A:
            cost = _objective(A)
            if cost < best_cost:
                best_cost = cost
                best_A = A[:]
                
        # Try a few other starts
        if d <= 20:
            for a in [1, 2, 3, 10, 100]:
                if a <= a_max:
                    A = _generate_ap(a, d, n, max_val)
                    if A:
                        cost = _objective(A)
                        if cost < best_cost:
                            best_cost = cost
                            best_A = A[:]
                            
        if time.time() > deadline - 10:
            break
            
    # 2. Local Search from the best AP found
    if best_A:
        improved_A, improved_cost = _local_search_ap(best_A, n, max_val, deadline)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_A = improved_A
    
    # 3. Try Hybrid Generators
    candidates = []
    candidates.append(_generate_hybrid(n, max_val))
    
    # Also try a "Geometric-like" set that fits
    # 1, 2, 4, ..., 2^13 (14 items) + 6 items from AP?
    # Let's just add a few random hybrids
    
    for c in candidates:
        if c and _validate(c, n, max_val):
            cost = _objective(c)
            if cost < best_cost:
                best_cost = cost
                best_A = c[:]
    
    # 4. Final Random Local Search if time permits
    if time.time() < deadline - 5:
        # Start from best_A and do more random swaps
        current_A = best_A[:]
        current_cost = best_cost
        
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, n - 1)
            old_val = current_A[idx]
            new_val = random.randint(1, max_val)
            if new_val == old_val or new_val in current_A:
                continue
            current_A[idx] = new_val
            new_cost = _objective(current_A)
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = current_A[:]
            else:
                current_A[idx] = old_val
                
    return best_A

# EVOLVE-BLOCK-END
