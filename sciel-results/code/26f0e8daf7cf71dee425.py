"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _compute_cost(A):
    """
    Compute max(|A+A|, |A*A|) for a list A of distinct integers.
    """
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _compute_cost_detailed(A):
    """Return (sum_size, prod_size, max_size) for a set A."""
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    ss, ps = len(sums), len(prods)
    return (ss, ps, max(ss, ps))

def _generate_ap(start=1, step=1, n=20, max_val=10000):
    A = [start + i * step for i in range(n)]
    if max(A) <= max_val:
        return sorted(A)
    return None

def _generate_gp(n=20, max_val=10000, base=2):
    """Generate a geometric progression with given base."""
    A = set()
    val = 1
    while len(A) < n and val <= max_val:
        A.add(val)
        val *= base
    if len(A) >= n:
        return sorted(list(A)[:n])
    # Fill remaining
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(A)

def _generate_log_spaced(n=20, max_val=10000):
    """Generate log-spaced integers."""
    A = set()
    # Use logarithmic spacing
    for i in range(n):
        val = int(1 + i * (max_val - 1) / (n - 1))
        A.add(max(1, val))
    # Ensure we have exactly n distinct values
    if len(A) < n:
        candidate = 1
        while len(A) < n:
            if candidate not in A:
                A.add(candidate)
            candidate += 1
    return sorted(list(A)[:n])

def _generate_power_mix(n=20, max_val=10000):
    """Mix of small consecutive and larger powers."""
    A = set()
    # Small consecutive block
    small_count = n // 2
    for i in range(1, small_count + 1):
        A.add(i)
    # Larger values with multiplicative structure
    base_vals = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    for bv in base_vals:
        if len(A) >= n:
            break
        # Add bv and some multiples
        for mult in [1, 2, 3]:
            v = bv * mult
            if v <= max_val and v not in A:
                A.add(v)
                if len(A) >= n:
                    break
    # Fill if needed
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(list(A)[:n])

def _generate_mixed(n=20, max_val=10000):
    """Generate a mix of small AP, medium GP, and large random."""
    A = set()
    # Small AP: 1..8
    for i in range(1, 9):
        A.add(i)
    # Medium GP: 10, 20, 40, 80
    for v in [10, 20, 40, 80]:
        if v not in A:
            A.add(v)
    # Large spaced
    for v in [100, 300, 700, 1500, 4000, 8000]:
        if len(A) >= n:
            break
        if v not in A:
            A.add(v)
    # Fill
    candidate = 1
    while len(A) < n:
        if candidate not in A:
            A.add(candidate)
        candidate += 1
    return sorted(list(A)[:n])

def _local_search(current_A, n, max_val, deadline, best_A, best_cost, max_steps=3000):
    """
    Perform local search starting from current_A.
    Returns (best_A, best_cost, current_A, current_cost).
    """
    current_cost = _compute_cost(current_A)
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_steps:
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        r = random.random()
        if r < 0.3:
            # Small perturbation
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        elif r < 0.5:
            # Medium perturbation
            delta = random.randint(-100, 100)
            new_val = old_val + delta
        elif r < 0.7:
            # Replace with random value in same order of magnitude
            if old_val <= 10:
                new_val = random.randint(1, 50)
            elif old_val <= 100:
                new_val = random.randint(10, 500)
            elif old_val <= 1000:
                new_val = random.randint(100, 5000)
            else:
                new_val = random.randint(1000, max_val)
        else:
            # Fully random
            new_val = random.randint(1, max_val)
        
        if new_val < 1:
            new_val = 1
        elif new_val > max_val:
            new_val = max_val
            
        if new_val == old_val:
            no_improve += 1
            continue
        
        # Check distinctness
        is_distinct = True
        for i, v in enumerate(current_A):
            if i != idx and v == new_val:
                is_distinct = False
                break
        
        if not is_distinct:
            no_improve += 1
            continue
        
        # Apply
        current_A[idx] = new_val
        new_cost = _compute_cost(current_A)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            current_A[idx] = old_val
            no_improve += 1
    
    return best_A, best_cost, current_A, current_cost

def _simulated_annealing(current_A, n, max_val, deadline, best_A, best_cost, max_steps=2000):
    """
    Simulated annealing to escape local minima.
    """
    current_cost = _compute_cost(current_A)
    temperature = 10.0
    cooling = 0.995
    
    for _ in range(max_steps):
        if time.time() >= deadline:
            break
        
        idx = random.randint(0, n - 1)
        old_val = current_A[idx]
        
        delta = random.randint(-20, 20)
        new_val = old_val + delta
        if new_val < 1:
            new_val = 1
        elif new_val > max_val:
            new_val = max_val
        
        if new_val == old_val:
            continue
        
        # Check distinctness
        is_distinct = True
        for i, v in enumerate(current_A):
            if i != idx and v == new_val:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
        
        current_A[idx] = new_val
        new_cost = _compute_cost(current_A)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_A = current_A[:]
        else:
            current_A[idx] = old_val
        
        temperature *= cooling
    
    return best_A, best_cost, current_A, current_cost

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 20
    max_val = 10000
    
    # Generate diverse initial candidates
    candidates = []
    
    # Basic structures
    candidates.append(_generate_ap(1, 1, n, max_val))       # 1..20
    candidates.append(_generate_ap(1, 2, n, max_val))       # 1,3,5,...,39
    candidates.append(_generate_ap(10, 1, n, max_val))      # 10..29
    candidates.append(_generate_ap(1, 5, n, max_val))       # 1,6,11,...
    
    # Geometric progressions
    candidates.append(_generate_gp(n, max_val, base=2))     # 1,2,4,8,...
    candidates.append(_generate_gp(n, max_val, base=3))     # 1,3,9,27,...
    candidates.append(_generate_gp(n, max_val, base=10))    # 1,10,100,...
    
    # Log-spaced
    candidates.append(_generate_log_spaced(n, max_val))
    
    # Mixed structures
    candidates.append(_generate_power_mix(n, max_val))
    candidates.append(_generate_mixed(n, max_val))
    
    # Random samples with bias toward small values
    for _ in range(5):
        A = set()
        # Bias: sample from 1..100 heavily, then 100..10000
        while len(A) < n:
            if random.random() < 0.7:
                A.add(random.randint(1, 100))
            else:
                A.add(random.randint(1, max_val))
        candidates.append(sorted(list(A)[:n]))
    
    # Pure random
    for _ in range(5):
        candidates.append(_generate_random(n, max_val) if ' _generate_random' in dir() else sorted(random.sample(range(1, max_val + 1), n)))
    
    # Evaluate all candidates
    best_A = None
    best_cost = float('inf')
    
    for cand in candidates:
        if cand is None:
            continue
        cost = _compute_cost(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = cand[:]
    
    # Local search from best candidate
    if best_A is not None:
        best_A, best_cost, _, _ = _local_search(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=2000)
    
    # Try local search from other good candidates
    for cand in candidates[:8]:
        if time.time() >= deadline:
            break
        if cand is None:
            continue
        cost = _compute_cost(cand)
        if cost <= best_cost + 5:  # Only search from reasonably good starts
            best_A, best_cost, _, _ = _local_search(cand[:], n, max_val, deadline, best_A, best_cost, max_steps=1000)
    
    # Simulated annealing from best
    if time.time() < deadline - 5:
        best_A, best_cost, _, _ = _simulated_annealing(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=1500)
    
    # Final local search polish
    if time.time() < deadline - 2:
        best_A, best_cost, _, _ = _local_search(best_A[:], n, max_val, deadline, best_A, best_cost, max_steps=1000)
    
    return best_A

# EVOLVE-BLOCK-END
