"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score(A):
    """Compute max(|A+A|, |A*A|) for a set A of integers."""
    A = list(A)
    n = len(A)
    # Sumset A+A with equal summands allowed
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
    # Product set A*A with equal factors allowed
    prods = set()
    for i in range(n):
        for j in range(i, n):
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _geometric_set(base_ratio, n=20, start=1):
    """Generate a geometric sequence of integers."""
    A = []
    val = start
    for _ in range(n):
        A.append(val)
        val = int(round(val * base_ratio))
        if val > 10000:
            break
    # If we didn't get 20, pad or adjust
    while len(A) < n:
        # Try to insert smaller numbers
        new_A = []
        for v in A:
            new_A.append(v)
        # This is a fallback; better to just return what we have if invalid
        # Actually, for geometric with ratio > 1, if it exceeds 10000, we need smaller start or ratio
        # Let's just return A if valid, else a default
        break
    if len(A) != n or any(x > 10000 for x in A) or len(set(A)) != n:
        return None
    return A

def _random_geometric():
    """Generate a random geometric-like set."""
    # Use logarithmic spacing
    # a_i = floor(c * r^i)
    # Choose r and c
    r = random.uniform(1.05, 1.8)
    c = random.uniform(1, 10)
    A = []
    for i in range(20):
        val = int(c * (r ** i))
        if val < 1:
            val = 1
        A.append(val)
    # Ensure distinctness and <= 10000
    # If not distinct, perturb
    # If > 10000, scale down
    max_val = max(A)
    if max_val > 10000:
        scale = 10000.0 / max_val
        A = [max(1, int(v * scale)) for v in A]
    
    # Fix distinctness
    # Sort and ensure strictly increasing
    A.sort()
    new_A = []
    last = 0
    for v in A:
        if v > last:
            new_A.append(v)
            last = v
        else:
            new_A.append(last + 1)
            last += 1
    # Check bounds
    if new_A[-1] > 10000:
        return None
    if len(new_A) != 20:
        return None
    return new_A

def _local_search(A, time_limit):
    """Perform local search (swap perturbations) to minimize score."""
    A = list(A)
    current_score = _score(A)
    start_time = time.time()
    best_A = A[:]
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Randomly pick two indices to swap or perturb
        i, j = random.sample(range(20), 2)
        # Perturb A[j]
        old_val = A[j]
        delta = random.randint(-10, 10)
        new_val = A[j] + delta
        if new_val < 1 or new_val > 10000:
            continue
        # Check distinctness
        if new_val in A[:j] + A[j+1:]:
            continue
        
        A[j] = new_val
        new_score = _score(A)
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A[:]
        else:
            A[j] = old_val
            
        # Occasionally try a swap
        if random.random() < 0.2:
            i, j = random.sample(range(20), 2)
            A[i], A[j] = A[j], A[i]
            new_score = _score(A)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
            else:
                A[i], A[j] = A[j], A[i]
                
    return best_A

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    # Try geometric starts
    for r in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
        if time.time() > deadline:
            break
        A = _geometric_set(r)
        if A:
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
                
    # Random geometric starts
    for _ in range(50):
        if time.time() > deadline:
            break
        A = _random_geometric()
        if A:
            s = _score(A)
            if s < best_score:
                best_score = s
                best_A = A[:]
                
    # Local search from best found
    remaining = deadline - time.time()
    if remaining > 1:
        best_A = _local_search(best_A, remaining)
        
    return best_A
# EVOLVE-BLOCK-END
