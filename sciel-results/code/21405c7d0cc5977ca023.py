"""Optimized search for integer sum-product extremal sets."""
import random
import time

def _objective(A):
    """Compute max(|A+A|, |A*A|) for set A of 20 distinct positive integers."""
    n = len(A)
    # Sumset: all a_i + a_j for i <= j (distinct sums)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
    # Product set: all a_i * a_j for i <= j
    prods = set()
    for i in range(n):
        for j in range(i, n):
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _is_valid(A):
    """Check A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if x < 1 or x > 10000:
            return False
    return True

def _generate_structured_candidates():
    """Generate diverse initial candidates using different structures."""
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(2, 22)))
    candidates.append(list(range(5, 25)))
    
    # 2. Powers of 2 (minimizes product set, maximizes sum set)
    powers2 = [2**i for i in range(20)]
    if max(powers2) <= 10000:
        candidates.append(powers2)
    else:
        # Truncated powers
        p = []
        v = 1
        while len(p) < 20 and v <= 10000:
            p.append(v)
            v *= 2
        while len(p) < 20:
            p.append(p[-1] + 1)
        candidates.append(p)
    
    # 3. Geometric progression with small ratio
    for ratio in [1.1, 1.2, 1.3, 1.5]:
        gp = []
        v = 1.0
        while len(gp) < 20 and v <= 10000:
            gp.append(int(round(v)))
            v *= ratio
        # Remove duplicates and ensure 20 distinct
        gp = list(dict.fromkeys(gp))
        while len(gp) < 20:
            gp.append(max(gp) + 1)
        if _is_valid(gp):
            candidates.append(gp[:20])
    
    # 4. Mixed: small numbers plus some large
    base = list(range(1, 11))
    large = [500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]
    mixed = base + large
    if _is_valid(mixed):
        candidates.append(mixed)
    
    # 5. Random samples
    for _ in range(10):
        r = random.sample(range(1, 10001), 20)
        candidates.append(sorted(r))
    
    # 6. Arithmetic progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5, 10]:
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000 and _is_valid(ap):
                candidates.append(ap)
    
    # 7. Fibonacci-like
    fib = [1, 1]
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    # Remove duplicate 1s
    fib = list(dict.fromkeys(fib))
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    if _is_valid(fib):
        candidates.append(fib[:20])
    
    return candidates

def _local_search(A, deadline, best_val, best_A):
    """Perform local search with random replacements and swaps."""
    A = list(A)
    cur_val = _objective(A)
    if cur_val < best_val:
        best_val = cur_val
        best_A = list(A)
    
    n = 20
    max_iter = 50000
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace one element
            idx = random.randint(0, n - 1)
            old = A[idx]
            # Try values near old or random
            if random.random() < 0.5:
                new = old + random.randint(-50, 50)
            else:
                new = random.randint(1, 10000)
            new = max(1, min(10000, new))
            if new == old or new in A:
                continue
            A[idx] = new
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                # Accept with probability based on improvement (simulated annealing)
                delta = new_val - cur_val
                if delta <= 0 or random.random() < 0.1:
                    no_improve += 1
                else:
                    A[idx] = old
                    no_improve += 1
        elif move_type < 0.7:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            A[i], A[j] = A[j], A[i]
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                A[i], A[j] = A[j], A[i]
                no_improve += 1
        else:
            # Perturb multiple elements
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, n - 1)
                new = A[idx] + random.randint(-100, 100)
                new = max(1, min(10000, new))
                if new not in A or new == A[idx]:
                    A[idx] = new
            
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_val, best_A

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = _generate_structured_candidates()
    
    # Evaluate all candidates
    best_val = float('inf')
    best_A = None
    for c in candidates:
        if not _is_valid(c):
            continue
        v = _objective(c)
        if v < best_val:
            best_val = v
            best_A = list(c)
    
    # Local search from best candidate
    if time.time() < deadline - 5.0:
        best_val, best_A = _local_search(best_A, deadline - 5.0, best_val, best_A)
    
    # Multiple restarts from different starting points
    restarts = 0
    while time.time() < deadline - 2.0 and restarts < 10:
        # Generate a new random starting point
        new_start = sorted(random.sample(range(1, 10001), 20))
        v = _objective(new_start)
        if v < best_val:
            best_val = v
            best_A = list(new_start)
        
        # Local search from this point
        best_val, best_A = _local_search(new_start, deadline - 2.0, best_val, best_A)
        restarts += 1
    
    # Final polish: try small perturbations on best_A
    if time.time() < deadline - 1.0:
        A = list(best_A)
        cur_val = best_val
        for _ in range(10000):
            if time.time() > deadline - 1.0:
                break
            idx = random.randint(0, 19)
            old = A[idx]
            new = old + random.randint(-10, 10)
            new = max(1, min(10000, new))
            if new == old or new in A:
                continue
            A[idx] = new
            v = _objective(A)
            if v < cur_val:
                cur_val = v
                best_val = v
                best_A = list(A)
            else:
                A[idx] = old
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
