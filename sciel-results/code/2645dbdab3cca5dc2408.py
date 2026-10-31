import math
import time
import random

def calculate_score(U):
    """Calculate the score for a given set U."""
    if not U:
        return 0.0
    U_set = set(U)
    if 0 not in U_set:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    U_list = sorted(U_set)
    n = len(U_list)
    
    # Calculate sumset and diffset
    sumset = set()
    diffset = set()
    
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_list[i] + U_list[j])
        for j in range(n):
            diffset.add(U_list[i] - U_list[j])
    
    size_sum = len(sumset)
    size_diff = len(diffset)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1 + math.log(ratio) / denom
    return score

def is_sidon(U):
    """Check if U is a Sidon set (distinct sums for i<=j)."""
    U_list = sorted(U)
    n = len(U_list)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = U_list[i] + U_list[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_minimal_sidon(n):
    """
    Generate a Sidon set of size n with the smallest possible maximum element
    using a greedy approach.
    """
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    U = [0]
    current_max = 0
    
    for _ in range(n - 1):
        candidate = current_max + 1
        while True:
            new_U = U + [candidate]
            if is_sidon(new_U):
                U = new_U
                current_max = candidate
                break
            candidate += 1
            
    return U

def local_search_optimize(U, deadline, max_iterations=200):
    """
    Perform local search to optimize the set U.
    Try adding, replacing, and removing elements.
    """
    current = sorted(set(U))
    best_score = calculate_score(current)
    best_set = list(current)
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_iterations:
        improved = False
        current_max = max(current)
        
        # Try adding an element
        if len(current) < 50:
            for new_el in range(1, current_max + 10):
                if new_el in current:
                    continue
                cand = sorted(current + [new_el])
                s = calculate_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    break
        
        if improved:
            no_improve = 0
            continue
        
        # Try replacing an element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                for new_el in range(1, current_max + 10):
                    if new_el in current:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = calculate_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            no_improve = 0
            continue
        
        # Try removing an element (if size > 3)
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                s = calculate_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                    current = list(cand)
                    improved = True
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    return best_set, best_score

def generate_structured_candidates(max_u):
    """Generate various structured candidate sets."""
    candidates = []
    
    # Geometric with ratio 2
    geo2 = [0, 1]
    while geo2[-1] * 2 <= max_u:
        geo2.append(geo2[-1] * 2)
    if max(geo2) <= max_u:
        candidates.append(geo2)
    
    # Geometric with ratio 3
    geo3 = [0, 1]
    while geo3[-1] * 3 <= max_u:
        geo3.append(geo3[-1] * 3)
    if max(geo3) <= max_u:
        candidates.append(geo3)
    
    # Fibonacci-like
    fib = [0, 1]
    while fib[-1] + fib[-2] <= max_u:
        fib.append(fib[-1] + fib[-2])
    for n in range(3, len(fib)):
        if fib[n-1] <= max_u:
            candidates.append(fib[:n])
    
    # Small arithmetic progressions
    for n in range(3, 10):
        step = max_u // (n - 1) if n > 1 else 1
        if step >= 1:
            ap = [i * step for i in range(n)]
            if max(ap) <= max_u:
                candidates.append(ap)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -1
    best_set = [0, 1, 3]  # Seed
    
    # Evaluate seed
    seed_score = calculate_score(best_set)
    best_score = seed_score
    
    # Phase 1: Generate and evaluate Sidon sets of increasing size
    # Go up to size 80, but check time
    for n in range(3, 81):
        if time.time() > deadline - 20:
            break
            
        try:
            U = generate_minimal_sidon(n)
        except Exception:
            continue
            
        score = calculate_score(U)
        
        if score > best_score:
            best_score = score
            best_set = U
    
    # Phase 2: Local search optimization on the best set found
    if time.time() < deadline - 15:
        optimized_set, optimized_score = local_search_optimize(
            best_set, deadline - 10, max_iterations=300
        )
        if optimized_score > best_score:
            best_score = optimized_score
            best_set = optimized_set
    
    # Phase 3: Try structured candidates and optimize them
    if time.time() < deadline - 20:
        for max_u in [10, 20, 50, 100, 200, 500]:
            if time.time() > deadline - 10:
                break
            
            candidates = generate_structured_candidates(max_u)
            for cand in candidates:
                if time.time() > deadline - 10:
                    break
                score = calculate_score(cand)
                if score > best_score:
                    best_score = score
                    best_set = cand
            
            # Also try random sets
            random.seed(42 + max_u)
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                n = random.randint(3, min(15, max_u))
                if n <= 1:
                    continue
                inner = list(range(1, max_u))
                if len(inner) < n - 1:
                    continue
                selected = random.sample(inner, n - 1)
                cand = sorted([0] + selected)
                score = calculate_score(cand)
                if score > best_score:
                    best_score = score
                    best_set = cand
    
    # Phase 4: Final local search on the overall best
    if time.time() < deadline - 5:
        optimized_set, optimized_score = local_search_optimize(
            best_set, deadline - 2, max_iterations=500
        )
        if optimized_score > best_score:
            best_score = optimized_score
            best_set = optimized_set
    
    return best_set
