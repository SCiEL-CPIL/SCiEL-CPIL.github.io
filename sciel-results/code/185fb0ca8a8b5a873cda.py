import time
import random
from math import log
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    # Use NumPy for faster sumset and difference set calculation
    U_arr = np.array(U)
    
    # Calculate all pairwise sums
    sums = np.unique(U_arr[:, None] + U_arr[None, :])
    
    # Calculate all pairwise differences
    diffs = np.unique(U_arr[:, None] - U_arr[None, :])
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Construct U = {floor(b^k) for k=0,...,m}"""
    U = [0]
    for k in range(1, m + 1):
        val = int(b ** k)
        if val <= 0:
            val = 1
        if val not in U:
            U.append(val)
    return sorted(U)

def score_geometric(b, m):
    """Score a geometric progression set."""
    U = geometric_set(b, m)
    if len(U) < 2:
        return -1
    return score_set(U)

def local_search(U, deadline, max_iter=2000):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    temperature = 0.05
    cooling_rate = 0.99
    
    iteration = 0
    while time.time() < deadline and iteration < max_iter:
        iteration += 1
        n = len(current)
        
        move_type = random.randint(1, 4)
        new_set = current
        
        if move_type == 1:
            i = random.randint(1, n - 1)
            old_val = current[i]
            if random.random() < 0.7:
                delta = random.randint(-10, 10)
            else:
                delta = random.randint(-int(old_val*0.5), int(old_val*0.5))
            
            new_val = old_val + delta
            if new_val <= 0:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
                
        elif move_type == 2:
            if n >= 25:
                continue
            max_val = max(current)
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val + 50)
            else:
                i = random.randint(1, n - 1)
                if current[i] > current[i-1] + 2:
                    new_val = random.randint(current[i-1] + 1, current[i] - 1)
                else:
                    new_val = random.randint(1, max_val + 50)
            
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        elif move_type == 3:
            if n <= 5:
                continue
            i = random.randint(1, n - 1)
            new_set = [current[j] for j in range(n) if j != i]
            
        else:
            i = random.randint(1, n - 1)
            old_val = current[i]
            new_val = random.randint(1, max(10, int(max(current) * 1.5)))
            if new_val == old_val:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue

        new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = current_score - new_score
            if random.random() < np.exp(-delta / max(temperature, 1e-9)):
                current = new_set
                current_score = new_score
        
        temperature *= cooling_rate

    return current, current_score

def ternary_search_geometric(b_lo, b_hi, m, deadline, tol=1e-4, max_iter=100):
    """Ternary search over base b for fixed m to maximize score."""
    best_b = None
    best_score = -1
    
    a, c = b_lo, b_hi
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        m1 = a + (c - a) / 3
        m2 = c - (c - a) / 3
        
        s1 = score_geometric(m1, m)
        s2 = score_geometric(m2, m)
        
        if s1 > s2:
            c = m2
        else:
            a = m1
        
        if abs(c - a) < tol:
            break
        
        # Track best
        for b in [a, (a+c)/2, c]:
            s = score_geometric(b, m)
            if s > best_score:
                best_score = s
                best_b = b
    
    # Final evaluation
    s = score_geometric((a+c)/2, m)
    if s > best_score:
        best_score = s
        best_b = (a+c)/2
    
    if best_b is None:
        return None, -1
    
    U = geometric_set(best_b, m)
    return U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Phase 1: Ternary search over geometric progressions
    # For each m (number of elements), find optimal base b
    for m in range(2, 25):
        if time.time() > deadline - 30:
            break
        
        # Search over base b in [1.5, 10]
        U, s = ternary_search_geometric(1.5, 10.0, m, deadline - 30, tol=1e-3, max_iter=60)
        if U is not None and s > best_score:
            best_score = s
            best_set = U
    
    # Phase 2: Generate diverse base candidates for local search
    candidates = []
    
    # Geometric progressions with various bases
    for b in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        for m in range(3, 15):
            U = geometric_set(b, m)
            if len(U) >= 2:
                candidates.append(U)
    
    # 1. Exponential growth (2^k - 1)
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
        
    # 2. Powers of 2
    for k in range(1, 18):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val
        candidates.append(U)
        
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 50000:
        U.append(b)
        a, b = b, a + b
    candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(20)]
    candidates.append(U)
    
    # 5. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
        random.seed(seed)
        for _ in range(10):
            U = [0]
            val = random.randint(1, 10)
            for _ in range(random.randint(5, 20)):
                U.append(val)
                val = random.randint(int(val * 1.1), int(val * 4.0))
            candidates.append(U)
    
    # 9. Sparse random sets
    random.seed(999)
    for _ in range(20):
        n_elem = random.randint(8, 25)
        max_val = random.randint(200, 5000)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
    
    # Evaluate and refine
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = score_set(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    # Process top candidates
    num_candidates_to_process = min(len(scored_candidates), 20)
    remaining_time = deadline - time.time() - 5
    if remaining_time > 0 and num_candidates_to_process > 0:
        time_per_candidate = remaining_time / num_candidates_to_process
        
        for i, (score, U) in enumerate(scored_candidates[:num_candidates_to_process]):
            if time.time() > deadline - 5:
                break
            
            candidate_deadline = min(deadline - 5, time.time() + time_per_candidate * 1.5)
            
            refined, refined_score = local_search(U, candidate_deadline)
            if refined_score > best_score:
                best_score = refined_score
                best_set = refined
    
    # Final refinement on best set
    if time.time() < deadline - 3:
        refined, refined_score = local_search(best_set, deadline - 1)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    return best_set
