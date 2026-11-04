import time
import random
from math import log, log2
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    sums = np.unique(U_arr[:, None] + U_arr[None, :])
    diffs = np.unique(U_arr[:, None] - U_arr[None, :])
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def construct_exponential_set(alpha, n_elements, base_gap=1):
    """
    Construct a set with exponentially growing gaps.
    U[0] = 0, U[k] = U[k-1] + floor(base_gap * alpha^k)
    """
    if n_elements <= 0:
        return [0]
    U = [0]
    current = 0
    for k in range(1, n_elements):
        gap = int(base_gap * (alpha ** k))
        if gap < 1:
            gap = 1
        current += gap
        U.append(current)
    return U

def score_exponential(alpha, n_elements, base_gap=1):
    """Score an exponentially growing set."""
    U = construct_exponential_set(alpha, n_elements, base_gap)
    if len(U) < 2:
        return -1
    return score_set(U)

def ternary_search_alpha(n_elements, base_gap=1, deadline=None):
    """
    Use ternary search to find optimal alpha for given n_elements.
    We search alpha in range [1.5, 3.0] which covers good geometric ratios.
    """
    lo = 1.5
    hi = 3.0
    max_iter = 50
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        
        s1 = score_exponential(m1, n_elements, base_gap)
        s2 = score_exponential(m2, n_elements, base_gap)
        
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    mid = (lo + hi) / 2
    return mid, score_exponential(mid, n_elements, base_gap)

def local_search(U, deadline, max_iter=3000):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    temperature = 0.03
    cooling_rate = 0.995
    
    iteration = 0
    while time.time() < deadline and iteration < max_iter:
        iteration += 1
        n = len(current)
        
        move_type = random.randint(1, 3)
        new_set = current
        
        if move_type == 1:
            i = random.randint(1, n - 1)
            old_val = current[i]
            # Perturb value with adaptive range
            if random.random() < 0.4:
                delta = random.randint(-5, 5)
            elif random.random() < 0.7:
                delta = random.randint(-int(old_val * 0.2), int(old_val * 0.2))
            else:
                delta = random.randint(-int(old_val * 0.5), int(old_val * 0.5))
            
            new_val = old_val + delta
            if new_val <= 0:
                continue
            new_set = [current[j] if j != i else new_val for j in range(n)]
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue
                
        elif move_type == 2:
            # Add element
            if n >= 15:
                continue
            max_val = max(current)
            if random.random() < 0.5:
                new_val = random.randint(max_val + 1, max_val + max(10, max_val // 3))
            else:
                new_val = random.randint(1, max_val + max(10, max_val // 3))
            
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove element
            if n <= 3:
                continue
            i = random.randint(1, n - 1)
            new_set = [current[j] for j in range(n) if j != i]

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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Phase 1: Ternary search over exponential families with different n_elements
    for n_elements in range(3, 16):
        if time.time() > deadline - 30:
            break
        alpha, s = ternary_search_alpha(n_elements, deadline=deadline - 25)
        if s > best_score:
            best_score = s
            best_set = construct_exponential_set(alpha, n_elements)
    
    # Phase 2: Try some base_gap variations for best n_elements
    if time.time() < deadline - 20:
        # Find best n_elements from Phase 1
        best_n = len(best_set)
        for base_gap in [1, 2, 3, 5]:
            if time.time() > deadline - 20:
                break
            alpha, s = ternary_search_alpha(best_n, base_gap, deadline=deadline - 15)
            if s > best_score:
                best_score = s
                best_set = construct_exponential_set(alpha, best_n, base_gap)
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 10:
        refined, refined_score = local_search(best_set, deadline - 5)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
    
    # Phase 4: Try additional candidate families
    candidates = []
    # Pure geometric progressions
    for b in [2, 3, 4, 5]:
        for m in range(3, 12):
            U = [0]
            val = 1
            for k in range(1, m + 1):
                val *= b
                U.append(val)
            candidates.append(U)
    
    # Exponential with small perturbations
    for alpha in [1.8, 2.0, 2.2, 2.5]:
        for n in range(4, 10):
            candidates.append(construct_exponential_set(alpha, n))
    
    for U in candidates:
        if time.time() > deadline - 8:
            break
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U
            # Quick local search from new best
            if time.time() < deadline - 5:
                refined, refined_score = local_search(best_set, deadline - 3)
                if refined_score > best_score:
                    best_score = refined_score
                    best_set = refined

    return best_set
