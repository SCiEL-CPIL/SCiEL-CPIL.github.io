import itertools
import math
import time
import random

def score_set(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    U = sorted(set(U))
    if U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for i in range(len(U)):
        for j in range(len(U)):
            sums.add(U[i] + U[j])
            diffs.add(U[i] - U[j])
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def geometric_set(num_intervals, base, alpha):
    """
    Construct a set U as a union of intervals with exponentially growing gaps.
    
    Parameters:
    - num_intervals: number of intervals
    - base: starting position (should be 0)
    - alpha: gap growth ratio (alpha > 1)
    
    The intervals are:
    - Interval i: [a_i, a_i + L_i - 1]
    - a_0 = 0, L_0 = 1
    - a_{i+1} = a_i + L_i + gap_i
    - gap_i = round(alpha * (a_i + L_i))
    - L_{i+1} = L_i + 1 (or some growth rule)
    """
    if num_intervals <= 0:
        return [0]
    
    U = []
    current_start = 0
    current_length = 1
    
    for i in range(num_intervals):
        # Add interval [current_start, current_start + current_length - 1]
        for x in range(current_start, current_start + current_length):
            U.append(x)
        
        if i < num_intervals - 1:
            # Calculate gap to next interval
            gap = max(1, round(alpha * (current_start + current_length)))
            current_start += current_length + gap
            current_length += 1  # Grow interval length
    
    return sorted(set(U))

def generate_candidates(num_intervals, alpha_min, alpha_max, alpha_steps):
    """Generate candidate sets with different alpha values."""
    candidates = []
    for i in range(alpha_steps):
        alpha = alpha_min + (alpha_max - alpha_min) * i / max(1, alpha_steps - 1)
        if alpha <= 1:
            alpha = 1.001
        U = geometric_set(num_intervals, 0, alpha)
        if U and 0 in U:
            candidates.append(U)
    return candidates

def local_search_dynamic(U, max_iterations=1000, time_limit=10.0):
    """
    Local search that dynamically adjusts the perturbation strength.
    """
    start_time = time.time()
    current_score = score_set(U)
    best_score = current_score
    best_set = U[:]
    
    # Try different types of modifications
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly choose a modification type
        mod_type = iteration % 3
        
        if mod_type == 0:
            # Add a random element
            max_u = max(U) if U else 0
            new_elem = random.randint(1, max_u * 2 + 10)
            if new_elem not in U:
                U_new = U + [new_elem]
                U_new = sorted(set(U_new))
                s = score_set(U_new)
                if s > best_score:
                    best_score = s
                    best_set = U_new
                    U = U_new
                    current_score = s
                elif s > current_score:
                    U = U_new
                    current_score = s
        elif mod_type == 1:
            # Remove a random element (keep 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U_new = U[:idx] + U[idx+1:]
                s = score_set(U_new)
                if s > best_score:
                    best_score = s
                    best_set = U_new
                    U = U_new
                    current_score = s
                elif s > current_score:
                    U = U_new
                    current_score = s
        else:
            # Perturb a random element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                delta = random.randint(-5, 5)
                new_val = max(0, U[idx] + delta)
                if new_val not in U:
                    U_new = U[:]
                    U_new[idx] = new_val
                    U_new = sorted(set(U_new))
                    s = score_set(U_new)
                    if s > best_score:
                        best_score = s
                        best_set = U_new
                        U = U_new
                        current_score = s
                    elif s > current_score:
                        U = U_new
                        current_score = s
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Strategy 1: Try geometric sets with various parameters
    for num_intervals in range(2, 12):
        for alpha in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            U = geometric_set(num_intervals, 0, alpha)
            if U and 0 in U:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
            if time.time() > deadline:
                return best_set
    
    # Strategy 2: Local search from best geometric set
    if time.time() < deadline - 20:
        U_opt, s_opt = local_search_dynamic(best_set, max_iterations=500, time_limit=15.0)
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt
    
    # Strategy 3: Try small exhaustive search for very small sets
    if time.time() < deadline - 30:
        # Size 4
        for x in range(1, 15):
            for y in range(x+1, 15):
                for z in range(y+1, 15):
                    U = [0, x, y, z]
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                    if time.time() > deadline:
                        return best_set
                    
        # Size 5
        for x in range(1, 12):
            for y in range(x+1, 12):
                for z in range(y+1, 12):
                    for w in range(z+1, 20):
                        U = [0, x, y, z, w]
                        s = score_set(U)
                        if s > best_score:
                            best_score = s
                            best_set = U
                        if time.time() > deadline:
                            return best_set
    
    # Final local search polish
    if time.time() < deadline - 5:
        U_opt, s_opt = local_search_dynamic(best_set, max_iterations=1000, time_limit=5.0)
        if s_opt > best_score:
            best_score = s_opt
            best_set = U_opt
    
    return best_set
