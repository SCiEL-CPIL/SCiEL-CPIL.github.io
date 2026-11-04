import time
import random
import numpy as np
from math import log

def score_set(U):
    """
    Computes the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
    """
    if not U:
        return -1
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """
    Vectorized scoring using NumPy for speed.
    """
    if not U:
        return -1
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U)
    # Compute all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums)
    sum_count = len(unique_sums)
    
    # Compute all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs)
    diff_count = len(unique_diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U_arr.max()
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Initial best
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # 1. Try some structured candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        # Add more structured candidates based on lattice packing ideas
        [0, 1, 5, 10, 16, 23, 31, 40],  # Quadratic
        [0, 1, 6, 12, 19, 27, 36, 46],  # Quadratic
        [0, 1, 3, 6, 10, 15, 21, 28],   # Triangular
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Improved local search with gap-based heuristics
    
    def gap_heuristic_score(U):
        """
        Auxiliary score: reward for large gaps between consecutive elements
        that don't create new sums, penalize small gaps that create many sums.
        """
        if len(U) < 2:
            return -1
        U = sorted(set(U))
        n = len(U)
        
        # Compute sumset and diffset sizes
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(U[i] + U[j])
            for j in range(i + 1, n):
                d = U[j] - U[i]
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        
        sum_count = len(sums)
        diff_count = len(diffs)
        
        if sum_count == 0:
            return -1
            
        ratio = diff_count / sum_count
        max_u = max(U)
        D = 2 * max_u + 1
        if D <= 1:
            return -1
            
        base_score = 1 + log(ratio) / log(D)
        
        # Gap-based bonus: reward for elements that are far from existing sums
        gap_bonus = 0
        for i in range(1, n):
            gap = U[i] - U[i-1]
            # Check if this gap creates new sums
            new_sums = 0
            for j in range(n):
                if j != i and U[i] + U[j] not in sums - {U[i] + U[j]}:
                    pass  # Simplified
            gap_bonus += min(gap, 50) / 50.0
        
        return base_score + 0.01 * gap_bonus / max(n, 1)

    def local_search(U, deadline, max_val_limit=800, max_n_limit=40):
        """
        Perform local search with multiple move types and adaptive step sizes.
        """
        U = sorted(set(U))
        current_score = score_set_fast(U)
        best_local_score = current_score
        best_local_set = list(U)
        
        # Temperature for simulated annealing
        T = 0.5
        cooling_rate = 0.998
        min_T = 0.005
        
        no_improvement = 0
        max_no_improvement = 150
        step_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        while time.time() < deadline - 3 and T > min_T:
            if no_improvement > max_no_improvement:
                break
                
            # Generate a perturbation
            move_type = random.random()
            new_U = list(U)
            
            if move_type < 0.25:
                # Add an element - use sum/diff gap heuristic
                if len(U) < max_n_limit:
                    candidates = set()
                    # Candidates from sums and differences
                    for a in U:
                        for b in U:
                            s = a + b
                            d = abs(a - b)
                            if 0 < s <= max_val_limit and s not in U:
                                candidates.add(s)
                            if 0 < d <= max_val_limit and d not in U:
                                candidates.add(d)
                    # Add random candidates
                    for _ in range(30):
                        r = random.randint(1, max_val_limit)
                        if r not in U:
                            candidates.add(r)
                    if candidates:
                        # Score candidates by gap heuristic (approximate)
                        cand_list = list(candidates)
                        # Pick from top 50% by random sampling for diversity
                        if len(cand_list) > 10:
                            random.shuffle(cand_list)
                            new_U.append(random.choice(cand_list[:max(10, len(cand_list)//2)]))
                        else:
                            new_U.append(random.choice(cand_list))
                        new_U.sort()
            elif move_type < 0.40:
                # Remove an element
                if len(U) > 3:
                    removable = [x for x in U if x != 0]
                    if removable:
                        # Prefer removing elements that create many sums
                        removable.sort(key=lambda x: -x)  # Larger elements first
                        idx = random.choice(removable[:max(1, len(removable)//2)])
                        new_U = [x for x in U if x != idx]
            elif move_type < 0.85:
                # Replace an element with adaptive step size
                if len(U) > 1:
                    idx = random.randint(1, len(U) - 1)
                    old_val = U[idx]
                    # Choose step size based on local density
                    step = random.choice(step_sizes)
                    direction = random.choice([-1, 1])
                    new_val = old_val + direction * step
                    if new_val < 1 or new_val in U:
                        # Try other directions
                        for alt_step in step_sizes:
                            for alt_dir in [-1, 1]:
                                alt_val = old_val + alt_dir * alt_step
                                if alt_val >= 1 and alt_val not in U:
                                    new_val = alt_val
                                    break
                            if new_val >= 1 and new_val not in U:
                                break
                    if new_val >= 1 and new_val not in U:
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U.sort()
                    else:
                        new_U = U.copy()
            else:
                # Swap two elements
                if len(U) > 2:
                    i, j = random.sample(range(1, len(U)), 2)
                    new_U = U.copy()
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    new_U.sort()
            
            if new_U == U:
                continue
                
            new_score = score_set_fast(new_U)
            if new_score is None or new_score < -1:
                continue
                
            # Simulated annealing acceptance
            if new_score > current_score:
                U = new_U
                current_score = new_score
                no_improvement = 0
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_set = list(U)
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if delta > -T * 5:  # Avoid underflow
                    prob = min(1.0, math.exp(-delta / T) if delta < 0 else 1.0)
                    if random.random() < prob:
                        U = new_U
                        current_score = new_score
                no_improvement += 1
            
            T *= cooling_rate
        
        return best_local_score, best_local_set

    # Restart loop with improved local search
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        
        # Choose starting set
        if restarts % 5 == 0:
            base = random.choice(structured_candidates)
            U = list(base)
        elif restarts % 5 == 1:
            # Quadratic progression
            n = random.randint(5, 18)
            a = random.randint(1, 4)
            b = random.randint(0, 6)
            U = [0] + [a*i*i + b*i for i in range(1, n) if a*i*i + b*i <= 800]
            U = sorted(set(U))
            if len(U) < 3:
                U = [0, 1, 4]
        elif restarts % 5 == 2:
            # Geometric progression
            r = random.uniform(1.4, 2.8)
            U = [0]
            val = 1
            while val < 800 and len(U) < 25:
                U.append(int(val))
                val *= r
            U = sorted(set(U))
        elif restarts % 5 == 3:
            # Sparse random with gap preference
            n = random.randint(5, 20)
            # Generate with increasing gaps
            U = [0]
            current = 0
            for _ in range(n - 1):
                gap = random.randint(2, 30)
                current += gap
                if current > 800:
                    break
                U.append(current)
            U = sorted(set(U))
        else:
            # Random
            n = random.randint(4, 18)
            elements = random.sample(range(1, 800), n - 1)
            U = [0] + elements
            U.sort()
        
        local_score, local_set = local_search(U, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Final polish with wider step sizes and more iterations
    if time.time() < deadline - 5:
        U = list(best_set)
        U.sort()
        current_score = best_score
        
        polish_steps = [1, 2, 3, 5, 8, 13, 21, 34, -1, -2, -3, -5, -8, -13, -21, -34]
        
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            found = False
            random.shuffle(polish_steps)
            for step in polish_steps:
                new_val = old_val + step
                if new_val < 1 or new_val in U:
                    continue
                new_U = U.copy()
                new_U[idx] = new_val
                new_U.sort()
                new_score = score_set_fast(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    found = True
                    break
            if found:
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
        
        # Try adding/removing elements in final polish
        if time.time() < deadline - 3:
            for _ in range(100):
                if time.time() > deadline - 3:
                    break
                if random.random() < 0.5 and len(U) < 35:
                    # Try adding
                    candidates = set()
                    for a in U:
                        for b in U:
                            s = a + b
                            d = abs(a - b)
                            if 0 < s <= 1000 and s not in U:
                                candidates.add(s)
                            if 0 < d <= 1000 and d not in U:
                                candidates.add(d)
                    if candidates:
                        new_val = random.choice(list(candidates))
                        new_U = U + [new_val]
                        new_U.sort()
                        new_score = score_set_fast(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
                elif len(U) > 4:
                    # Try removing
                    removable = [x for x in U if x != 0]
                    if removable:
                        idx = random.choice(removable)
                        new_U = [x for x in U if x != idx]
                        new_score = score_set_fast(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)

    return best_set

# Helper for simulated annealing
import math
def exp(x):
    return math.exp(x)
