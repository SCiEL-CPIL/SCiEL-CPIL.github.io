"""Optimized program: Combined exhaustive search, local search, and simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import log, exp

def _compute_score(U):
    """Compute the score for a given set U."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0 or 0 not in U:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U
    sumset = np.unique(U[:, None] + U[None, :])
    diffset = np.unique(U[:, None] - U[None, :])
    len_sum = len(sumset)
    len_diff = len(diffset)
    
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _local_search(U, max_iterations=5000, timeout=10.0):
    """Local search: add/remove/swap elements to improve score."""
    start_time = time.time()
    current = sorted(set(U))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    best = list(current)
    best_score = current_score
    
    for i in range(max_iterations):
        if time.time() - start_time > timeout:
            break
        
        U_set = set(current)
        max_u = max(U_set) if U_set else 0
        n = len(U_set)
        
        # Generate candidates
        candidates = set()
        # Perturbations
        for x in list(U_set):
            for off in [-3, -2, -1, 1, 2, 3, 5, 8, 13]:
                c = x + off
                if 0 <= c <= 200: # Slightly larger search space
                    candidates.add(c)
        # Sums/Diffs
        elems = list(U_set)
        for a in elems:
            for b in elems:
                s = a + b
                if s <= 200:
                    candidates.add(s)
                d = abs(a - b)
                if 0 < d <= 200:
                    candidates.add(d)
        # Random
        for _ in range(15):
            candidates.add(random.randint(0, 200))
        # Boundary
        if max_u > 0:
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u * 2)
            
        candidate_list = list(candidates)
        random.shuffle(candidate_list)
        
        improved = False
        for cand in candidate_list:
            if time.time() - start_time > timeout:
                break
            if cand in U_set:
                continue
            new_set = U_set | {cand}
            new_score = _compute_score(new_set)
            if new_score > current_score + 1e-10:
                current = sorted(new_set)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                improved = True
                break
        
        if not improved:
            # Try removals
            removable = [x for x in current if x != 0]
            random.shuffle(removable)
            for cand in removable[:15]:
                if time.time() - start_time > timeout:
                    break
                new_set = U_set - {cand}
                if len(new_set) < 2:
                    continue
                new_score = _compute_score(new_set)
                if new_score > current_score + 1e-10:
                    current = sorted(new_set)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = list(current)
                    improved = True
                    break
        
        if not improved and i > 10:
            # Swap
            if random.random() < 0.5:
                removable = [x for x in current if x != 0]
                if removable:
                    rem = random.choice(removable)
                    new_elem = random.randint(0, 200)
                    new_set = (U_set - {rem}) | {new_elem}
                    if len(new_set) >= 2:
                        new_score = _compute_score(new_set)
                        if new_score > current_score + 1e-10:
                            current = sorted(new_set)
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = list(current)
    return best, best_score

def _sa_search(initial_set, max_val, timeout=10.0):
    """Simulated Annealing to find a good set."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.05
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
            
        # Generate neighbor
        move_type = random.random()
        new_set = list(current)
        
        if move_type < 0.5:
            # Change an element
            if len(new_set) <= 1:
                continue
            idx = random.randint(0, len(new_set) - 1)
            if idx == 0 and new_set[0] == 0:
                # Don't change 0
                idx = random.randint(1, len(new_set) - 1) if len(new_set) > 1 else 0
            old_val = new_set[idx]
            # Try nearby values
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 0:
                new_val = random.randint(0, max_val)
            if new_val > max_val:
                new_val = random.randint(0, max_val)
            if new_val in new_set and new_val != old_val:
                continue
            new_set[idx] = new_val
        elif move_type < 0.75:
            # Add an element
            new_elem = random.randint(0, max_val)
            if new_elem not in new_set:
                new_set.append(new_elem)
            else:
                continue
        else:
            # Remove an element (not 0)
            removable = [x for x in new_set if x != 0]
            if not removable:
                continue
            rem = random.choice(removable)
            new_set.remove(rem)
            
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
            
        new_score = _compute_score(new_set)
        
        # Acceptance
        if new_score >= current_score or random.random() < exp((new_score - current_score) / T):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
                
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin for 150s limit

    best_result = [0, 1, 2]
    best_score = _compute_score(best_result)
    
    # 1. Exhaustive search for small sets (n=3 to 6) within a bounded range
    # This is fast and likely to find the global optimum for small n
    max_val_exhaustive = 35
    for n in range(3, 7):
        if time.time() > deadline - 20:
            break
        # Use combinations for exact search
        for combo in combinations(range(1, max_val_exhaustive + 1), n - 1):
            U = [0] + list(combo)
            score = _compute_score(U)
            if score > best_score:
                best_score = score
                best_result = U
    
    # 2. Local search from the best found
    if time.time() < deadline - 10:
        improved, score = _local_search(best_result, max_iterations=20000, timeout=min(20.0, deadline - time.time() - 2))
        if score > best_score:
            best_score = score
            best_result = improved
            
    # 3. Simulated Annealing from diverse starting points
    # This helps escape local optima found by local search
    if time.time() < deadline - 10:
        # Start from best result
        result, score = _sa_search(best_result, max_val=200, timeout=min(15.0, deadline - time.time() - 2))
        if score > best_score:
            best_score = score
            best_result = result
            
        # Start from random sets
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            # Generate a random small set
            n = random.randint(3, 8)
            rand_set = {0}
            while len(rand_set) < n:
                rand_set.add(random.randint(1, 100))
            rand_list = sorted(rand_set)
            result, score = _sa_search(rand_list, max_val=150, timeout=min(5.0, deadline - time.time() - 1))
            if score > best_score:
                best_score = score
                best_result = result

    # 4. Final refinement on the best result
    if time.time() < deadline - 2:
        improved, score = _local_search(best_result, max_iterations=10000, timeout=min(5.0, deadline - time.time() - 1))
        if score > best_score:
            best_score = score
            best_result = improved

    # Final check
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 2, 4]
        
    return best_result
# EVOLVE-BLOCK-END
