"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using numpy for speed
    U_list = sorted(U_set)
    U_arr = np.array(U_list, dtype=np.int64)
    
    # Use numpy broadcasting for sumset and diffset
    sumset = set(np.unique(U_arr[:, None] + U_arr[None, :]).tolist())
    diffset = set(np.unique(U_arr[:, None] - U_arr[None, :]).tolist())
    
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _fast_score(U):
    """Faster scoring for local search - assumes U is a sorted list with no duplicates."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # For small sets, use numpy
    U_arr = np.array(U, dtype=np.int64)
    sumset = set(np.unique(U_arr[:, None] + U_arr[None, :]).tolist())
    diffset = set(np.unique(U_arr[:, None] - U_arr[None, :]).tolist())
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1.0:
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1) if ratio > 0 else 0.0
    
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _greedy_prime_construction(max_size=30, deadline=None):
    """
    Build a set by greedily adding elements that maximize incremental score.
    Start with {0, 1} and keep adding the best candidate.
    """
    U = [0, 1]
    start_time = time.time()
    
    for step in range(max_size - 2):
        if deadline and time.time() > deadline:
            break
            
        best_score = _fast_score(U)
        best_candidate = None
        best_new_score = best_score
        
        # Try candidates in a range around current max
        max_u = max(U)
        search_start = max(1, max_u - 5)
        search_end = max_u + 30
        
        for candidate in range(search_start, search_end + 1):
            if candidate in U:
                continue
            new_U = U + [candidate]
            new_U.sort()
            s = _fast_score(new_U)
            if s > best_new_score + 1e-10:
                best_new_score = s
                best_candidate = candidate
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
        else:
            # Try wider range if local search fails
            found = False
            for candidate in range(1, max_u + 100):
                if candidate in U:
                    continue
                new_U = U + [candidate]
                new_U.sort()
                s = _fast_score(new_U)
                if s > best_new_score + 1e-10:
                    best_new_score = s
                    best_candidate = candidate
                    found = True
                    break
            if found:
                U.append(best_candidate)
                U.sort()
            else:
                break
                
    return U

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search with add/remove/move operations."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 50
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element
        for i in range(len(best_U)):
            if improved:
                break
            old_val = best_U[i]
            if old_val == 0:
                continue
            # Try nearby values first
            search_range = list(range(max(1, old_val - 15), min(upper_bound, old_val + 15) + 1))
            random.shuffle(search_range)
            
            for new_val in search_range:
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try adding a new element
        existing = set(best_U)
        candidates_add = [v for v in range(1, upper_bound + 1) if v not in existing]
        random.shuffle(candidates_add)
        
        for new_val in candidates_add:
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _fast_score(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing an element (if size > 3)
        if len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _fast_score(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _simulated_annealing(U, deadline, start_temp=10.0, end_temp=0.01, max_iters=3000):
    """
    Simulated annealing to escape local minima.
    Allows worse moves with probability exp(-delta/T).
    """
    U = sorted(set(U))
    best_U = U[:]
    best_score = _fast_score(U)
    current_U = U[:]
    current_score = best_score
    
    start_time = time.time()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Temperature schedule
        progress = it / max_iters
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Generate a random move
        move_type = random.random()
        new_U = current_U[:]
        
        if move_type < 0.5:
            # Move an element
            i = random.randint(1, len(new_U) - 1)
            old_val = new_U[i]
            delta = random.randint(-20, 20)
            new_val = old_val + delta
            if new_val > 0 and new_val not in new_U:
                new_U[i] = new_val
                new_U.sort()
                new_U = list(set(new_U))
            else:
                continue
        elif move_type < 0.8:
            # Add an element
            max_u = max(new_U)
            new_val = random.randint(1, max_u + 40)
            if new_val not in new_U:
                new_U.append(new_val)
                new_U.sort()
            else:
                continue
        else:
            # Remove an element
            if len(new_U) > 3:
                i = random.randint(1, len(new_U) - 1)
                new_U = new_U[:i] + new_U[i+1:]
            else:
                continue
        
        if len(new_U) != len(current_U) and move_type < 0.5:
            continue  # Skip if size changed unexpectedly in move
            
        new_score = _fast_score(new_U)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(-abs(delta) / max(temp, 1e-10)):
            current_U = new_U[:]
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = new_U[:]
                
    return sorted(set(best_U)), best_score

def _search_small_sets(max_n=6, max_val=35, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            s = _fast_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = _fast_score(best_U)
    
    # 1. Exhaustive search for small sets (fast, high quality baseline)
    try:
        U_sol, s_sol = _search_small_sets(max_n=6, max_val=35, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Greedy prime construction - builds a good initial set
    if time.time() < deadline:
        greedy_U = _greedy_prime_construction(max_size=25, deadline=deadline)
        s_greedy = _fast_score(greedy_U)
        if s_greedy > best_score:
            best_score = s_greedy
            best_U = greedy_U
    else:
        greedy_U = best_U[:]
        
    # 3. Local search on the best candidate
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=1500, deadline=deadline)
        
    # 4. Simulated annealing to escape local minima
    if time.time() < deadline - 10:
        sa_U, sa_score = _simulated_annealing(best_U, deadline, max_iters=2000)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U
    
    # 5. Final local search polish
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=500, deadline=deadline)
        
    # 6. Multiple restarts with perturbation if time remains
    if time.time() < deadline - 10:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = best_U[:]
            num_perturb = min(3, max(1, len(perturbed) // 3))
            for _ in range(num_perturb):
                if len(perturbed) > 3:
                    idx = random.randint(1, len(perturbed) - 1)
                    old_val = perturbed[idx]
                    delta = random.randint(-15, 15)
                    new_val = old_val + delta
                    if new_val > 0 and new_val not in perturbed:
                        perturbed[idx] = new_val
                        perturbed.sort()
                        perturbed = list(set(perturbed))
                        if 0 in perturbed:
                            s = _fast_score(perturbed)
                            if s > best_score:
                                best_score = s
                                best_U = perturbed[:]
            
            if time.time() < deadline - 10:
                U_local, s_local = _local_search(perturbed, max_iters=300, deadline=deadline)
                if s_local > best_score:
                    best_score = s_local
                    best_U = U_local
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
