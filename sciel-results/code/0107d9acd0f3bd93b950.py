"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    diffset = set()
    U_list = list(U_set)
    n = len(U_list)
    
    for i in range(n):
        for j in range(i, n):
            sumset.add(U_list[i] + U_list[j])
        for j in range(n):
            diffset.add(U_list[i] - U_list[j])
    
    # |U-U| counts all differences, including negatives and zero
    # |U+U| counts all sums
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / denom
    return score


def _local_search(U, max_steps=500, time_limit=None, start_time=None):
    """Perform local search to improve the set."""
    if start_time is None:
        start_time = time.time()
    
    U = list(set(U))
    U.sort()
    if 0 not in U:
        U.append(0)
        U.sort()
    
    best_score = _compute_score(U)
    best_U = U[:]
    no_improvement = 0
    max_no_improvement = 30
    
    while no_improvement < max_no_improvement:
        if time_limit and time.time() - start_time > time_limit - 5:
            break
        
        improved = False
        # Try adding a random element
        max_u = max(best_U)
        
        # Try several random moves
        for _ in range(20):
            if time_limit and time.time() - start_time > time_limit - 5:
                break
            
            trial_U = best_U[:]
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a new element
                # Choose a candidate near existing structure
                if random.random() < 0.5:
                    base = random.choice(best_U)
                    delta = random.randint(1, max(1, max_u // 10 + 1))
                    candidate = base + delta
                else:
                    candidate = random.randint(1, max_u + max_u // 5 + 1)
                
                if candidate not in trial_U and candidate >= 0:
                    trial_U.append(candidate)
                    trial_U.sort()
                    new_score = _compute_score(trial_U)
                    if new_score > best_score:
                        best_score = new_score
                        best_U = trial_U
                        improved = True
                        break
            elif move_type < 0.7:
                # Remove an element (not 0)
                if len(best_U) > 1:
                    candidates = [x for x in best_U if x != 0]
                    if candidates:
                        idx = random.choice(candidates)
                        trial_U.remove(idx)
                        new_score = _compute_score(trial_U)
                        if new_score > best_score:
                            best_score = new_score
                            best_U = trial_U
                            improved = True
                            break
            else:
                # Swap: remove one, add another
                if len(best_U) > 1:
                    candidates = [x for x in best_U if x != 0]
                    if candidates:
                        remove_val = random.choice(candidates)
                        trial_U.remove(remove_val)
                        if random.random() < 0.5:
                            base = random.choice(trial_U) if trial_U else 0
                            delta = random.randint(1, max(1, max_u // 10 + 1))
                            candidate = base + delta
                        else:
                            candidate = random.randint(1, max_u + max_u // 5 + 1)
                        if candidate not in trial_U and candidate >= 0:
                            trial_U.append(candidate)
                            trial_U.sort()
                            new_score = _compute_score(trial_U)
                            if new_score > best_score:
                                best_score = new_score
                                best_U = trial_U
                                improved = True
                                break
        
        if not improved:
            no_improvement += 1
        else:
            no_improvement = 0
    
    # Final polish: try small perturbations
    if time_limit and time.time() - start_time < time_limit - 3:
        max_u = max(best_U)
        for _ in range(100):
            if time_limit and time.time() - start_time > time_limit - 3:
                break
            trial_U = best_U[:]
            # Try adding elements in a range around existing ones
            added = False
            for _ in range(5):
                base = random.choice(best_U)
                delta = random.randint(-40, 40)
                candidate = base + delta
                if candidate >= 0 and candidate not in trial_U:
                    trial_U.append(candidate)
                    trial_U.sort()
                    added = True
                    break
            
            if added:
                new_score = _compute_score(trial_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = trial_U
    
    return best_U, best_score


def _construct_greedy(max_size=2000, time_limit=None, start_time=None):
    """Construct a set greedily to maximize the score."""
    if start_time is None:
        start_time = time.time()
    
    U = [0]
    
    for _ in range(max_size):
        if time_limit and time.time() - start_time > time_limit - 10:
            break
        
        best_candidate = None
        best_score = _compute_score(U)
        
        # Try candidates in a structured way
        max_u = max(U)
        # Generate candidates
        candidates = set()
        for u in U:
            for d in range(1, max(2, max_u // 20 + 1)):
                candidates.add(u + d)
        # Add some random candidates
        for _ in range(20):
            candidates.add(random.randint(1, max_u + max_u // 10 + 1))
        
        best_local_score = best_score
        for cand in candidates:
            if time_limit and time.time() - start_time > time_limit - 10:
                break
            if cand in U or cand < 0:
                continue
            trial_U = U + [cand]
            trial_U.sort()
            s = _compute_score(trial_U)
            if s > best_local_score:
                best_local_score = s
                best_candidate = cand
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
        else:
            break
    
    return U


def _random_initial_set(size=50, max_val=1000):
    """Generate a random initial set containing 0."""
    U = {0}
    for _ in range(size - 1):
        U.add(random.randint(1, max_val))
    return list(U)


def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    random.seed(42)
    
    best_U = None
    best_score = 0.0
    
    # Strategy 1: Greedy construction
    U_greedy = _construct_greedy(max_size=500, time_limit=time_limit, start_time=start_time)
    score_greedy = _compute_score(U_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        best_U = U_greedy[:]
    
    # Strategy 2: Local search from greedy result
    if time.time() - start_time < time_limit - 20:
        U_ls, score_ls = _local_search(U_greedy, max_steps=500, time_limit=time_limit, start_time=start_time)
        if score_ls > best_score:
            best_score = score_ls
            best_U = U_ls[:]
    
    # Strategy 3: Multiple random restarts with local search
    restarts = 5
    for r in range(restarts):
        if time.time() - start_time > time_limit - 20:
            break
        
        U_rand = _random_initial_set(size=30 + r * 10, max_val=500 + r * 500)
        U_ls, score_ls = _local_search(U_rand, max_steps=300, time_limit=time_limit, start_time=start_time)
        if score_ls > best_score:
            best_score = score_ls
            best_U = U_ls[:]
    
    # Final polish on the best solution
    if best_U and time.time() - start_time < time_limit - 5:
        U_final, score_final = _local_search(best_U, max_steps=200, time_limit=time_limit, start_time=start_time)
        if score_final > best_score:
            best_score = score_final
            best_U = U_final[:]
    
    if best_U is None:
        best_U = [0, 1, 3]
    
    # Ensure 0 is in the set and all elements are nonnegative
    best_U = sorted(set(x for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

# EVOLVE-BLOCK-END
