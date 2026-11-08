"""Construct a finite set U of nonnegative integers containing 0 to maximize the sumset/difference-set exponent lower bound."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U:
        return 0.0
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    n_sum = len(sumset)
    n_diff = len(diffset)
    
    if n_sum == 0 or max_u <= 0:
        return 0.0
    
    log_arg = n_diff / n_sum
    if log_arg <= 0:
        return 0.0
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + math.log(log_arg) / denom


def _try_exhaustive_small(n):
    """Try to find good sets of size n by random sampling."""
    best_score = 0.0
    best_set = None
    
    # For small n, try random sets with values up to some bound
    max_val = 50
    n_trials = 5000
    
    for _ in range(n_trials):
        # Generate random set containing 0
        if n == 1:
            U = [0]
        else:
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
        
        score = _compute_score(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
    
    return best_set, best_score


def _local_search(U, time_limit):
    """Local search to improve a set U."""
    start_time = time.time()
    current_score = _compute_score(U)
    current_set = sorted(U)
    
    # Try adding, removing, or modifying elements
    max_val = max(current_set) + 100 if current_set else 100
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try adding a new element
        for _ in range(50):
            new_val = random.randint(1, max_val)
            if new_val in current_set:
                continue
            new_set = sorted(current_set + [new_val])
            new_score = _compute_score(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                max_val = max(current_set) + 50
                improved = True
                break
        
        # Try removing an element (except 0)
        if len(current_set) > 2:
            for _ in range(20):
                idx = random.randint(1, len(current_set) - 1)
                new_set = current_set[:idx] + current_set[idx+1:]
                new_score = _compute_score(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try modifying an element
        for _ in range(30):
            idx = random.randint(1, len(current_set) - 1)
            old_val = current_set[idx]
            new_val = max(1, old_val + random.randint(-20, 20))
            if new_val in current_set:
                continue
            new_set = current_set[:idx] + [new_val] + current_set[idx+1:]
            new_set = sorted(new_set)
            new_score = _compute_score(new_set)
            if new_score > current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                break
        
        if not improved:
            # Stuck, try larger perturbations
            for _ in range(10):
                new_set = current_set[:]
                # Add a few random elements
                for _ in range(random.randint(1, 3)):
                    new_val = random.randint(1, max_val * 2)
                    if new_val not in new_set:
                        new_set.append(new_val)
                # Remove a few random elements (not 0)
                for _ in range(random.randint(0, 2)):
                    if len(new_set) > 2:
                        idx = random.randint(1, len(new_set) - 1)
                        new_set.pop(idx)
                new_set = sorted(new_set)
                new_score = _compute_score(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    max_val = max(current_set) + 50
                    improved = True
                    break
        
        if not improved and time.time() - start_time > time_limit * 0.5:
            break
    
    return current_set, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Try exhaustive search for small sets to get good seeds
    for n in range(2, 8):
        if time.time() > deadline - 30:
            break
        seed_set, seed_score = _try_exhaustive_small(n)
        if seed_set and seed_score > best_score:
            best_set = seed_set
            best_score = seed_score
    
    # Local search from best seed
    if time.time() < deadline - 10:
        improved_set, improved_score = _local_search(best_set, deadline - time.time() - 10)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try multiple random restarts with local search
    while time.time() < deadline - 5:
        # Generate a random starting set
        n = random.randint(3, 15)
        max_val = random.randint(20, 200)
        if n == 1:
            U = [0]
        else:
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
        
        # Quick local search with time limit
        remaining = deadline - time.time() - 2
        if remaining < 2:
            break
        search_time = min(remaining / 2, 5)  # Cap individual search time
        improved_set, improved_score = _local_search(U, search_time)
        
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Ensure 0 is in the set and all are nonnegative integers
    best_set = sorted(set(x for x in best_set if x >= 0))
    if 0 not in best_set:
        best_set.insert(0, 0)
    
    return best_set
# EVOLVE-BLOCK-END
