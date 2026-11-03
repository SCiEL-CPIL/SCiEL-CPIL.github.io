"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _compute_score(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    max_u = max(U_set)
    
    # Compute sum set U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute difference set U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0 or max_u == 0:
        return 0.0
    
    denominator = math.log(2 * max_u + 1)
    if denominator == 0:
        return 0.0
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / denominator


def _quadratic_set(n):
    """Generate a quadratic sequence set."""
    return list(set([i * i for i in range(n)]))


def _triangular_set(n):
    """Generate a triangular number set."""
    return list(set([i * (i + 1) // 2 for i in range(n)]))


def _power_set(n, base=3):
    """Generate a power sequence set."""
    return list(set([base ** i for i in range(n)]))


def _local_search(U, max_iter=5000, time_limit=100):
    """Improve set U using local search."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    
    best_score = _compute_score(U)
    best_U = list(U)
    
    start_time = time.time()
    current_U = list(U)
    current_score = best_score
    
    max_u = max(current_U)
    n = len(current_U)
    
    # Define candidate moves
    def get_candidates(U, max_u, n):
        candidates = []
        U_set = set(U)
        
        # Try adding new elements
        for val in range(1, max_u + n * 5 + 10):
            if val not in U_set:
                candidates.append(('add', val))
        
        # Try removing elements (not 0)
        for val in U:
            if val != 0:
                candidates.append(('remove', val))
        
        # Try swapping
        for val in U:
            if val != 0:
                for new_val in range(1, max_u + n * 5 + 10):
                    if new_val not in U_set and new_val != val:
                        candidates.append(('swap', val, new_val))
        
        return candidates
    
    candidates = get_candidates(current_U, max_u, n)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        if not candidates:
            break
        
        # Randomly pick a candidate
        idx = random.randint(0, len(candidates) - 1)
        move = candidates[idx]
        
        if move[0] == 'add':
            new_U = list(set(current_U) | {move[1]})
        elif move[0] == 'remove':
            new_U = list(set(current_U) - {move[1]})
            if 0 not in new_U:
                continue
        elif move[0] == 'swap':
            new_U = list(set(current_U) - {move[1]} | {move[2]})
        
        if len(new_U) < 2:
            continue
        
        new_score = _compute_score(new_U)
        
        if new_score > current_score:
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = list(new_U)
            
            # Update candidates
            max_u = max(current_U)
            n = len(current_U)
            candidates = get_candidates(current_U, max_u, n)
    
    return best_U, best_score


def _algebraic_construction():
    """Try various algebraic constructions and return the best one."""
    constructions = []
    
    # Quadratic
    for n in range(10, 50):
        constructions.append(_quadratic_set(n))
    
    # Triangular
    for n in range(10, 50):
        constructions.append(_triangular_set(n))
    
    # Powers of 3
    for n in range(5, 15):
        constructions.append(_power_set(n, 3))
    
    # Powers of 2
    for n in range(5, 15):
        constructions.append(_power_set(n, 2))
    
    # Mixed: quadratic + small linear
    for n in range(5, 30):
        s = set([i * i for i in range(n)])
        s.add(0)
        constructions.append(list(s))
    
    # Find best construction
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    for U in constructions:
        if 0 not in U:
            U = list(set(U) | {0})
        score = _compute_score(U)
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    return best_U, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with algebraic constructions
    best_U, best_score = _algebraic_construction()
    
    # Check time
    if time.time() > deadline:
        return sorted(set(best_U))
    
    # Local search to improve
    improved_U, improved_score = _local_search(
        best_U, 
        max_iter=20000, 
        time_limit=100
    )
    
    if improved_score > best_score:
        best_U = improved_U
        best_score = improved_score
    
    # Final check
    if time.time() > deadline:
        return sorted(set(best_U))
    
    # Ensure 0 is in the set
    result = sorted(set(best_U) | {0})
    
    return result
# EVOLVE-BLOCK-END
