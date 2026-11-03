"""Construct a finite set U to maximize the sumset/difference-set exponent lower bound."""
import time
import random
import math
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 115  # Leave safety margin before 120s kill

def _elapsed():
    return time.time() - _START_TIME

def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    U_set = set(U)
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    if n_sum == 0:
        return 0.0
    log_ratio = math.log(n_diff / n_sum)
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def _random_base_expansion(B, num_digits, seed):
    """Generate a set using base-B expansion with binary digits."""
    rng = random.Random(seed)
    elements = [0]
    for j in range(num_digits):
        # Each digit position is either 0 or B^j
        if rng.random() < 0.5:
            elements.append(B ** j)
    # Generate all subset sums of the selected positions
    U = [0]
    for e in elements[1:]:
        new_U = [x + e for x in U]
        U = U + new_U
    return sorted(set(U))

def _greedy_construct(B, max_elements, seed):
    """Greedy construction: add elements that maximize the score increment."""
    rng = random.Random(seed)
    U = [0]
    # Start with some random base elements
    for _ in range(min(5, max_elements)):
        candidate = rng.randint(1, B ** 3)
        if candidate not in U:
            U.append(candidate)
    U = sorted(set(U))
    
    best_score = _score_set(U)
    
    # Try to improve by adding elements
    attempts = 0
    max_attempts = 5000
    while attempts < max_attempts and len(U) < max_elements and _elapsed() < TIME_LIMIT:
        # Generate a candidate
        if rng.random() < 0.3:
            # Try a structured element
            j = rng.randint(1, 6)
            candidate = B ** j
        else:
            candidate = rng.randint(1, B ** 4)
        
        if candidate in U:
            attempts += 1
            continue
        
        new_U = U + [candidate]
        new_score = _score_set(new_U)
        
        if new_score > best_score:
            U = new_U
            best_score = new_score
            attempts = 0
        else:
            attempts += 1
        
        if attempts > 200:
            # Reset attempts and try a different strategy
            attempts = 0
            # Try removing a random element and adding a new one
            if len(U) > 2:
                idx = rng.randint(1, len(U) - 1)
                U = U[:idx] + U[idx+1:]
                U = sorted(set(U))
                best_score = _score_set(U)
    
    return sorted(set(U)), best_score

def _local_search(U, B, max_steps, seed):
    """Local search to improve the set."""
    rng = random.Random(seed)
    U = list(U)
    best_score = _score_set(U)
    
    for step in range(max_steps):
        if _elapsed() > TIME_LIMIT:
            break
        
        # Try a random swap or addition
        op = rng.random()
        if op < 0.4 and len(U) < 30:
            # Add a new element
            candidate = rng.randint(1, B ** 4)
            if candidate not in U:
                new_U = U + [candidate]
                new_score = _score_set(new_U)
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
        elif op < 0.7 and len(U) > 2:
            # Replace an element
            idx = rng.randint(1, len(U) - 1)
            old = U[idx]
            new_val = rng.randint(1, B ** 4)
            if new_val not in U:
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_score = _score_set(new_U)
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
        else:
            # Try a small perturbation
            idx = rng.randint(0, len(U) - 1)
            delta = rng.choice([-1, 1, -2, 2, -B, B])
            new_val = U[idx] + delta
            if new_val >= 0 and new_val not in U:
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_score = _score_set(new_U)
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
    
    return sorted(set(U)), best_score

def _base_expansion_optimal(B, num_digits):
    """Construct set using optimal base expansion with all digits active."""
    # U = all subset sums of {B^0, B^1, ..., B^(k-1)}
    # This gives |U| = 2^k, max(U) = sum of B^j
    # |U+U| is bounded, |U-U| is larger
    U = [0]
    for j in range(num_digits):
        new_U = [x + B ** j for x in U]
        U = U + new_U
    return sorted(set(U))

def construct():
    """Main entry point: construct a set U maximizing the score."""
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Strategy 1: Base expansion with various bases
    for B in [2, 3, 4, 5, 7, 10, 13]:
        if _elapsed() > TIME_LIMIT - 10:
            break
        for num_digits in range(2, 8):
            if _elapsed() > TIME_LIMIT - 10:
                break
            U = _base_expansion_optimal(B, num_digits)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Strategy 2: Greedy construction
    for B in [3, 5, 7, 10]:
        if _elapsed() > TIME_LIMIT - 20:
            break
        for seed in range(10):
            if _elapsed() > TIME_LIMIT - 20:
                break
            U, score = _greedy_construct(B, 20, seed)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Strategy 3: Local search from best found
    if _elapsed() < TIME_LIMIT - 15:
        for seed in range(5):
            if _elapsed() > TIME_LIMIT - 15:
                break
            U, score = _local_search(best_U, 5, 200, seed)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Strategy 4: Try larger base expansions
    for B in [2, 3, 4]:
        if _elapsed() > TIME_LIMIT - 10:
            break
        for num_digits in range(2, 10):
            if _elapsed() > TIME_LIMIT - 10:
                break
            U = _base_expansion_optimal(B, num_digits)
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U = [0] + best_U
    best_U = sorted(set(best_U))
    
    return best_U
