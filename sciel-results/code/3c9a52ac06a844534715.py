"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score for a set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _geometric_set(base, n):
    """Generate a geometric set {0, 1, base, base^2, ..., base^(n-1)}."""
    U = [0, 1]
    val = 1
    for i in range(1, n):
        val *= base
        U.append(val)
    return sorted(U)


def _mixed_set(base, n, offset=0):
    """Generate a mixed geometric-arithmetic set."""
    U = [0]
    for i in range(1, n):
        U.append(offset + i * base)
    return sorted(U)


def _prime_gap_set(n, start=2):
    """Generate a set with prime-step gaps."""
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    U = [0]
    current = 0
    p = start
    for i in range(1, n):
        while not is_prime(p):
            p += 1
        current += p
        U.append(current)
        p += 1
    return sorted(U)


def _local_search_add(U, max_iterations=5000, time_limit=10.0):
    """Local search trying to add new elements."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        margin = max(10, max_u // 5)
        new_val = random.randint(0, max_u + margin)
        if new_val in U:
            continue
        
        new_U = sorted(U + [new_val])
        new_score = _compute_score(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
            max_u = U[-1]
    
    return best_U, best_score


def _local_search_swap(U, max_iterations=5000, time_limit=10.0):
    """Local search with swap operations."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        if n > 1:
            i = random.randint(1, n - 1)  # Don't swap 0
            old_val = U[i]
            margin = max(10, max_u // 5)
            new_val = random.randint(0, max_u + margin)
            if new_val != old_val and new_val not in U:
                new_U = list(U)
                new_U[i] = new_val
                new_U = sorted(new_U)
                new_score = _compute_score(new_U)
                
                if new_score > best_score:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    max_u = U[-1]
    
    return best_U, best_score


def _local_search_remove(U, max_iterations=3000, time_limit=8.0):
    """Local search trying to remove elements."""
    U = sorted(set(U))
    n = len(U)
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if len(U) <= 2:
            break
        
        i = random.randint(1, len(U) - 1)  # Don't remove 0
        new_U = U[:i] + U[i+1:]
        new_score = _compute_score(new_U)
        
        if new_score > best_score:
            best_score = new_score
            best_U = new_U
            U = new_U
    
    return best_U, best_score


def _polish(U, time_limit=15.0):
    """Final high-precision polish: alternate add, swap, remove."""
    U = sorted(set(U))
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Add
        U, s = _local_search_add(best_U, max_iterations=2000, time_limit=3.0)
        if s > best_score:
            best_score = s
            best_U = U
        
        # Swap
        U, s = _local_search_swap(best_U, max_iterations=2000, time_limit=3.0)
        if s > best_score:
            best_score = s
            best_U = U
        
        # Remove
        U, s = _local_search_remove(best_U, max_iterations=1000, time_limit=2.0)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # Try geometric sets with various bases
    for base in range(2, 30):
        for n in range(3, 12):
            if time.time() > deadline - 40:
                break
            U = _geometric_set(base, n)
            if max(U) > 100000:
                continue
            s = _compute_score(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
    
    # Try prime-gap sets
    for n in range(4, 15):
        if time.time() > deadline - 35:
            break
        U = _prime_gap_set(n)
        if max(U) > 100000:
            continue
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Try random sets
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        n = random.randint(3, 15)
        max_val = random.randint(20, 500)
        U = [0] + sorted(random.sample(range(1, max_val + 1), n - 1))
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Local search from the best seed
    if time.time() < deadline - 25:
        U, s = _local_search_add(best_U, max_iterations=10000, time_limit=20)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Swap-based local search
    if time.time() < deadline - 20:
        U, s = _local_search_swap(best_U, max_iterations=10000, time_limit=15)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Remove-based local search
    if time.time() < deadline - 15:
        U, s = _local_search_remove(best_U, max_iterations=5000, time_limit=10)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Final polish
    if time.time() < deadline - 5:
        U, s = _polish(best_U, time_limit=min(15, deadline - time.time() - 2))
        if s > best_score:
            best_score = s
            best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
