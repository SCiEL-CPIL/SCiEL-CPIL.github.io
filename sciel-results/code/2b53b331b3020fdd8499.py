"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _score_set(U):
    """Compute the exact score for a set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Compute |U-U| and |U+U| exactly
    diff_set = set()
    sum_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
            sum_set.add(a + b)
    
    diff_size = len(diff_set)
    sum_size = len(sum_set)
    
    if sum_size == 0:
        return -1
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return -1
    
    return 1.0 + log_ratio / log_denom


def _quadratic_residues(p):
    """Generate the set of quadratic residues modulo p (as integers 0..p-1)."""
    residues = set()
    for i in range(p):
        residues.add((i * i) % p)
    # Ensure 0 is in the set (it always is: 0^2 = 0)
    residues.add(0)
    return sorted(residues)


def _local_search(U, max_iters=5000, time_limit=10.0):
    """Local search to improve the score by adding/removing elements."""
    start_time = time.time()
    U = sorted(set(U))
    current_score = _score_set(U)
    best_U = U[:]
    best_score = current_score
    
    # Try to add elements that improve the score
    # Focus on small elements to keep max(U) low
    max_u = max(U) if U else 0
    
    # Candidate elements to try adding
    candidates = set()
    # Try elements up to 2*max_u + 10
    upper_bound = min(2 * max_u + 50, 200)
    for i in range(upper_bound + 1):
        candidates.add(i)
    
    # Also try some random larger elements
    for _ in range(50):
        candidates.add(random.randint(0, upper_bound + 100))
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try adding an element
        for elem in sorted(candidates):
            if elem in U:
                continue
            new_U = U + [elem]
            new_score = _score_set(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                current_score = new_score
                improved = True
                # Update candidates
                max_u = max(U)
                upper_bound = min(2 * max_u + 50, 300)
                candidates = set()
                for i in range(upper_bound + 1):
                    candidates.add(i)
                for _ in range(30):
                    candidates.add(random.randint(0, upper_bound + 100))
                break
        
        # Try removing an element (except 0)
        if not improved:
            for elem in U:
                if elem == 0:
                    continue
                new_U = [x for x in U if x != elem]
                if not new_U:
                    continue
                new_score = _score_set(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try random swaps
            if len(U) >= 2:
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    i = random.randint(0, len(U) - 1)
                    if U[i] == 0:
                        continue
                    j = random.randint(0, len(U) - 1)
                    new_elem = random.randint(0, max_u + 20)
                    if new_elem == U[i] or new_elem in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_elem
                    new_score = _score_set(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
    
    return best_U, best_score


def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try quadratic residues for various primes
    primes_to_try = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for p in primes_to_try:
        if time.time() > deadline - 30:
            break
        
        U = _quadratic_residues(p)
        score = _score_set(U)
        
        if score > best_score:
            best_score = score
            best_U = U[:]
        
        # Try local search from this starting point
        if time.time() > deadline - 20:
            break
        
        improved_U, improved_score = _local_search(U, max_iters=2000, time_limit=15.0)
        
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U[:]
    
    # Also try local search from the seed
    if time.time() > deadline - 15:
        return best_U
    
    improved_U, improved_score = _local_search([0, 1, 3], max_iters=5000, time_limit=10.0)
    if improved_score > best_score:
        best_score = improved_score
        best_U = improved_U[:]
    
    # Final: try some small hand-crafted sets
    small_sets = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 1, 3, 4, 9, 10, 12, 13, 16, 17, 19, 20, 24, 25, 27, 28, 32, 33, 35, 36],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 10, 15, 21, 26, 32, 37, 43, 48, 54, 59, 65, 70, 76, 81, 87, 92, 98],
    ]
    
    for U in small_sets:
        if time.time() > deadline - 10:
            break
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_U = U[:]
    
    # Final local search from the best
    if time.time() > deadline - 8:
        return best_U
    
    improved_U, improved_score = _local_search(best_U, max_iters=3000, time_limit=8.0)
    if improved_score > best_score:
        best_U = improved_U
    
    return best_U

# EVOLVE-BLOCK-END
