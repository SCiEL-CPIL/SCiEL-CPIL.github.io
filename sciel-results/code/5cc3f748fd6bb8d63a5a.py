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


def _compute_score_fast(U):
    """Faster score computation using set operations for smaller sets."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _base_b_set(digits, B):
    """Construct set from binary digit vector and base B."""
    U = set()
    for mask in range(1 << len(digits)):
        val = 0
        for i, d in enumerate(digits):
            if mask & (1 << i):
                val += d
        U.add(val)
    return sorted(U)


def _base_b_score(digits, B):
    """Compute score for a base-B positional set."""
    U = _base_b_set(digits, B)
    if not U or 0 not in U:
        return 0.0
    return _compute_score_fast(U)


def _optimize_base_b(digits, B, max_iters=5000, time_limit=10.0):
    """Local search on base B and digit positions."""
    start_time = time.time()
    best_score = _base_b_score(digits, B)
    best_digits = list(digits)
    best_B = B
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try changing the base
        new_B = B + random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
        if new_B > 1:
            s = _base_b_score(digits, new_B)
            if s > best_score:
                best_score = s
                best_B = new_B
                B = new_B
        
        # Try modifying a digit position
        if len(digits) > 1:
            i = random.randint(1, len(digits) - 1)  # Keep digits[0] = 1
            old_d = digits[i]
            # Try nearby values
            new_d = old_d + random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10])
            if new_d > 0:
                new_digits = list(digits)
                new_digits[i] = new_d
                s = _base_b_score(new_digits, B)
                if s > best_score:
                    best_score = s
                    best_digits = new_digits
                    digits = new_digits
        
        # Try adding a new digit position
        if len(digits) < 15:
            new_d = random.randint(B, B * 3)
            new_digits = list(digits) + [new_d]
            s = _base_b_score(new_digits, B)
            if s > best_score:
                best_score = s
                best_digits = new_digits
                digits = new_digits
        
        # Try removing a digit position
        if len(digits) > 2:
            i = random.randint(1, len(digits) - 1)
            new_digits = list(digits[:i] + digits[i+1:])
            s = _base_b_score(new_digits, B)
            if s > best_score:
                best_score = s
                best_digits = new_digits
                digits = new_digits
    
    return best_digits, best_B, best_score


def _local_search(U, max_iterations=5000, time_limit=10.0):
    """Local search to improve the set U."""
    U = sorted(set(U))
    n = len(U)
    max_u = U[-1]
    best_score = _compute_score(U)
    best_U = list(U)
    start_time = time.time()
    
    # Try adding new elements
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a new element
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
        
        # Try swapping one element
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


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second limit for safety
    
    best_U = [0, 1, 3]
    best_score = _compute_score(best_U)
    
    # === Base-B positional representation approach ===
    # Try various base-B constructions
    for B in [3, 4, 5, 7, 10, 13, 17, 23, 31, 41, 53, 67, 83]:
        if time.time() > deadline - 60:
            break
        
        # Start with simple digit vectors
        for n_digits in range(2, 8):
            if time.time() > deadline - 55:
                break
            
            # Superincreasing-like digits
            digits = [1]
            for i in range(1, n_digits):
                digits.append(digits[-1] * B + 1)
            
            # Optimize this configuration
            opt_digits, opt_B, opt_score = _optimize_base_b(
                digits, B, max_iters=2000, time_limit=5.0
            )
            
            if opt_score > best_score:
                best_score = opt_score
                U = _base_b_set(opt_digits, opt_B)
                if 0 in U:
                    best_U = U
    
    # Try random base-B configurations
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 50:
            break
        
        B = random.choice([3, 4, 5, 7, 10, 13, 17, 23, 31])
        n_digits = random.randint(2, 6)
        
        # Random digits
        digits = [1]
        for i in range(1, n_digits):
            digits.append(random.randint(B, B * 4))
        
        opt_digits, opt_B, opt_score = _optimize_base_b(
            digits, B, max_iters=3000, time_limit=5.0
        )
        
        if opt_score > best_score:
            best_score = opt_score
            U = _base_b_set(opt_digits, opt_B)
            if 0 in U:
                best_U = U
    
    # === Traditional approaches ===
    # Try some random sets
    for trial in range(15):
        if time.time() > deadline - 40:
            break
        n = random.randint(3, 20)
        max_val = random.randint(10, 200)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        score = _compute_score(U)
        
        if score > best_score:
            best_score = score
            best_U = list(U)
    
    # Local search from the best seed
    if time.time() < deadline - 20:
        U, score = _local_search(best_U, max_iterations=15000, time_limit=25)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try swap-based local search
    if time.time() < deadline - 15:
        U, score = _local_search_swap(best_U, max_iterations=15000, time_limit=20)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try a few more local search rounds with different random seeds
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb the best solution
            U = list(best_U)
            # Remove a random element (not 0)
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            # Add a random element
            max_u = max(U) if U else 0
            new_val = random.randint(0, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            
            U, score = _local_search(U, max_iterations=3000, time_limit=8)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)

# EVOLVE-BLOCK-END
