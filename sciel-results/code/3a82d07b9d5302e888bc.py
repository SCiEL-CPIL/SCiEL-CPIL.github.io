"""Rewritten program: Legendre-symbol initialization + incremental AP-discrepancy local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # safety margin before 90s kill


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _legendre_symbol(a, p):
    """Compute (a|p) for prime p and integer a."""
    if p == 2:
        return 1  # For p=2, Legendre symbol is defined as 1 for odd a
    if a % p == 0:
        return 0
    # Euler's criterion: (a|p) = a^((p-1)/2) mod p
    result = pow(a, (p - 1) // 2, p)
    return 1 if result == 1 else -1


def _legendre_init(n):
    """
    Generate a ±1 sequence using Legendre symbols of consecutive integers
    modulo a prime p >= n. This gives a low-discrepancy sequence with
    good AP properties.
    """
    # Find smallest prime p >= n
    p = n
    while not _is_prime(p):
        p += 1
    
    signs = np.empty(n, dtype=np.int8)
    for i in range(1, n + 1):
        ls = _legendre_symbol(i, p)
        signs[i - 1] = ls if ls != 0 else 1  # 0 shouldn't happen for i < p
    return signs


def _thue_morse_init(n):
    """Thue-Morse sequence mapped to ±1."""
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        t = 0
        x = i
        while x:
            t ^= 1
            x >>= 1
        signs[i] = 1 if t == 0 else -1
    return signs


def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [1, n]. signs is a 1D numpy array of int8 (-1 or +1), length n.
    Returns the max absolute sum (an integer).
    """
    n = len(signs)
    max_disc = 1  # singletons give |sum| = 1

    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            
            # Kadane-like: track prefix sum and running min/max of prefix
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            local_max = 0
            local_min = 0
            
            for val in seq:
                prefix += val
                if prefix - min_prefix > local_max:
                    local_max = prefix - min_prefix
                if prefix - max_prefix < local_min:
                    local_min = prefix - max_prefix
                
                if prefix < min_prefix:
                    min_prefix = prefix
                if prefix > max_prefix:
                    max_prefix = prefix
            
            disc = max(local_max, -local_min)
            if disc > max_disc:
                max_disc = disc
    
    return int(max_disc)


def _compute_max_discrepancy_with_early_exit(signs, threshold):
    """
    Same as _compute_max_discrepancy but exits early if max_disc exceeds threshold.
    Returns the max absolute sum found, or threshold+1 if exceeded.
    """
    n = len(signs)
    max_disc = 1

    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            local_max = 0
            local_min = 0
            
            for val in seq:
                prefix += val
                if prefix - min_prefix > local_max:
                    local_max = prefix - min_prefix
                if prefix - max_prefix < local_min:
                    local_min = prefix - max_prefix
                
                if prefix < min_prefix:
                    min_prefix = prefix
                if prefix > max_prefix:
                    max_prefix = prefix
                
                # Early exit check
                disc = max(local_max, -local_min)
                if disc > threshold:
                    return disc
            
            disc = max(local_max, -local_min)
            if disc > max_disc:
                max_disc = disc
                if max_disc > threshold:
                    return max_disc
    
    return max_disc


def _find_worst_ap(signs):
    """
    Find an arithmetic progression that achieves the maximum absolute sum.
    Returns (d, r, start_idx, end_idx) where the AP is signs[r::d][start_idx:end_idx+1]
    and the sum is maximized in absolute value.
    """
    n = len(signs)
    best_disc = 1
    best_d = 1
    best_r = 0
    best_start = 0
    best_end = 0
    best_sign = 1  # +1 for max sum, -1 for min sum

    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            
            prefix = 0
            min_prefix = 0
            max_prefix = 0
            min_prefix_idx = 0
            max_prefix_idx = 0
            local_max = 0
            local_min = 0
            best_end_idx = 0
            best_start_idx_max = 0
            best_start_idx_min = 0
            current_min_idx = 0
            current_max_idx = 0
            
            for i, val in enumerate(seq):
                prefix += val
                if prefix - min_prefix > local_max:
                    local_max = prefix - min_prefix
                    best_end_idx = i
                    best_start_idx_max = current_min_idx
                if prefix - max_prefix < local_min:
                    local_min = prefix - max_prefix
                    best_start_idx_min = current_max_idx
                    # best_end_idx stays as the one for min
                
                if prefix < min_prefix:
                    min_prefix = prefix
                    current_min_idx = i
                if prefix > max_prefix:
                    max_prefix = prefix
                    current_max_idx = i
            
            disc_pos = local_max
            disc_neg = -local_min
            if disc_pos >= disc_neg:
                disc = disc_pos
                s_idx = best_start_idx_max
                e_idx = best_end_idx
                sign_val = 1
            else:
                disc = disc_neg
                s_idx = best_start_idx_min
                e_idx = best_end_idx  # need to track this properly
                sign_val = -1
            
            if disc > best_disc:
                best_disc = disc
                best_d = d
                best_r = r
                best_start = s_idx
                best_end = e_idx
                best_sign = sign_val
    
    return best_d, best_r, best_start, best_end, best_sign


def _local_search(signs, max_iter=20000):
    """
    First-improvement hill climbing with random single-flip moves.
    Uses early-exit discrepancy computation for speed.
    """
    n = _N
    signs = signs.copy()
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(42)
    deadline = _DEADLINE - 3.0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random position to flip
        idx = rng.randrange(n)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        # Try to see if this improves; use early exit with threshold = current_disc
        new_disc = _compute_max_discrepancy_with_early_exit(signs, current_disc - 1)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[idx] = old_val
    
    return best_signs, best_disc


def _simulated_annealing(signs, max_iter=30000):
    """
    Simulated annealing with single-flip moves.
    """
    n = _N
    signs = signs.copy()
    current_disc = _compute_max_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    rng = random.Random(7)
    deadline = _DEADLINE - 3.0
    
    T_start = 5.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max(max_iter, 1))
    T = T_start
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        idx = rng.randrange(n)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        new_disc = _compute_max_discrepancy_with_early_exit(signs, current_disc + 2)
        
        delta = new_disc - current_disc
        if delta <= 0:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-delta / T)
            if T > 0.01 and rng.random() < np.exp(-delta / T):
                current_disc = new_disc
            else:
                signs[idx] = old_val
        
        T *= cooling
    
    return best_signs, best_disc


def _iterated_local_search(signs, num_restarts=5, max_iter_per_restart=15000):
    """
    ILS: local search from perturbed versions of the best solution.
    """
    best_signs = signs.copy()
    best_disc = _compute_max_discrepancy(signs)
    
    rng = random.Random(99)
    deadline = _DEADLINE - 5.0
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the best solution
        perturbed = best_signs.copy()
        num_flips = rng.randint(3, 10)
        for _ in range(num_flips):
            idx = rng.randrange(_N)
            perturbed[idx] = -perturbed[idx]
        
        # Local search from perturbed
        signs, disc = _local_search(perturbed, max_iter=max_iter_per_restart)
        
        if disc < best_disc:
            best_disc = disc
            best_signs = signs
    
    return best_signs, best_disc


def construct():
    start_time = time.time()
    
    # Multiple initializations
    inits = []
    inits.append(_legendre_init(_N))
    inits.append(_thue_morse_init(_N))
    
    # Random initializations
    rng = random.Random(2024)
    for _ in range(3):
        inits.append(np.array([1 if rng.random() < 0.5 else -1 for _ in range(_N)], dtype=np.int8))
    
    # Evaluate all initializations and pick the best
    best_signs = None
    best_disc = 1000000
    for init in inits:
        disc = _compute_max_discrepancy(init)
        if disc < best_disc:
            best_disc = disc
            best_signs = init.copy()
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 15.0:
        return [int(s) for s in best_signs]
    
    # Local search from best initialization
    signs, disc = _local_search(best_signs, max_iter=20000)
    if disc < best_disc:
        best_disc = disc
        best_signs = signs.copy()
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 15.0:
        return [int(s) for s in best_signs]
    
    # Simulated annealing from best
    signs, disc = _simulated_annealing(best_signs, max_iter=20000)
    if disc < best_disc:
        best_disc = disc
        best_signs = signs.copy()
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 10.0:
        return [int(s) for s in best_signs]
    
    # ILS from best
    signs, disc = _iterated_local_search(best_signs, num_restarts=5, max_iter_per_restart=10000)
    if disc < best_disc:
        best_disc = disc
        best_signs = signs.copy()
    
    return [int(s) for s in best_signs]
# EVOLVE-BLOCK-END
