import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _elapsed():
    return time.time() - _START

def _compute_score(U):
    """Compute score for set U using O(n²) exact computation."""
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = int(U[-1])
    if max_u <= 0:
        return 0.0
    
    # Compute all pairwise sums and differences
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    
    base = 2 * max_u + 1
    if base <= 1:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(base)

def _primes_up_to(n):
    """Generate primes up to n using sieve."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

# Known optimal Golomb rulers (including 0)
_KNOWN_GOLOMB = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 114],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 114, 131],
}

def _fast_local_search(U, time_limit=10.0):
    """Fast local search: add, remove, and replace elements to improve score."""
    U = list(sorted(set(U)))
    cur_score = _compute_score(U)
    best_score = cur_score
    best_U = list(U)
    start = time.time()
    rng = random.Random(hash(tuple(U)) % (2**31))
    
    while time.time() - start < time_limit and _elapsed() < _DEADLINE:
        improved = False
        max_u = max(U) if U else 1
        n = len(U)
        
        # Try adding a new element - focus on values that create unique differences
        for _ in range(30):
            if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                break
            # Try values that are "far" from existing elements to create new differences
            if rng.random() < 0.5:
                margin = max(10, max_u // 3)
                new_val = rng.randint(1, max_u + margin)
            else:
                # Try values between existing elements
                if n >= 2:
                    i = rng.randint(0, n - 2)
                    gap = U[i+1] - U[i]
                    if gap > 2:
                        new_val = rng.randint(U[i] + 1, U[i+1] - 1)
                    else:
                        new_val = rng.randint(1, max_u + max(5, max_u // 5))
                else:
                    new_val = rng.randint(1, max_u + max(5, max_u // 5))
            
            if new_val in U or new_val < 0:
                continue
            trial = sorted(U + [new_val])
            s = _compute_score(trial)
            if s > cur_score + 1e-12:
                cur_score = s
                U = trial
                if s > best_score:
                    best_score = s
                    best_U = list(U)
                improved = True
                break
        
        # Try removing an element (if set is large enough)
        if not improved and n > 3:
            for _ in range(15):
                if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                    break
                idx = rng.randint(1, n - 1)  # Don't remove 0
                trial = U[:idx] + U[idx+1:]
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    n = len(U)
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
                    break
        
        # Try replacing an element with a nearby value
        if not improved and n > 2:
            for _ in range(20):
                if time.time() - start > time_limit or _elapsed() > _DEADLINE:
                    break
                idx = rng.randint(1, n - 1)
                old_val = U[idx]
                # Try small perturbations
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                new_val = old_val + delta
                if new_val <= 0 or new_val in U:
                    # Try larger perturbation
                    margin = max(5, max_u // 5)
                    new_val = rng.randint(1, max_u + margin)
                    if new_val == old_val or new_val in U:
                        continue
                trial = sorted(U[:idx] + [new_val] + U[idx+1:])
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
                    break
        
        if not improved:
            # Try a larger perturbation: remove 2 elements, add 2 new ones
            if n > 4 and time.time() - start < time_limit * 0.8:
                idx1 = rng.randint(1, n - 2)
                idx2 = rng.randint(1, n - 1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % (n - 1) + 1
                trial = U[:]
                # Remove in reverse order to preserve indices
                for idx in sorted([idx1, idx2], reverse=True):
                    trial.pop(idx)
                # Add two new elements - try values that create unique differences
                for _ in range(2):
                    margin = max(10, max_u // 3)
                    new_val = rng.randint(1, max_u + margin)
                    if new_val not in trial and new_val >= 0:
                        trial.append(new_val)
                trial = sorted(trial)
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
        
        if not improved:
            # Try swapping two elements
            if n > 3 and time.time() - start < time_limit * 0.6:
                idx1 = rng.randint(1, n - 2)
                idx2 = rng.randint(1, n - 1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % (n - 1) + 1
                trial = U[:]
                # Swap values
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                s = _compute_score(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    if s > best_score:
                        best_score = s
                        best_U = list(U)
                    improved = True
        
        if not improved:
            break
    
    return best_U, best_score

def _greedy_prime_construct(max_val=200):
    """Greedy construction using primes to create unique differences."""
    primes = _primes_up_to(max_val)
    U = [0]
    for p in primes:
        if p > max_val:
            break
        trial = sorted(U + [p])
        # Check if this improves the score
        s = _compute_score(trial)
        if s > _compute_score(U) + 1e-12:
            U = trial
    return U

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    best_U = [0]
    best_score = 0.0
    
    # Phase 1: Evaluate known Golomb rulers
    for n, U in _KNOWN_GOLOMB.items():
        if _elapsed() > _DEADLINE - 10:
            break
        s = _compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Phase 2: Try greedy prime construction
    if _elapsed() < _DEADLINE - 30:
        U_prime = _greedy_prime_construct(max_val=150)
        s = _compute_score(U_prime)
        if s > best_score:
            best_score = s
            best_U = list(U_prime)
    
    # Phase 3: Local search from the best starting point
    if _elapsed() < _DEADLINE - 25:
        U, score = _fast_local_search(best_U, time_limit=25.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 4: Try local search from other Golomb rulers
    for n in [17, 16, 15, 14, 13, 12]:
        if _elapsed() > _DEADLINE - 20:
            break
        U_seed = list(_KNOWN_GOLOMB.get(n, [0, 1, 3]))
        U, score = _fast_local_search(U_seed, time_limit=8.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 5: Multiple restarts with perturbation from best solution
    rng = random.Random(42)
    for restart in range(4):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Perturb the best solution
        U = list(best_U)
        n = len(U)
        
        # Remove 1-2 random elements (not 0)
        if n > 2:
            num_remove = min(rng.randint(1, 2), n - 1)
            for _ in range(num_remove):
                idx = rng.randint(1, len(U) - 1)
                U.pop(idx)
        
        # Add 1-2 new random elements
        if U:
            max_u = max(U)
            for _ in range(rng.randint(1, 2)):
                new_val = rng.randint(1, max_u + max(10, max_u // 3))
                if new_val not in U:
                    U.append(new_val)
        
        U = sorted(U)
        U, score = _fast_local_search(U, time_limit=12.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Phase 6: Final polish with tighter local search
    if _elapsed() < _DEADLINE - 5:
        U, score = _fast_local_search(best_U, time_limit=10.0)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Ensure 0 is in the set and return sorted
    if 0 not in best_U:
        best_U.append(0)
    
    return sorted(best_U)
