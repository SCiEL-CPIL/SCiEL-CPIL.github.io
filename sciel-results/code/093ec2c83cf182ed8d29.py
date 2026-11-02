"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

def _compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            sum_set.add(U[i] + U[j])
        for j in range(n):
            diff_set.add(U[i] - U[j])
    if len(sum_set) == 0:
        return 0.0
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _generate_primes(limit):
    """Generate primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _greedy_prime_step_extended(deadline, max_elements=30):
    """
    Enhanced greedy construction using prime-step gaps with wider prime selection.
    U = {0, p1, p1+p2, p1+p2+p3, ...}
    At each step, choose the next prime p_k from a pool to maximize the instantaneous score.
    """
    primes = _generate_primes(500)
    if len(primes) < max_elements:
        primes += _generate_primes(2000)[len(primes):]
    
    current = [0]
    best_U = [0, 1]
    best_score = _compute_score([0, 1])
    
    # First element: try small primes
    for p in primes[:10]:
        candidate = [0, p]
        s = _compute_score(candidate)
        if s > best_score:
            best_score = s
            best_U = candidate
            current = candidate
        if time.time() > deadline - 100:
            break
    
    # Greedily add more elements
    used_primes = set()
    for _ in range(max_elements - 1):
        if time.time() > deadline - 80:
            break
        
        best_candidate = None
        best_candidate_score = best_score
        
        # Try each available prime
        for p in primes:
            if p in used_primes:
                continue
            new_val = current[-1] + p
            candidate = current + [new_val]
            s = _compute_score(candidate)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = candidate
                best_prime = p
        
        if best_candidate is not None:
            best_U = best_candidate
            best_score = best_candidate_score
            current = best_candidate
            used_primes.add(best_prime)
        else:
            break
    
    return best_U

def _local_search_enhanced(U, deadline, max_iter=30000):
    """
    Enhanced local search with adaptive perturbation sizes and better acceptance.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _compute_score(best_U)
    
    rng = random.Random(12345)
    base_perturbation = 10
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Pick a random index to modify (not 0)
        idx = rng.randint(1, len(best_U) - 1)
        old_val = best_U[idx]
        
        # Adaptive perturbation size: larger when stuck
        if it % 500 == 0:
            base_perturbation = max(5, min(100, base_perturbation * 1.2))
        
        # Try multiple perturbation sizes
        improved = False
        for scale in [1, 2, 5, 10]:
            perturbation = int(base_perturbation * scale)
            if perturbation < 1:
                perturbation = 1
            
            for _ in range(5):  # Try 5 random deltas per scale
                delta = rng.randint(-perturbation, perturbation)
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 0:
                    continue
                
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                
                if 0 not in new_U:
                    continue
                if len(new_U) != len(best_U):
                    continue
                
                s = _compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
            
            if improved:
                break
        
        # Occasionally try swapping two elements
        if not improved and it % 100 == 0:
            if len(best_U) >= 3:
                i, j = rng.sample(range(1, len(best_U)), 2)
                new_U = best_U[:]
                new_U[i], new_U[j] = new_U[j], new_U[i]
                new_U = sorted(set(new_U))
                if 0 in new_U and len(new_U) == len(best_U):
                    s = _compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
    
    return best_U

def _scale_and_truncate(U, deadline, scales=[1, 2, 3, 5, 7, 11, 13]):
    """Try scaling the set and truncating to different sizes."""
    best_U = U[:]
    best_score = _compute_score(U)
    
    for scale in scales:
        if time.time() > deadline - 15:
            break
        
        scaled = [x * scale for x in U]
        s = _compute_score(scaled)
        if s > best_score:
            best_score = s
            best_U = scaled
        
        # Try truncating to smaller sizes
        for target_size in range(2, len(U) + 1):
            if time.time() > deadline - 10:
                break
            truncated = scaled[:target_size]
            s = _compute_score(truncated)
            if s > best_score:
                best_score = s
                best_U = truncated
    
    return best_U

def _try_add_remove(U, deadline):
    """Combined add/remove element search."""
    best_U = U[:]
    best_score = _compute_score(U)
    
    max_u = U[-1] if U else 0
    
    # Try adding elements
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        v = random.randint(1, max_u + 100)
        if v in U:
            continue
        new_U = sorted(U + [v])
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
            max_u = best_U[-1]
    
    # Try removing elements
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        if len(best_U) <= 2:
            break
        i = random.randint(1, len(best_U) - 1)
        new_U = best_U[:i] + best_U[i+1:]
        s = _compute_score(new_U)
        if s > best_score + 1e-12:
            best_score = s
            best_U = new_U
    
    return best_U

def construct():
    deadline = time.time() + 135  # Safety margin from 150s
    
    # Generate multiple diverse starting configurations
    start_variants = []
    
    # 1. Enhanced greedy prime-step
    if time.time() < deadline - 60:
        start_variants.append(_greedy_prime_step_extended(deadline, max_elements=25))
    
    # 2. Different initial bases
    for base in [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 7]]:
        if time.time() > deadline - 50:
            break
        current = list(base)
        primes = _generate_primes(100)
        for p in primes[:15]:
            candidate = current + [current[-1] + p]
            start_variants.append(candidate)
            current = candidate
    
    # 3. Superincreasing-like sequences
    for growth in [2, 3, 4]:
        if time.time() > deadline - 40:
            break
        current = [0, 1]
        for _ in range(10):
            next_val = current[-1] * growth + 1
            current.append(next_val)
        start_variants.append(current)
    
    # Pick the best starting variant
    best_start = None
    best_start_score = -1
    for sv in start_variants:
        s = _compute_score(sv)
        if s > best_start_score:
            best_start_score = s
            best_start = sv
    
    U = best_start if best_start else [0, 1, 3]
    
    # Enhanced local search
    if time.time() < deadline - 40:
        U = _local_search_enhanced(U, deadline, max_iter=20000)
    
    # Scale and truncate
    if time.time() < deadline - 20:
        U = _scale_and_truncate(U, deadline)
    
    # Try adding/removing elements
    if time.time() < deadline - 15:
        U = _try_add_remove(U, deadline)
    
    # Final local search polish
    if time.time() < deadline - 10:
        U = _local_search_enhanced(U, deadline, max_iter=10000)
    
    # Ensure 0 is in the set and all are nonnegative
    U = sorted(set([x for x in U if x >= 0]))
    if 0 not in U:
        U = [0] + U
    
    return U

# EVOLVE-BLOCK-END
