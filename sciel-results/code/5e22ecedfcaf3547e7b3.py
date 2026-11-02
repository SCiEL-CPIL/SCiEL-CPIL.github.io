"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set efficiently for small n
    sumset = set()
    diffset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
            diffset.add(a - b)
            
    # Avoid division by zero
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _search_small_sets(max_n=6, max_val=40, deadline=None):
    """Exhaustive search for small sets U."""
    best_score = 0.0
    best_U = [0, 1, 3]
    
    start_time = time.time()
    
    for n in range(3, max_n + 1):
        for combo in combinations(range(1, max_val + 1), n - 1):
            if deadline and time.time() > deadline:
                return best_U, best_score
                
            U = [0] + list(combo)
            U.sort()
            
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _generate_prime_gap_sets(deadline=None):
    """Generate sets based on prime gaps: {0, p1, p1+p2, ...}."""
    candidates = []
    primes = [p for p in range(2, 200) if _is_prime(p)]
    
    # Try various lengths and prime selections
    for n in range(2, 8):
        # Use combinations of n-1 distinct primes
        # To keep it fast, limit the range of primes considered for larger n
        max_prime_idx = min(len(primes), 50)
        for combo in combinations(primes[:max_prime_idx], n - 1):
            if deadline and time.time() > deadline:
                return candidates
            
            U = [0]
            current = 0
            for p in combo:
                current += p
                U.append(current)
            
            if len(set(U)) == len(U):
                candidates.append(U)
                
    return candidates

def _greedy_prime_optimization(U, deadline=None):
    """Optimize a prime-gap set by swapping primes."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    
    # Reconstruct the primes from U
    # U = [0, p1, p1+p2, ...]
    primes = []
    prev = 0
    for x in U[1:]:
        primes.append(x - prev)
        prev = x
        
    # Greedy swap
    for _ in range(20):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(primes)):
            old_p = primes[i]
            # Try swapping with a nearby prime
            for new_p in range(2, 200):
                if not _is_prime(new_p) or new_p == old_p:
                    continue
                # Check if new_p is already in primes (must be distinct)
                if new_p in primes and primes.index(new_p) != i:
                    continue
                
                new_primes = primes[:]
                new_primes[i] = new_p
                
                new_U = [0]
                curr = 0
                for p in new_primes:
                    curr += p
                    new_U.append(curr)
                    
                if len(set(new_U)) != len(new_U):
                    continue
                    
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    primes = new_primes
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def _construct_heuristic_sets(deadline=None):
    """Construct sets based on known good structures."""
    candidates = []
    
    # Structure 1: Almost AP
    for k in range(3, 10):
        for M in range(k, 50):
            if deadline and time.time() > deadline:
                break
            U = list(range(k - 1)) + [M]
            if len(set(U)) == len(U):
                candidates.append(U)
                
    # Structure 2: Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
    ]
    candidates.extend(known_gr)
    
    # Structure 3: Randomized local search on AP
    random.seed(42)
    for _ in range(50):
        if deadline and time.time() > deadline:
            break
        n = random.randint(3, 8)
        base_ap = list(range(n))
        i = random.randint(1, n - 1)
        new_val = random.randint(1, 60)
        U = base_ap[:]
        U[i] = new_val
        if len(set(U)) == len(U) and 0 in U:
            candidates.append(sorted(set(U)))
            
    # Structure 4: Prime gap sets
    prime_sets = _generate_prime_gap_sets(deadline)
    candidates.extend(prime_sets)
    
    return candidates

def _local_search(U, max_iters=200, deadline=None):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 20
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(len(U)):
            old_val = U[i]
            if old_val == 0:
                continue
            for new_val in range(1, upper_bound + 1):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # 1. Exhaustive search for small sets
    try:
        U_sol, s_sol = _search_small_sets(max_n=5, max_val=50, deadline=deadline)
        if s_sol > best_score:
            best_score = s_sol
            best_U = U_sol
    except Exception:
        pass
        
    # 2. Heuristic candidates (including prime gaps)
    candidates = _construct_heuristic_sets(deadline=deadline)
    
    # Track the best prime-gap candidate specifically for greedy optimization
    best_prime_U = None
    best_prime_score = 0.0
    
    for U in candidates:
        if deadline and time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
        # Identify if it's a prime-gap set (simple check: differences are prime)
        diffs = [U[i+1]-U[i] for i in range(len(U)-1)]
        if all(_is_prime(d) for d in diffs):
            if s > best_prime_score:
                best_prime_score = s
                best_prime_U = U[:]
                
    # 3. Greedy optimization on best prime-gap set
    if best_prime_U and time.time() < deadline:
        opt_U, opt_score = _greedy_prime_optimization(best_prime_U, deadline=deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U
            
    # 4. Local search on best found
    if time.time() < deadline:
        best_U, best_score = _local_search(best_U, max_iters=100, deadline=deadline)
        
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
