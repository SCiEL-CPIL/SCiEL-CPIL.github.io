"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _score_set(U):
    U = sorted(set(U))
    if 0 not in U or len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _score_set_fast(U, sums_cache, diffs_cache, max_u):
    """Fast score computation when we have cached sums/diffs."""
    n_sums = len(sums_cache)
    n_diffs = len(diffs_cache)
    if n_sums == 0 or n_diffs == 0:
        return -1
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return -1
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1 + math.log(ratio) / denom

def _build_caches(U):
    """Build sums and diffs caches for set U."""
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    return sums, diffs

def _prime_gaps():
    """Generate a list of small distinct primes for use as step sizes."""
    primes = []
    n = 2
    while len(primes) < 40:
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
        n += 1
    return primes

def _prime_based_construct(deadline, max_steps=12, max_prime_idx=20):
    """Construct sets using prime-step gaps to minimize sumset collisions."""
    primes = _prime_gaps()
    best_set = [0]
    best_score = -1
    
    # Greedy: build up by adding elements with prime-step gaps
    U = [0]
    current_pos = 0
    used_primes = set()
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
        
        # Try adding an element at current_pos + p for various primes
        best_candidate = None
        best_candidate_score = -1
        
        for p_idx in range(min(max_prime_idx, len(primes))):
            p = primes[p_idx]
            if p in used_primes:
                continue
            new_elem = current_pos + p
            if new_elem <= 0:
                continue
            new_U = sorted(U + [new_elem])
            s = _score_set(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = new_U
                best_candidate_prime = p
        
        if best_candidate is not None and best_candidate_score > best_score:
            U = best_candidate
            best_set = U[:]
            best_score = best_candidate_score
            used_primes.add(best_candidate_prime)
            current_pos = max(U)
        else:
            # Try a different strategy: add element not necessarily at current_pos + p
            # but somewhere that improves the score
            improved = False
            max_u = max(U)
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                # Random element in a reasonable range
                new_elem = random.randint(1, max_u * 2 + 10)
                if new_elem in U:
                    continue
                new_U = sorted(U + [new_elem])
                s = _score_set(new_U)
                if s > best_score + 1e-12:
                    U = new_U
                    best_set = U[:]
                    best_score = s
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_set, best_score

def _local_search(U, deadline, max_iter=5000):
    """Local search with incremental scoring: add, remove, or swap elements."""
    U = sorted(set(U))
    current_score = _score_set(U)
    if current_score < 0:
        return U, current_score
    
    # Build initial caches
    sums_cache, diffs_cache = _build_caches(U)
    max_u = max(U)
    
    for _ in range(max_iter):
        if time.time() > deadline - 2:
            break
        improved = False
        U_set = set(U)
        
        # Try adding an element - incremental scoring
        # Expand search range based on current max
        search_upper = max_u + 40
        for new_elem in range(1, search_upper + 1):
            if new_elem in U_set:
                continue
            # Compute new sums and diffs incrementally
            new_sums = sums_cache.copy()
            new_diffs = diffs_cache.copy()
            for u in U:
                new_sums.add(new_elem + u)
                new_diffs.add(new_elem - u)
                new_diffs.add(u - new_elem)
            new_max = max(max_u, new_elem)
            s = _score_set_fast(U, new_sums, new_diffs, new_max)
            if s > current_score + 1e-12:
                U = sorted(U + [new_elem])
                U_set = set(U)
                current_score = s
                sums_cache = new_sums
                diffs_cache = new_diffs
                max_u = new_max
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element
        for i in range(1, len(U)):
            elem_to_remove = U[i]
            if len(U) < 3:
                continue
            # Rebuild without this element
            new_U = U[:i] + U[i+1:]
            new_sums, new_diffs = _build_caches(new_U)
            new_max = max(new_U)
            s = _score_set_fast(U, new_sums, new_diffs, new_max)
            if s > current_score + 1e-12:
                U = new_U
                U_set = set(U)
                current_score = s
                sums_cache = new_sums
                diffs_cache = new_diffs
                max_u = new_max
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping an element - more aggressive with wider range
        for i in range(1, len(U)):
            old_elem = U[i]
            # Try a range of replacement values
            for new_elem in range(1, max_u + 50):
                if new_elem in U_set or new_elem == old_elem:
                    continue
                new_U = U[:i] + [new_elem] + U[i+1:]
                new_U_sorted = sorted(new_U)
                new_sums, new_diffs = _build_caches(new_U_sorted)
                new_max = max(new_U_sorted)
                s = _score_set_fast(U, new_sums, new_diffs, new_max)
                if s > current_score + 1e-12:
                    U = new_U_sorted
                    U_set = set(U)
                    current_score = s
                    sums_cache = new_sums
                    diffs_cache = new_diffs
                    max_u = new_max
                    improved = True
                    break
            if improved:
                break
        
        # Try 2-swap: remove one, add another
        if not improved and len(U) >= 3:
            for i in range(1, len(U)):
                elem_to_remove = U[i]
                new_U_without = U[:i] + U[i+1:]
                for new_elem in range(1, max_u + 30):
                    if new_elem in set(new_U_without):
                        continue
                    new_U = sorted(new_U_without + [new_elem])
                    new_sums, new_diffs = _build_caches(new_U)
                    new_max = max(new_U)
                    s = _score_set_fast(U, new_sums, new_diffs, new_max)
                    if s > current_score + 1e-12:
                        U = new_U
                        U_set = set(U)
                        current_score = s
                        sums_cache = new_sums
                        diffs_cache = new_diffs
                        max_u = new_max
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try random perturbation to escape local optimum
            if time.time() > deadline - 5:
                break
            random_improved = False
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                # Randomly modify the set
                new_U = U[:]
                if random.random() < 0.5 and len(new_U) > 2:
                    # Remove a random element
                    idx = random.randint(1, len(new_U) - 1)
                    new_U.pop(idx)
                else:
                    # Add a random element
                    new_elem = random.randint(1, max_u + 60)
                    if new_elem not in new_U:
                        new_U.append(new_elem)
                new_U = sorted(set(new_U))
                new_sums, new_diffs = _build_caches(new_U)
                new_max = max(new_U)
                s = _score_set_fast(U, new_sums, new_diffs, new_max)
                if s > current_score + 1e-12:
                    U = new_U
                    U_set = set(U)
                    current_score = s
                    sums_cache = new_sums
                    diffs_cache = new_diffs
                    max_u = new_max
                    random_improved = True
                    break
            if random_improved:
                continue
            break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Systematic search for size 3 - expanded range
    for a in range(1, 35):
        for b in range(a + 1, 60):
            if time.time() > deadline - 5:
                return best_set
            U = [0, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Systematic search for size 4 - expanded range
    for a in range(1, 25):
        for b in range(a + 1, 40):
            for c in range(b + 1, 65):
                if time.time() > deadline - 5:
                    return best_set
                U = [0, a, b, c]
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Systematic search for size 5 - limited
    if time.time() < deadline - 30:
        for a in range(1, 15):
            for b in range(a + 1, 25):
                for c in range(b + 1, 35):
                    for d in range(c + 1, 50):
                        if time.time() > deadline - 5:
                            break
                        U = [0, a, b, c, d]
                        s = _score_set(U)
                        if s > best_score:
                            best_score = s
                            best_set = U[:]
    
    # Prime-based construction
    if time.time() < deadline - 20:
        prime_set, prime_score = _prime_based_construct(deadline, max_steps=12, max_prime_idx=20)
        if prime_score > best_score:
            best_set = prime_set
            best_score = prime_score
    
    # Local search from best found with more iterations
    if time.time() < deadline - 15:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=10000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Try local search from several promising starting points
    starting_points = [
        [0, 1, 4, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 2, 5, 9],
        [0, 1, 5, 8],
        [0, 1, 4, 7],
        [0, 2, 4, 7],
        [0, 1, 3, 5],
        [0, 1, 4, 9],
        [0, 2, 5, 11],
        [0, 1, 6, 10],
        [0, 3, 5, 8],
        [0, 2, 7, 13],
        [0, 3, 7, 11],
        [0, 1, 5, 12],
        [0, 1, 3, 8, 12],
        [0, 1, 4, 8, 13],
        [0, 2, 5, 9, 14],
        [0, 1, 5, 9, 14],
        [0, 2, 6, 10, 15],
    ]
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s = _score_set(start)
        if s > best_score - 0.15:
            improved_set, improved_score = _local_search(start, deadline, max_iter=5000)
            if improved_score > best_score:
                best_set = improved_set
                best_score = improved_score
    
    # Final polish: try more aggressive local search from best
    if time.time() < deadline - 3:
        improved_set, improved_score = _local_search(best_set, deadline, max_iter=5000)
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    if time.time() > deadline - 1:
        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
