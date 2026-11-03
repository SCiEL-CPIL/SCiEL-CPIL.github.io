"""Twin-prime ladder construction with local search for sumset/difference-set exponent."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _is_prime(n):
    """Check if n is prime."""
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

def _twin_prime_pairs(limit):
    """Generate list of twin prime pairs (p, p+2) up to limit."""
    pairs = []
    for p in range(2, limit - 1):
        if _is_prime(p) and _is_prime(p + 2):
            pairs.append((p, p + 2))
    return pairs

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy for moderate sizes."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    arr = np.array(sorted(U_set), dtype=np.int64)
    
    if n <= 150:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _construct_twin_prime_ladder(start_prime, num_pairs, use_gap=False):
    """
    Construct U as cumulative sum of twin prime pairs.
    U = {0, p1, p1+p2, p1+p2+p3, ...} where (p_i, p_i+2) are consecutive twin primes.
    If use_gap, we add the gap (p_i+2 - p_i = 2) to create more structure.
    """
    pairs = _twin_prime_pairs(start_prime + 1000)  # Get enough pairs
    
    if len(pairs) < num_pairs:
        # Extend the search
        pairs = _twin_prime_pairs(max(10000, start_prime + 10000))
    
    if len(pairs) < num_pairs:
        return None
    
    # Take the first num_pairs twin primes starting from start_prime
    selected = []
    for p, p2 in pairs:
        if p >= start_prime:
            selected.append(p)
        if len(selected) >= num_pairs:
            break
    
    if len(selected) < num_pairs:
        return None
    
    # Build cumulative sum
    U = [0]
    current = 0
    for p in selected:
        current += p
        U.append(current)
    
    return U

def _construct_twin_prime_with_gaps(start_prime, num_pairs):
    """
    Construct U using twin prime gaps more explicitly.
    Each step adds p_i, then p_i+2, creating a staircase pattern.
    """
    pairs = _twin_prime_pairs(start_prime + 5000)
    
    if len(pairs) < num_pairs:
        pairs = _twin_prime_pairs(max(10000, start_prime + 10000))
    
    if len(pairs) < num_pairs:
        return None
    
    selected = []
    for p, p2 in pairs:
        if p >= start_prime:
            selected.append((p, p2))
        if len(selected) >= num_pairs:
            break
    
    if len(selected) < num_pairs:
        return None
    
    U = [0]
    current = 0
    for p, p2 in selected:
        current += p
        U.append(current)
        # Optionally add the twin partner offset
        current += 2  # gap of 2
        U.append(current)
    
    return U

def _construct_prime_ladder(start_prime, num_primes):
    """Construct U as cumulative sum of consecutive primes."""
    primes = []
    p = start_prime
    while len(primes) < num_primes:
        if _is_prime(p):
            primes.append(p)
        p += 1
    
    if len(primes) < num_primes:
        return None
    
    U = [0]
    current = 0
    for p in primes:
        current += p
        U.append(current)
    
    return U

def _local_search(U, max_iters=2000, deadline=None, max_size=40):
    """Improve set U via local search."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 50
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            candidates = []
            for delta in [-3, -2, -1, 1, 2, 3, 5, 8, 13, 21]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            for _ in range(2):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = _score_set(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                for new_val in range(0, upper_bound + 1):
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
        
        if not improved:
            # Try removing an element
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random mutation
            if random.random() < 0.15:
                mutated = U[:]
                for _ in range(random.randint(1, 2)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                removable = [x for x in mutated if x != 0]
                if removable and random.random() < 0.3:
                    idx = random.randint(0, len(removable) - 1)
                    val = removable.pop(idx)
                    mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = _score_set(mutated)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 50:
            upper_bound += 30
            U = best_U[:]
    
    return sorted(set(best_U)), best_score

def _exhaustive_small_search(max_size=6, max_val=30, deadline=None):
    """Exhaustive search over small sets to find good candidates."""
    best_score = -1
    best_U = [0]
    
    # Try all subsets of {0, 1, ..., max_val} of size up to max_size
    # This is expensive, so we do it selectively
    if max_size <= 5 and max_val <= 20:
        for n in range(2, max_size + 1):
            if deadline and time.time() > deadline:
                break
            for combo in combinations(range(1, max_val + 1), n - 1):
                if deadline and time.time() > deadline:
                    break
                U = [0] + list(combo)
                U = sorted(U)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Phase 1: Twin-prime ladder constructions
    for start_p in [5, 11, 17, 29, 41, 59, 71, 101, 139, 179]:
        if time.time() > deadline:
            break
        for num_pairs in range(2, 12):
            if time.time() > deadline:
                break
            
            # Standard cumulative sum
            U1 = _construct_twin_prime_ladder(start_p, num_pairs)
            if U1:
                s = _score_set(U1)
                if s > best_score:
                    best_score = s
                    best_U = U1[:]
            
            # With gaps
            U2 = _construct_twin_prime_with_gaps(start_p, num_pairs)
            if U2:
                s = _score_set(U2)
                if s > best_score:
                    best_score = s
                    best_U = U2[:]
    
    # Phase 2: Prime ladder constructions
    for start_p in [2, 3, 5, 7, 11, 13, 17, 19]:
        if time.time() > deadline:
            break
        for num_primes in range(3, 15):
            if time.time() > deadline:
                break
            U = _construct_prime_ladder(start_p, num_primes)
            if U:
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Phase 3: Exhaustive small search
    if time.time() < deadline - 30:
        ex_U, ex_score = _exhaustive_small_search(max_size=5, max_val=25, deadline=deadline)
        if ex_score > best_score:
            best_score = ex_score
            best_U = ex_U[:]
    
    # Phase 4: Local search on best candidate
    if time.time() < deadline - 10:
        improved_U, improved_score = _local_search(best_U, max_iters=3000, deadline=deadline, max_size=35)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 5: Local search on diverse candidates
    if time.time() < deadline - 5:
        candidates = []
        
        # Add some structured candidates
        for start_p in [5, 11, 17]:
            if time.time() > deadline:
                break
            for num_pairs in [4, 6, 8]:
                U = _construct_twin_prime_ladder(start_p, num_pairs)
                if U:
                    candidates.append(U)
        
        # Random candidates
        random.seed(42)
        for _ in range(8):
            if time.time() > deadline:
                break
            n = random.randint(4, 12)
            max_val = random.randint(15, 60)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            candidates.append(U)
        
        for U in candidates:
            if time.time() > deadline - 5:
                break
            s = _score_set(U)
            if s > best_score - 0.03:
                improved_U, improved_score = _local_search(U, max_iters=1500, deadline=deadline, max_size=30)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
