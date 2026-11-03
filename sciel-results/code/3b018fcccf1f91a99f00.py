"""Improved program: optimized sumset/difference-set exponent search."""
import time
import math
import random

# EVOLVE-BLOCK-START
def _score_set_fast(U):
    """Compute the score for a set U of nonnegative integers containing 0.
    
    Uses vectorized-like approach with sorted arrays for efficiency.
    """
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Compute |U+U| and |U-U| efficiently
    # U+U: all pairwise sums
    # U-U: all pairwise differences (can be negative)
    
    sum_set = set()
    diff_set = set()
    
    # For efficiency, use the fact that we can break early if sets get too large
    for i in range(n):
        a = U[i]
        for j in range(i, n):  # i<=j for sums (symmetric)
            s = a + U[j]
            sum_set.add(s)
            # For differences, we need all pairs
            diff_set.add(a - U[j])
            if j > i:
                diff_set.add(U[j] - a)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1.0:
        return 1.0  # Minimum possible score
    
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 1.0
    
    score = 1 + math.log(ratio) / denom
    return score


def _greedy_construct(max_elements=15, max_val=500):
    """Greedy construction: add elements that maximize marginal score improvement."""
    U = [0]
    best_score = 1.0
    
    random.seed(42)
    
    # Candidate pool
    candidates = set(range(1, max_val + 1))
    
    for _ in range(max_elements):
        if not candidates:
            break
        
        best_elem = None
        best_new_score = best_score
        
        # Sample a subset of candidates for efficiency
        sample_size = min(len(candidates), 50)
        sample = random.sample(sorted(candidates), sample_size)
        
        for x in sample:
            new_U = U + [x]
            s = _score_set_fast(new_U)
            if s > best_new_score:
                best_new_score = s
                best_elem = x
        
        if best_elem is not None:
            U.append(best_elem)
            U.sort()
            candidates.discard(best_elem)
            best_score = best_new_score
    
    return U, best_score


def _local_search_improved(start_set, max_iter=5000, max_val=200, temperature=0.1):
    """Improved local search with simulated annealing."""
    U = sorted(set(start_set))
    if 0 not in U:
        U.insert(0, 0)
    
    best = U[:]
    best_score = _score_set_fast(U)
    current = U[:]
    current_score = best_score
    
    random.seed(123)
    
    for i in range(max_iter):
        # Temperature schedule
        temp = temperature * (1.0 - i / max_iter) + 0.01
        
        # Generate neighbor
        new_U = current[:]
        action = random.random()
        
        if action < 0.35 and len(new_U) < 25:
            # Add element
            x = random.randint(1, max_val)
            if x not in new_U:
                new_U.append(x)
        elif action < 0.65 and len(new_U) > 2:
            # Remove element
            non_zero = [x for x in new_U if x != 0]
            if non_zero:
                new_U.remove(random.choice(non_zero))
        else:
            # Swap
            if len(new_U) > 2:
                non_zero = [x for x in new_U if x != 0]
                if non_zero:
                    old = random.choice(non_zero)
                    new_U.remove(old)
                    new_val = random.randint(1, max_val)
                    if new_val not in new_U:
                        new_U.append(new_val)
        
        new_U = sorted(set(new_U))
        if 0 not in new_U:
            new_U.insert(0, 0)
        
        new_score = _score_set_fast(new_U)
        
        # Accept if better or with probability exp(-delta/temp)
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
            current = new_U
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best = new_U[:]
    
    return best, best_score


def _structured_candidates():
    """Generate various structured candidate sets."""
    candidates = []
    
    # Geometric progressions with various ratios
    for ratio in range(2, 10):
        for terms in range(3, 12):
            U = [0]
            val = 1
            for _ in range(terms):
                U.append(val)
                val *= ratio
            candidates.append(U)
    
    # Powers of 2
    for n in range(3, 14):
        candidates.append([0] + [2**i for i in range(n)])
    
    # Sparse arithmetic progressions
    for step in range(2, 15):
        for length in range(3, 10):
            candidates.append([0] + [step * i for i in range(1, length)])
    
    # Mixed: geometric + offset
    for base in range(2, 6):
        for offset in range(1, 5):
            U = [0]
            val = offset
            for _ in range(8):
                U.append(val)
                val = val * base + offset
            candidates.append(U)
    
    # Prime-based sparse sets
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for start in range(len(primes) - 5):
        for count in range(3, 8):
            U = [0]
            for i in range(start, min(start + count, len(primes))):
                U.append(primes[i])
            candidates.append(U)
    
    # Random geometric-like
    random.seed(99)
    for _ in range(20):
        U = [0]
        val = random.randint(1, 5)
        for _ in range(random.randint(5, 12)):
            U.append(val)
            val = val * random.randint(2, 4) + random.randint(0, 3)
        candidates.append(sorted(set(U)))
    
    return candidates


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_set = [0, 1, 3]
    best_score = _score_set_fast(best_set)
    
    # Phase 1: Evaluate structured candidates
    structured = _structured_candidates()
    for cand in structured:
        if time.time() >= deadline:
            break
        s = _score_set_fast(cand)
        if s > best_score:
            best_score = s
            best_set = cand[:]
    
    # Phase 2: Greedy construction from multiple seeds
    seeds = [best_set, [0, 1, 2], [0, 1, 4], [0, 2, 5], [0, 3, 7]]
    for seed in seeds:
        if time.time() >= deadline:
            break
        greedy_set, greedy_score = _greedy_construct(max_elements=12, max_val=300)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = greedy_set
    
    # Phase 3: Local search with simulated annealing
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        max_iter = int(min(8000, remaining_time * 50))  # ~50 iterations/sec estimate
        ls_set, ls_score = _local_search_improved(best_set, max_iter=max_iter, max_val=150, temperature=0.15)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    # Phase 4: Final polish with tighter parameters
    if time.time() < deadline - 5:
        remaining_time = deadline - time.time()
        max_iter = int(min(3000, remaining_time * 40))
        final_set, final_score = _local_search_improved(best_set, max_iter=max_iter, max_val=100, temperature=0.05)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    
    return best_set
# EVOLVE-BLOCK-END
