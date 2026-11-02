"""Gap-multiplication construction for sumset/difference-set exponent lower bound."""
import time
import math
import random
from itertools import combinations

_DEADLINE = 150.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _deadline_reached(margin=2.0):
    return _elapsed() > _DEADLINE - margin

def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U or max(U) == 0:
        return 0.0
    U_set = set(U)
    max_val = max(U)
    
    # Compute sumset U+U
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    n_sum = len(sumset)
    n_diff = len(diffset)
    
    if n_sum == 0 or n_diff == 0:
        return 0.0
    
    log_arg = n_diff / n_sum
    if log_arg <= 0:
        return 0.0
    
    log_denom = math.log(2 * max_val + 1)
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + math.log(log_arg) / log_denom

def _primes_up_to(n):
    """Generate primes up to n using sieve."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _build_from_gaps(gaps):
    """Build set from cumulative sum of gaps. Returns sorted list including 0."""
    U = [0]
    current = 0
    for g in gaps:
        current += g
        U.append(current)
    return U

def _greedy_prime_gaps(max_gaps, primes_pool, initial_score, deadline_margin=5.0):
    """
    Greedily select gaps as products of distinct primes to maximize instantaneous score improvement.
    """
    gaps = []
    U = [0]
    current = 0
    best_score = initial_score
    
    # Precompute available primes
    available_primes = list(primes_pool)
    
    for _ in range(max_gaps):
        if _deadline_reached(deadline_margin):
            break
        
        best_gap = None
        best_new_score = best_score
        best_new_U = None
        
        # Try single primes and products of 2-3 small primes
        candidates = []
        # Single primes
        for p in available_primes[:50]:
            candidates.append(p)
        # Products of 2 primes
        for i in range(min(20, len(available_primes))):
            for j in range(i + 1, min(30, len(available_primes))):
                candidates.append(available_primes[i] * available_primes[j])
        # Products of 3 primes (small)
        for i in range(min(10, len(available_primes))):
            for j in range(i + 1, min(15, len(available_primes))):
                for k in range(j + 1, min(20, len(available_primes))):
                    candidates.append(available_primes[i] * available_primes[j] * available_primes[k])
        
        # Also try some larger single values
        for v in range(2, 200):
            if v not in candidates:
                candidates.append(v)
        
        # Deduplicate and sort by size
        candidates = list(set(candidates))
        candidates.sort()
        
        # Limit candidates for speed
        if len(candidates) > 200:
            candidates = candidates[:200]
        
        for g in candidates:
            if g <= 0:
                continue
            new_U = U + [current + g]
            new_score = _score_set(new_U)
            if new_score > best_new_score + 1e-12:
                best_new_score = new_score
                best_gap = g
                best_new_U = new_U
        
        if best_gap is not None:
            gaps.append(best_gap)
            current += best_gap
            U = best_new_U
            best_score = best_new_score
        else:
            # No improvement found; try a random small gap to break symmetry
            g = random.choice([1, 2, 3, 4, 5])
            gaps.append(g)
            current += g
            U.append(current)
            best_score = _score_set(U)
    
    return gaps, U, best_score

def _local_search_gaps(gaps, U, score, max_iterations=500, deadline_margin=5.0):
    """
    Local search: try replacing or adjusting individual gaps.
    """
    best_gaps = list(gaps)
    best_U = list(U)
    best_score = score
    
    for iteration in range(max_iterations):
        if _deadline_reached(deadline_margin):
            break
        
        improved = False
        n = len(best_gaps)
        
        # Try modifying each gap
        for i in range(n):
            if _deadline_reached(deadline_margin):
                break
            
            orig_gap = best_gaps[i]
            # Try nearby values
            deltas = [-3, -2, -1, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
            
            for d in deltas:
                new_gap = orig_gap + d
                if new_gap < 1:
                    continue
                
                new_gaps = list(best_gaps)
                new_gaps[i] = new_gap
                new_U = _build_from_gaps(new_gaps)
                new_score = _score_set(new_U)
                
                if new_score > best_score + 1e-12:
                    best_gaps = new_gaps
                    best_U = new_U
                    best_score = new_score
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try adding a new gap
            if len(best_gaps) < 30:
                best_pos = random.randint(0, len(best_gaps))
                for trial_gap in [2, 3, 5, 7, 11, 13, 17, 23, 31, 43, 59, 73, 97, 127, 163, 199, 251, 311, 389, 479]:
                    new_gaps = list(best_gaps)
                    new_gaps.insert(best_pos, trial_gap)
                    new_U = _build_from_gaps(new_gaps)
                    new_score = _score_set(new_U)
                    if new_score > best_score + 1e-12:
                        best_gaps = new_gaps
                        best_U = new_U
                        best_score = new_score
                        improved = True
                        break
            
            if not improved:
                # Try removing a gap
                if len(best_gaps) > 3:
                    for i in range(len(best_gaps)):
                        new_gaps = list(best_gaps)
                        new_gaps.pop(i)
                        new_U = _build_from_gaps(new_gaps)
                        new_score = _score_set(new_U)
                        if new_score > best_score + 1e-12:
                            best_gaps = new_gaps
                            best_U = new_U
                            best_score = new_score
                            improved = True
                            break
            
            if not improved:
                break
    
    return best_gaps, best_U, best_score

def _random_restart_gaps(n_gaps, max_gap, deadline_margin=5.0):
    """Generate random gaps for restart."""
    gaps = []
    for _ in range(n_gaps):
        gaps.append(random.randint(1, max_gap))
    return gaps

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_score = 0.0
    best_U = [0, 1, 3]  # Fallback
    best_gaps = [1, 2]
    
    # Phase 1: Greedy prime-based construction
    primes_pool = _primes_up_to(500)
    
    for max_gaps in [5, 8, 12, 15, 20]:
        if _deadline_reached(10.0):
            break
        
        gaps, U, score = _greedy_prime_gaps(max_gaps, primes_pool, 0.0, deadline_margin=8.0)
        
        if score > best_score:
            best_score = score
            best_U = U
            best_gaps = gaps
        
        # Phase 2: Local search refinement
        if _deadline_reached(8.0):
            break
        
        refined_gaps, refined_U, refined_score = _local_search_gaps(
            gaps, U, score, max_iterations=300, deadline_margin=6.0
        )
        
        if refined_score > best_score:
            best_score = refined_score
            best_U = refined_U
            best_gaps = refined_gaps
    
    # Phase 3: Random restarts with local search
    if not _deadline_reached(10.0):
        for restart in range(10):
            if _deadline_reached(8.0):
                break
            
            n_gaps = random.randint(4, 15)
            max_gap = random.randint(10, 200)
            gaps = _random_restart_gaps(n_gaps, max_gap, deadline_margin=5.0)
            U = _build_from_gaps(gaps)
            score = _score_set(U)
            
            # Local search from random start
            if not _deadline_reached(5.0):
                refined_gaps, refined_U, refined_score = _local_search_gaps(
                    gaps, U, score, max_iterations=200, deadline_margin=4.0
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_U = refined_U
                    best_gaps = refined_gaps
    
    # Final polish: try small perturbations on the best solution
    if not _deadline_reached(3.0) and len(best_gaps) > 0:
        final_gaps, final_U, final_score = _local_search_gaps(
            best_gaps, best_U, best_score, max_iterations=100, deadline_margin=2.0
        )
        if final_score > best_score:
            best_score = final_score
            best_U = final_U
            best_gaps = final_gaps
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U = [0] + best_U
    
    # Sort and deduplicate
    best_U = sorted(set(best_U))
    
    return best_U
