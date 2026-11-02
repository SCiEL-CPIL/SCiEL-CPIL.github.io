"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    
    U_list = list(U)
    # Using list comprehension for speed
    sum_set = {a + b for a in U_list for b in U_list}
    diff_set = {a - b for a in U_list for b in U_list}
    
    max_u = max(U_list)
    if max_u == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    ratio = len(diff_set) / len(sum_set)
    score = 1 + math.log(ratio) / log_denom
    return score

def _generate_primes(limit):
    """Generate primes up to limit."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def _greedy_prime_selection(primes, max_elements=100):
    """
    Greedy selection of primes to form U = {0, p1, p1+p2, p1+p2+p3, ...}
    maximizing the score incrementally.
    """
    U = [0]
    used_primes = set()
    
    for _ in range(max_elements):
        best_prime = None
        best_score = _score_set(U)
        
        for p in primes:
            if p in used_primes:
                continue
            new_U = U + [U[-1] + p]
            new_score = _score_set(new_U)
            if new_score > best_score:
                best_score = new_score
                best_prime = p
        
        if best_prime is None:
            break
        
        U.append(U[-1] + best_prime)
        used_primes.add(best_prime)
    
    return U

def _local_search(U, primes, max_iterations=5000, neighbor_range=10):
    """
    Local search on the gap sequence derived from U.
    Gaps are U[i+1] - U[i].
    """
    current_gaps = [U[i+1] - U[i] for i in range(len(U)-1)]
    current_U = [0]
    for g in current_gaps:
        current_U.append(current_U[-1] + g)
    current_score = _score_set(current_U)
    
    for _ in range(max_iterations):
        improved = False
        
        # Try modifying each gap
        for i in range(len(current_gaps)):
            old_g = current_gaps[i]
            best_candidate = old_g
            best_score = current_score
            
            # Try nearby values and primes
            candidates = set()
            for delta in range(-neighbor_range, neighbor_range + 1):
                new_g = old_g + delta
                if new_g > 0:
                    candidates.add(new_g)
            
            # Add nearby primes from the pool
            for p in primes:
                if abs(p - old_g) < neighbor_range * 5:
                    candidates.add(p)
            
            for new_g in candidates:
                if new_g == old_g or new_g <= 0:
                    continue
                
                new_gaps = current_gaps[:]
                new_gaps[i] = new_g
                new_U = [0]
                for g in new_gaps:
                    new_U.append(new_U[-1] + g)
                
                if len(set(new_U)) != len(new_U):
                    continue
                
                new_score = _score_set(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = new_g
            
            if best_candidate != old_g:
                current_gaps[i] = best_candidate
                current_score = best_score
                improved = True
                current_U = [0]
                for g in current_gaps:
                    current_U.append(current_U[-1] + g)
        
        # Try adding a new gap
        if len(current_gaps) < 150:
            # Try a few random primes or the smallest unused ones
            # To save time, we only try a subset of primes for addition
            for p in primes[:20]:
                if p in current_gaps:
                    continue
                new_gaps = current_gaps + [p]
                new_U = current_U + [current_U[-1] + p]
                new_score = _score_set(new_U)
                if new_score > current_score:
                    current_gaps = new_gaps
                    current_U = new_U
                    current_score = new_score
                    improved = True
        
        # Try removing last gap
        if len(current_gaps) > 1:
            new_gaps = current_gaps[:-1]
            new_U = current_U[:-1]
            new_score = _score_set(new_U)
            if new_score > current_score:
                current_gaps = new_gaps
                current_U = new_U
                current_score = new_score
                improved = True
        
        if not improved:
            break
    
    return current_U, current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the sumset/difference-set exponent lower bound.
    
    Uses greedy prime selection followed by local search optimization.
    """
    # Generate a pool of primes
    # Larger pool allows for better local search moves
    primes = _generate_primes(1000)
    
    # Greedy selection to get a good starting point
    # Start with a slightly larger set to give local search more room
    greedy_U = _greedy_prime_selection(primes, max_elements=50)
    
    # Local search to refine
    # Increase iterations and neighbor range
    best_U, best_score = _local_search(greedy_U, primes, max_iterations=3000, neighbor_range=10)
    
    return best_U
# EVOLVE-BLOCK-END
