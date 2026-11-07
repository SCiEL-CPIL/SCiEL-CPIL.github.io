"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def _factorize(n, primes):
    """Return list of exponents for each prime in primes."""
    exps = []
    for p in primes:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        exps.append(e)
    return exps

def _exponent_vectors(numbers, primes):
    """Get exponent vector for each number."""
    return [_factorize(n, primes) for n in numbers]

def _rank_of_vectors(vectors):
    """Compute rank of a list of integer vectors using fraction-based Gaussian elimination."""
    if not vectors:
        return 0
    vecs = [[Fraction(x) for x in v] for v in vectors]
    n = len(vecs)
    m = len(vecs[0]) if n > 0 else 0
    rank = 0
    for col in range(m):
        # Find pivot
        pivot = -1
        for row in range(rank, n):
            if vecs[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap
        vecs[rank], vecs[pivot] = vecs[pivot], vecs[rank]
        # Eliminate below
        for row in range(rank + 1, n):
            if vecs[row][col] != 0:
                factor = vecs[row][col] / vecs[rank][col]
                for j in range(col, m):
                    vecs[row][j] -= factor * vecs[rank][j]
        rank += 1
    return rank

def _is_independent(candidate_vec, existing_vecs, primes):
    """Check if adding candidate_vec to existing_vecs maintains linear independence."""
    vecs = existing_vecs + [candidate_vec]
    return _rank_of_vectors(vecs) == len(vecs)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s (or 90s) kill
    
    n_max = 40
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Precompute exponent vectors for all numbers 2..40
    all_numbers = list(range(2, n_max + 1))
    exp_vecs = {}
    for num in all_numbers:
        exp_vecs[num] = _factorize(num, primes)
    
    best_set = []
    best_size = 0
    
    # Strategy: Try greedy with different orderings
    # Ordering 1: by number of prime factors (fewer first - primes are best)
    def num_prime_factors(num):
        return sum(exp_vecs[num])
    
    orderings = [
        # Primes first (single prime factor)
        sorted(all_numbers, key=lambda x: (num_prime_factors(x), x)),
        # By value
        sorted(all_numbers),
        # Primes first, then by value
        sorted(all_numbers, key=lambda x: (0 if num_prime_factors(x) == 1 else 1, x)),
        # By number of distinct prime factors
        sorted(all_numbers, key=lambda x: (len([e for e in exp_vecs[x] if e > 0]), x)),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        current_set = []
        current_vecs = []
        for num in ordering:
            if time.time() > deadline:
                break
            vec = exp_vecs[num]
            # Quick check: is this vector independent of current set?
            if _is_independent(vec, current_vecs, primes):
                current_set.append(num)
                current_vecs.append(vec)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Try to improve: swap out elements to allow more additions
    if time.time() < deadline - 5:
        # Try local search: for each element in best_set, try swapping with an element not in set
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_set = list(best_set)
            current_vecs = [exp_vecs[n] for n in current_set]
            in_set = set(current_set)
            not_in_set = [n for n in all_numbers if n not in in_set]
            
            for i, old_num in enumerate(current_set):
                if time.time() > deadline - 5:
                    break
                for new_num in not_in_set:
                    if time.time() > deadline - 5:
                        break
                    # Try swapping
                    trial_vecs = current_vecs[:i] + [exp_vecs[new_num]] + current_vecs[i+1:]
                    if _rank_of_vectors(trial_vecs) == len(trial_vecs):
                        # Check if this allows us to add more
                        trial_set = current_set[:i] + [new_num] + current_set[i+1:]
                        trial_in_set = set(trial_set)
                        # Try to add more elements
                        trial_vecs2 = list(trial_vecs)
                        trial_set2 = list(trial_set)
                        for candidate in all_numbers:
                            if time.time() > deadline - 5:
                                break
                            if candidate not in trial_in_set:
                                if _is_independent(exp_vecs[candidate], trial_vecs2, primes):
                                    trial_set2.append(candidate)
                                    trial_vecs2.append(exp_vecs[candidate])
                        if len(trial_set2) > best_size:
                            best_size = len(trial_set2)
                            best_set = trial_set2
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    # Verify the best set
    if best_set:
        vecs = [exp_vecs[n] for n in best_set]
        r = _rank_of_vectors(vecs)
        if r != len(vecs):
            # Fallback: use primes
            best_set = [p for p in primes if p <= 40]
    
    return best_set
# EVOLVE-BLOCK-END
