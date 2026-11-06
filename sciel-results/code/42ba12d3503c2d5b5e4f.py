import math
import random
import time
from fractions import Fraction
import sys

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)

def _coverage_length(assign):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    MAX_CHECK = 100000
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += Fraction(1, _PRIMES[idx])
    return c <= Fraction(1, 1)

def _seed_solution():
    """The seed: i-th prime from 13 upward covers integer i."""
    primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    return [[p, (i + 1) % p] for i, p in enumerate(primes)]

def _solution_to_assign(sol):
    """Convert solution [[p, r], ...] to internal assign list."""
    assign = [0] * NUM_PRIMES
    for p, r in sol:
        if p in _PRIMES:
            idx = _PRIMES.index(p)
            assign[idx] = r
    return assign

def _assign_to_solution(assign):
    """Convert internal assign list to solution format."""
    return [[_PRIMES[i], r] for i, r in enumerate(assign) if r != 0]

def _simulated_annealing(assign, time_limit):
    """
    Simulated Annealing with incremental coverage updates.
    Escapes local minima by accepting temporary coverage drops.
    Returns improved assign list.
    """
    start = time.time()
    random.seed(42)  # Reproducibility
    
    # Start with current best
    current = assign[:]
    current_L = _coverage_length(current)
    best = current[:]
    best_L = current_L
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    max_iters = 200000
    
    # Precompute prime indices for quick access
    prime_indices = list(range(NUM_PRIMES))
    
    # Track which primes are currently used
    used_primes = set(i for i, r in enumerate(current) if r != 0)
    
    for iteration in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Update temperature (geometric cooling)
        T = T_start * (T_end / T_start) ** (iteration / max_iters)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Change residue of a used prime
            if not used_primes:
                continue
            idx = random.choice(list(used_primes))
            p = _PRIMES[idx]
            old_r = current[idx]
            new_r = random.randint(1, p)
            if new_r == old_r:
                continue
            current[idx] = new_r
            
        elif move_type < 0.7:
            # Add a new prime
            available = [i for i in prime_indices if current[i] == 0]
            if not available:
                continue
            idx = random.choice(available)
            p = _PRIMES[idx]
            current[idx] = random.randint(1, p)
            used_primes.add(idx)
            
        else:
            # Remove a prime
            if not used_primes:
                continue
            idx = random.choice(list(used_primes))
            current[idx] = 0
            used_primes.discard(idx)
        
        # Check validity (cost constraint)
        if not _is_valid(current):
            # Revert
            if move_type < 0.4:
                pass  # residue change doesn't affect cost
            elif move_type < 0.7:
                current[idx] = 0
                used_primes.discard(idx)
            else:
                pass  # removal doesn't affect cost
            continue
        
        # Evaluate new coverage
        new_L = _coverage_length(current)
        
        # Acceptance criterion
        if new_L >= current_L:
            # Always accept improvement or equal
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best = current[:]
        else:
            # Accept with probability exp(-(delta)/T)
            delta = current_L - new_L
            if delta > 0:
                prob = math.exp(-delta / max(T, 1e-10))
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    if move_type < 0.4:
                        current[idx] = old_r
                    elif move_type < 0.7:
                        current[idx] = 0
                        used_primes.discard(idx)
                    else:
                        pass
    
    return best, best_L

def _local_search(assign, time_limit):
    """
    Local search: try all single-residue changes and pick the best improvement.
    Repeat until no improvement or time limit.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    best = current[:]
    best_L = current_L
    
    used_primes = set(i for i, r in enumerate(current) if r != 0)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        best_delta = 0
        best_move = None
        
        # Try changing residue of each used prime
        for idx in list(used_primes):
            if time.time() - start > time_limit:
                break
            p = _PRIMES[idx]
            old_r = current[idx]
            
            for new_r in range(1, p + 1):
                if new_r == old_r:
                    continue
                current[idx] = new_r
                new_L = _coverage_length(current)
                current[idx] = old_r
                
                delta = new_L - current_L
                if delta > best_delta:
                    best_delta = delta
                    best_move = (idx, new_r)
        
        # Try adding a new prime
        available = [i for i in range(NUM_PRIMES) if current[i] == 0]
        for idx in available:
            if time.time() - start > time_limit:
                break
            p = _PRIMES[idx]
            
            # Check if we can afford this prime
            cost = Fraction(1, p)
            current_cost = Fraction(0)
            for i, r in enumerate(current):
                if r != 0:
                    current_cost += Fraction(1, _PRIMES[i])
            
            if current_cost + cost > Fraction(1, 1):
                continue
            
            for new_r in range(1, p + 1):
                current[idx] = new_r
                new_L = _coverage_length(current)
                current[idx] = 0
                
                delta = new_L - current_L
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', idx, new_r)
        
        # Try removing a prime
        for idx in list(used_primes):
            if time.time() - start > time_limit:
                break
            old_r = current[idx]
            current[idx] = 0
            new_L = _coverage_length(current)
            current[idx] = old_r
            
            delta = new_L - current_L
            if delta > best_delta:
                best_delta = delta
                best_move = ('remove', idx)
        
        if best_move is not None and best_delta > 0:
            if best_move[0] == 'add':
                current[best_move[1]] = best_move[2]
                used_primes.add(best_move[1])
            elif best_move[0] == 'remove':
                current[best_move[1]] = 0
                used_primes.discard(best_move[1])
            else:
                current[best_move[0]] = best_move[1]
            
            current_L += best_delta
            improved = True
            
            if current_L > best_L:
                best_L = current_L
                best = current[:]
    
    return best, best_L

def construct():
    start = time.time()
    deadline = start + 145  # safety margin before 150s kill
    
    # Start with seed solution
    seed_sol = _seed_solution()
    best_assign = _solution_to_assign(seed_sol)
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        # Fallback to empty
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Phase 1: Local search to quickly improve
    time_left = deadline - time.time() - 5
    if time_left > 1:
        ls_time = min(time_left * 0.3, 30)
        best_assign, best_L = _local_search(best_assign, ls_time)
    
    # Phase 2: Simulated annealing to escape local minima
    time_left = deadline - time.time() - 5
    if time_left > 1:
        sa_time = min(time_left * 0.7, 90)
        sa_assign, sa_L = _simulated_annealing(best_assign, sa_time)
        if sa_L > best_L:
            best_L = sa_L
            best_assign = sa_assign
    
    # Phase 3: Final local search polish
    time_left = deadline - time.time() - 2
    if time_left > 1:
        best_assign, best_L = _local_search(best_assign, time_left)
    
    # Final validation
    if best_assign is not None and _is_valid(best_assign):
        return _assign_to_solution(best_assign)
    else:
        # Fallback to seed
        return _seed_solution()

if __name__ == '__main__':
    result = construct()
    print(f"Found {len(result)} classes")
    # Verify
    assign = _solution_to_assign(result)
    L = _coverage_length(assign)
    print(f"Coverage length: {L}")
    print(f"Valid: {_is_valid(assign)}")
