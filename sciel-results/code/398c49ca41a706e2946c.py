"""Rewritten program: maximize pairwise-disjoint infinite residue classes with distinct moduli in [2,128]."""
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _are_disjoint(r1, m1, r2, m2):
    """Two residue classes r1 mod m1 and r2 mod m2 are disjoint iff gcd(m1,m2) does not divide (r1-r2)."""
    g = _gcd(m1, m2)
    return (r1 - r2) % g != 0

def _validate(pairs):
    """Check all pairs are disjoint and moduli are distinct and in [2,128], residues canonical."""
    moduli = [p[1] for p in pairs]
    if len(set(moduli)) != len(moduli):
        return False
    for r, m in pairs:
        if not (2 <= m <= 128):
            return False
        if not (0 <= r < m):
            return False
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if not _are_disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _try_assign_residues(moduli, max_iters=500000, deadline=None):
    """
    Given a set of moduli, try to find residues r_i in [0, m_i) such that all pairs are disjoint.
    Uses backtracking with constraint propagation.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    # Precompute gcds
    gcds = [[_gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
    
    # For pair (i,j) with gcd g, the constraint is: (r_i - r_j) % g != 0
    # i.e., r_i % g != r_j % g
    
    # Order moduli: put those with more constraints (higher gcds with others) first
    constraint_count = [sum(1 for j in range(n) if j != i and gcds[i][j] > 1) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-constraint_count[i], -moduli[i]))
    
    residues = [None] * n
    iters = 0
    
    def backtrack(idx):
        nonlocal iters
        if idx == len(order):
            return True
        if deadline and time.time() > deadline:
            return False
        iters += 1
        if iters > max_iters:
            return False
        
        i = order[idx]
        m = moduli[i]
        
        # Determine valid residues for i
        # For each already-assigned j, we need r_i % gcds[i][j] != residues[j] % gcds[i][j]
        # We can compute the set of forbidden residues mod m
        
        # For efficiency, build forbidden set
        forbidden = set()
        for pos in range(idx):
            j = order[pos]
            g = gcds[i][j]
            if g <= 1:
                continue
            rj = residues[j]
            # r_i mod g != rj mod g
            # So r_i must NOT be in {rj mod g + k*g : k} intersected with [0, m)
            target = rj % g
            # r_i values in [0,m) where r_i % g == target are forbidden
            start = target
            while start < m:
                forbidden.add(start)
                start += g
        
        # Try each residue in [0, m)
        for r in range(m):
            if r in forbidden:
                continue
            residues[i] = r
            if backtrack(idx + 1):
                return True
        residues[i] = None
        return False
    
    if backtrack(0):
        return [[residues[order[i]], moduli[order[i]]] for i in range(n)]
    return None

def _greedy_build(moduli_candidates, deadline):
    """
    Greedily add moduli one at a time, trying to find a valid residue assignment.
    """
    result = []
    result_moduli = set()
    
    # Sort candidates by some heuristic: prefer moduli that share gcd>1 with few others
    # Actually, we want to build a set where residues can be assigned.
    # Let's try multiple orderings.
    
    for ordering in ['asc', 'desc', 'prime_first', 'composite_first']:
        if deadline and time.time() > deadline:
            break
        candidates = list(moduli_candidates)
        if ordering == 'asc':
            candidates.sort()
        elif ordering == 'desc':
            candidates.sort(reverse=True)
        elif ordering == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for p in range(2, int(n**0.5) + 1):
                    if n % p == 0:
                        return False
                return True
            candidates.sort(key=lambda m: (not is_prime(m), m))
        elif ordering == 'composite_first':
            def is_prime(n):
                if n < 2:
                    return False
                for p in range(2, int(n**0.5) + 1):
                    if n % p == 0:
                        return False
                return True
            candidates.sort(key=lambda m: (is_prime(m), m))
        
        current = []
        current_moduli = set()
        for m in candidates:
            if deadline and time.time() > deadline:
                break
            if m in current_moduli:
                continue
            trial = current + [[None, m]]
            # Try to assign residues
            moduli_list = [p[1] for p in trial]
            res = _try_assign_residues(moduli_list, max_iters=50000, deadline=deadline)
            if res is not None:
                current = res
                current_moduli.add(m)
        
        if len(current) > len(result):
            result = current
    
    return result

def _local_search(pairs, deadline, max_rounds=100):
    """
    Try to improve by swapping out a modulus for another.
    """
    if not pairs:
        return pairs
    
    used_moduli = set(p[1] for p in pairs)
    all_moduli = set(range(2, 129))
    available = all_moduli - used_moduli
    
    best = list(pairs)
    best_len = len(best)
    
    for _ in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        improved = False
        for m_remove in list(used_moduli):
            if deadline and time.time() > deadline:
                break
            # Remove this modulus
            remaining = [p for p in best if p[1] != m_remove]
            remaining_moduli = set(p[1] for p in remaining)
            
            for m_add in available:
                if m_add in remaining_moduli:
                    continue
                trial = remaining + [[None, m_add]]
                moduli_list = [p[1] for p in trial]
                res = _try_assign_residues(moduli_list, max_iters=30000, deadline=deadline)
                if res is not None and len(res) > best_len:
                    best = res
                    best_len = len(best)
                    used_moduli = set(p[1] for p in best)
                    available = all_moduli - used_moduli
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    all_moduli = list(range(2, 129))
    
    # Start with greedy
    result = _greedy_build(all_moduli, deadline)
    
    if not result:
        # Fallback: power of 2 heuristic
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    # Try local search to improve
    result = _local_search(result, deadline)
    
    # Try random restarts with different subsets
    rng = random.Random(42)
    best_result = result
    best_len = len(best_result)
    
    while time.time() < deadline - 5:
        # Random subset of moduli
        subset_size = min(len(all_moduli), best_len + rng.randint(0, 10))
        subset = rng.sample(all_moduli, subset_size)
        res = _greedy_build(subset, deadline)
        if res and len(res) > best_len:
            best_result = res
            best_len = len(res)
    
    # Final local search on best
    best_result = _local_search(best_result, deadline)
    
    # Validate
    if not _validate(best_result):
        # Fallback
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return best_result
# EVOLVE-BLOCK-END
