import time
import random
from math import gcd

def _is_prime(n):
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

def _count_available(m, assigned):
    """Count how many residues r in [0,m) are not forbidden by assigned classes."""
    forbidden = [False] * m
    for r_j, m_j in assigned:
        g = gcd(m, m_j)
        if g == 1:
            # No constraint when gcd=1... wait, that's wrong.
            # If gcd=1, then for any r_j there exists some r in [0,m) that conflicts.
            # Actually: r conflicts with (r_j, m_j) iff r ≡ r_j (mod g).
            # If g=1, then ALL r in [0,m) satisfy r ≡ 0 (mod 1) = r_j % 1 = 0.
            # So if g=1, all residues are forbidden? No!
            # Two classes (r1,m1) and (r2,m2) are disjoint iff gcd(m1,m2) does NOT divide (r1-r2).
            # So r is forbidden if gcd(m, m_j) divides (r - r_j), i.e., r ≡ r_j (mod g).
            # If g=1, then r ≡ r_j (mod 1) is always true, so ALL r are forbidden.
            # That means if any assigned class has gcd=1 with m, then NO residue works for m!
            return 0
        start = r_j % g
        for r_val in range(start, m, g):
            forbidden[r_val] = True
    count = 0
    for r in range(m):
        if not forbidden[r]:
            count += 1
    return count

def _find_available_residues(m, assigned):
    """Return list of available residues for modulus m given assigned classes."""
    forbidden = [False] * m
    for r_j, m_j in assigned:
        g = gcd(m, m_j)
        if g == 1:
            return []
        start = r_j % g
        for r_val in range(start, m, g):
            forbidden[r_val] = True
    available = []
    for r in range(m):
        if not forbidden[r]:
            available.append(r)
    return available

def _backtrack_csp(unassigned, assigned, depth=0):
    """
    Backtracking CSP: select next modulus using most-constrained-first heuristic.
    Returns a solution (list of (r,m) pairs) or None if stuck.
    """
    if not unassigned:
        return list(assigned)
    
    # Most-constrained-first: pick modulus with fewest available residues
    best_idx = -1
    best_count = float('inf')
    best_residues = []
    
    for i, m in enumerate(unassigned):
        residues = _find_available_residues(m, assigned)
        count = len(residues)
        if count == 0:
            return None  # Dead end
        if count < best_count:
            best_count = count
            best_idx = i
            best_residues = residues
            if count == 1:
                break  # Can't do better than 1
    
    m = unassigned[best_idx]
    remaining = unassigned[:best_idx] + unassigned[best_idx+1:]
    
    # Try each available residue
    for r in best_residues:
        new_assigned = assigned + [(r, m)]
        result = _backtrack_csp(remaining, new_assigned, depth+1)
        if result is not None:
            return result
    
    return None

def _greedy_prime_first(moduli):
    """Greedy with primes first, then composites by ascending order."""
    primes = [m for m in moduli if _is_prime(m)]
    composites = [m for m in moduli if not _is_prime(m)]
    order = primes + sorted(composites)
    
    solution = []
    for m in order:
        residues = _find_available_residues(m, solution)
        if residues:
            # Pick residue that minimizes future conflicts: choose r that shares
            # largest gcd with remaining moduli to maximize future options? 
            # Or just pick the first available.
            solution.append((residues[0], m))
    return solution

def _local_search_improve(solution, moduli_set, time_budget):
    """Try to add more classes by removing some and re-adding in different order."""
    if time_budget <= 0:
        return solution
    
    best_sol = list(solution)
    best_size = len(solution)
    
    remaining = [m for m in moduli_set if m not in {mm for _, mm in best_sol}]
    if not remaining:
        return best_sol
    
    # Try adding one remaining modulus at a time
    for m in remaining:
        residues = _find_available_residues(m, best_sol)
        if residues:
            best_sol.append((residues[0], m))
            best_size += 1
    
    return best_sol

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_mod = 128
    all_moduli = list(range(2, max_mod + 1))
    
    best_result = []
    best_size = 0
    
    # Phase 1: Prime-first greedy (fast, good baseline)
    greedy_sol = _greedy_prime_first(all_moduli)
    if len(greedy_sol) > best_size:
        best_size = len(greedy_sol)
        best_result = greedy_sol
    
    # Phase 2: Backtracking CSP with most-constrained-first
    # Start with primes assigned (they have gcd=1 with everything, so they're most constraining)
    # Actually, primes have gcd=1 with all non-multiples, which means they forbid ALL residues
    # for those moduli. So we should NOT start with primes in the CSP.
    # Instead, start with highly composite numbers or just do full backtracking.
    
    # Try backtracking from scratch with a time limit
    random.seed(42)
    
    # Strategy: Try backtracking with different initial seeds
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # For first attempt, try pure backtracking
        # For subsequent attempts, try with some random pre-assignment
        if attempt == 0:
            initial = []
            unassigned = list(all_moduli)
        else:
            # Randomly pre-assign a few moduli
            k = random.randint(0, 5)
            pre_mods = random.sample(all_moduli, k)
            initial = []
            for m in pre_mods:
                residues = _find_available_residues(m, initial)
                if residues:
                    initial.append((random.choice(residues), m))
            unassigned = [m for m in all_moduli if m not in {mm for _, mm in initial}]
        
        # Run backtracking with a node limit to avoid infinite loops
        node_limit = 500000
        result = _backtrack_with_limit(unassigned, initial, node_limit)
        
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Phase 3: Local improvement - try to add remaining moduli
    if time.time() < deadline - 5:
        improved = _local_search_improve(best_result, set(all_moduli), deadline - time.time())
        if len(improved) > best_size:
            best_size = len(improved)
            best_result = improved
    
    # Convert to required format
    final_result = [[r, m] for r, m in best_result]
    
    if not final_result:
        final_result = [[1, 2]]
    
    return final_result

def _backtrack_with_limit(unassigned, assigned, node_limit):
    """Backtracking with a node count limit to prevent timeout."""
    count = [0]
    
    def bt(remaining, current):
        if count[0] >= node_limit:
            return None
        count[0] += 1
        
        if not remaining:
            return list(current)
        
        # Most-constrained-first
        best_idx = -1
        best_count = float('inf')
        best_residues = []
        
        for i, m in enumerate(remaining):
            residues = _find_available_residues(m, current)
            cnt = len(residues)
            if cnt == 0:
                return None
            if cnt < best_count:
                best_count = cnt
                best_idx = i
                best_residues = residues
                if cnt == 1:
                    break
        
        m = remaining[best_idx]
        rest = remaining[:best_idx] + remaining[best_idx+1:]
        
        for r in best_residues:
            result = bt(rest, current + [(r, m)])
            if result is not None:
                return result
        
        return None
    
    return bt(unassigned, assigned)
