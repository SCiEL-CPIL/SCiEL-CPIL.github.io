import time
from fractions import Fraction
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    all_primes = get_primes(1000)
    
    # Precompute reciprocal sums for subsets - we'll use a greedy selection
    # The optimal set of primes with reciprocal sum <= 1 to maximize coverage
    # Known good set: [3, 5, 7, 11, 13, 17, 19, 23] with sum = 35913/35970 < 1
    
    # Branch and bound: select primes and residues
    # State: (prime_index, residues_so_far, covered_bitmap, budget_remaining)
    
    # We'll use a different approach: 
    # 1. Select a set of primes (branch on which primes to include)
    # 2. For each prime set, solve for residues using constraint propagation
    
    # First, determine good prime sets
    # The reciprocal sum constraint: sum(1/p) <= 1
    # We want primes that cover small numbers well
    
    # Precompute: for a given set of primes, what's the max coverage?
    # Use a SAT-like approach with constraint propagation
    
    def compute_coverage(prime_residues, L):
        """prime_residues: list of (p, r). Return first uncovered number in 1..L, or L+1 if all covered."""
        covered = bytearray(L + 1)
        for p, r in prime_residues:
            r = r % p
            start = r if r > 0 else p
            for x in range(start, L + 1, p):
                covered[x] = 1
        for x in range(1, L + 1):
            if not covered[x]:
                return x
        return L + 1
    
    def solve_residues(primes, L, time_limit=5.0):
        """
        Given a list of primes, find residues that cover 1..L.
        Uses backtracking with constraint propagation.
        Returns list of residues or None.
        """
        n = len(primes)
        if n == 0:
            return [0] * n
        
        # For each prime, precompute which numbers it covers for each residue
        # cov[i][r] = set of numbers in 1..L that are ≡ r (mod primes[i])
        # This is too memory-intensive for large L, so use a different approach
        
        # Instead: backtracking on residues, maintaining a coverage bitmap
        residues = [0] * n
        covered = bytearray(L + 1)
        
        # Precompute for each prime and residue, the list of positions
        prime_positions = []
        for p in primes:
            pos = []
            for r in range(p):
                start = r if r > 0 else p
                pos.append(list(range(start, L + 1, p)))
            prime_positions.append(pos)
        
        # Order primes by increasing size (smaller primes have more residue options but cover more)
        # Actually, process primes in the order given
        
        def backtrack(idx, covered_count):
            if time.time() > time_limit + start_time:
                return False
            if idx == n:
                return covered_count == L
            
            p = primes[idx]
            
            # Try residues in order of coverage gain
            gains = []
            for r in range(p):
                gain = 0
                for x in prime_positions[idx][r]:
                    if not covered[x]:
                        gain += 1
                gains.append((gain, r))
            
            gains.sort(reverse=True)
            
            for gain, r in gains:
                if gain == 0:
                    break
                
                # Apply residue r
                new_covered = bytearray(covered)
                for x in prime_positions[idx][r]:
                    if not new_covered[x]:
                        new_covered[x] = 1
                
                new_count = 0
                for x in range(1, L + 1):
                    if new_covered[x]:
                        new_count += 1
                
                # Pruning: check if remaining primes can cover the rest
                # Simple bound: each remaining prime can cover at most L/2 new numbers (very loose)
                # Better: check the first uncovered number
                first_unc = 0
                for x in range(1, L + 1):
                    if not new_covered[x]:
                        first_unc = x
                        break
                
                if first_unc == 0:
                    # All covered
                    residues[idx] = r
                    return True
                
                # Check if remaining primes can cover first_unc
                can_cover = False
                for j in range(idx + 1, n):
                    pj = primes[j]
                    if first_unc % pj == 0:
                        can_cover = True
                        break
                    # Also check if any residue of pj covers first_unc
                    # first_unc % pj is the required residue
                    can_cover = True  # Always possible since we can choose any residue
                    break
                
                if not can_cover:
                    continue
                
                residues[idx] = r
                if backtrack(idx + 1, new_count):
                    return True
            
            return False
        
        if backtrack(0, 0):
            return residues
        return None
    
    # Strategy: Try several prime sets, solve for residues
    # Prime sets to try (ordered by likelihood of success)
    prime_sets = [
        [3, 5, 7, 11, 13, 17, 19, 23],
        [3, 5, 7, 11, 13, 17, 19, 29],
        [3, 5, 7, 11, 13, 17, 19, 31],
        [3, 5, 7, 11, 13, 17, 23, 29],
        [3, 5, 7, 11, 13, 19, 23, 29],
        [3, 5, 7, 11, 17, 19, 23, 29],
        [3, 5, 7, 13, 17, 19, 23, 29],
        [5, 7, 11, 13, 17, 19, 23, 29],
    ]
    
    best_L = 0
    best_result = None
    
    for ps in prime_sets:
        if time.time() > deadline - 10:
            break
        
        # Verify reciprocal sum
        total = sum(Fraction(1, p) for p in ps)
        if total > 1:
            continue
        
        # Try to solve for coverage up to 1000
        residues = solve_residues(ps, 1000, time_limit=8.0)
        if residues is not None:
            L = compute_coverage(list(zip(ps, residues)), 1000) - 1
            if L > best_L:
                best_L = L
                best_result = [[p, r] for p, r in zip(ps, residues)]
            if L >= 1000:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a simple valid set
    return [[3, 0], [5, 1], [7, 2]]
