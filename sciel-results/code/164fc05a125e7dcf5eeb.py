"""Rewritten program: CSP-style backtracking with heuristic pruning for prime residue covering."""
import sys
from fractions import Fraction
from math import isqrt
import time

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES_1000 = _primes_up_to(1000)
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES_1000)}

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    # We will try to find a solution for L starting from a high value downwards.
    # The seed covers ~900. We aim for 999.
    # We use a backtracking CSP solver with limited branching (beam search style) 
    # to keep it within time limits.
    
    L_target = 999
    # If we can't find 999, we might try lower, but the task says "Maximize".
    # The evaluator runs construct() once. We should try to get the best L possible.
    # Let's try a range of L values from high to low, but with a time limit per L.
    
    # Actually, a single run of a well-tuned solver for L=999 is best.
    # If it fails, we fall back to a greedy solution.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead and fallback
    
    # State variables for CSP
    # We represent the covered status as a bytearray for speed.
    # We need to backtrack, so we might need to copy the state or use a trail.
    # Given L=999, a bytearray of size 1000 is small. Copying it at each step is fine.
    
    def solve_csp(L):
        """
        Attempt to find a covering set for 1..L using backtracking CSP.
        Returns a list of [p, r] or None if not found within time/heuristic limits.
        """
        # Initial state
        covered = bytearray(L + 1)
        used_primes = set()
        current_budget = Fraction(0)
        solution = []
        
        # Precompute prime costs as Fractions for quick comparison
        prime_costs = {p: Fraction(1, p) for p in PRIMES_1000}
        
        # Recursive search function
        def backtrack(covered, used_primes, current_budget, solution):
            if time.time() - start_time > time_limit:
                return None
            
            # Find the smallest uncovered integer in 1..L
            next_n = -1
            for i in range(1, L + 1):
                if not covered[i]:
                    next_n = i
                    break
            
            if next_n == -1:
                return solution[:]  # All covered, success
            
            # Generate candidates: unused primes p such that current_budget + 1/p <= 1
            candidates = []
            for p in PRIMES_1000:
                if p not in used_primes:
                    cost = prime_costs[p]
                    if current_budget + cost <= 1:
                        # Calculate how many *new* integers in 1..L this prime covers
                        # if we set r = next_n % p
                        r = next_n % p
                        new_coverage = 0
                        # Check coverage for all m in 1..L
                        # Optimization: only check if m % p == r
                        for m in range(1, L + 1):
                            if not covered[m] and m % p == r:
                                new_coverage += 1
                        
                        # Heuristic: prefer primes that cover more new integers
                        # Tie-breaker: prefer smaller p? No, smaller p costs more.
                        # Prefer larger p? Larger p costs less.
                        # Let's sort by (new_coverage, -cost) or just new_coverage.
                        # Actually, covering more is better.
                        candidates.append((new_coverage, -cost, p, r))
            
            if not candidates:
                return None
            
            # Sort candidates: highest new_coverage first, then lowest cost (largest p)
            # Note: -cost means larger cost (smaller p) is "smaller" in sort if we want lowest cost?
            # We want to minimize cost, so we want larger p.
            # Let's sort by new_coverage descending, then p descending (to save budget).
            candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
            
            # Limit branching factor to avoid explosion
            # Try top 10 candidates
            for new_cov, neg_cost, p, r in candidates[:10]:
                # Apply assignment
                new_covered = bytearray(covered)
                for m in range(1, L + 1):
                    if m % p == r:
                        new_covered[m] = 1
                
                new_used = used_primes | {p}
                new_budget = current_budget + prime_costs[p]
                new_solution = solution + [[p, r]]
                
                # Forward checking: 
                # Are there any integers in 1..L that can *only* be covered by primes 
                # that are now unavailable or over budget?
                # This is expensive to check fully. 
                # Simple check: if new_budget > 1, skip (already handled).
                
                result = backtrack(new_covered, new_used, new_budget, new_solution)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(covered, used_primes, current_budget, solution)

    # Try to solve for L=999
    # We might need to adjust L if 999 is too hard.
    # Let's try 999 first.
    best_solution = None
    best_L = 0
    
    # Try a few L values in descending order
    for L in [999, 998, 997, 995, 990, 980, 970, 960, 950, 940, 930, 920, 910, 900]:
        if time.time() - start_time > time_limit:
            break
            
        # Reset time limit for this specific L attempt? 
        # No, global time limit is better.
        
        sol = solve_csp(L)
        if sol:
            best_solution = sol
            best_L = L
            break # Found a solution for this L, which is the max we tried. 
            # Wait, we want to MAXIMIZE L. 
            # If we found a solution for 999, that's great.
            # If we found 998, maybe 999 is possible?
            # The loop tries 999 first. If it fails, it tries 998.
            # So the first success is the highest L found.
            
    if best_solution is None:
        # Fallback to the original greedy method if CSP fails
        L_max = 100000
        covered = bytearray(L_max + 1)
        primes_desc_sorted = sorted(PRIMES_1000, reverse=True)
        num_primes = len(primes_desc_sorted)
        is_used = [False] * num_primes
        solution = []
        remaining_budget = Fraction(1, 1)
        
        for n in range(1, L_max + 1):
            if covered[n]:
                continue
            
            found = False
            for idx in range(num_primes):
                if not is_used[idx]:
                    p = primes_desc_sorted[idx]
                    is_used[idx] = True
                    r = n % p
                    solution.append([p, r])
                    
                    start = r if r > 0 else p
                    if start < 1:
                        start = 1
                    for m in range(start, L_max + 1, p):
                        covered[m] = 1
                    
                    remaining_budget -= Fraction(1, p)
                    found = True
                    break
            
            if not found:
                break
            if remaining_budget < 0:
                solution.pop()
                is_used[idx] = False
                break
        
        # Compute L for fallback
        L = 0
        for i in range(1, L_max + 1):
            if covered[i]:
                L = i
            else:
                break
        return solution

    return best_solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    # Compute covered length
    L_max = 100000
    covered = bytearray(L_max + 1)
    for p, r in sol:
        start = r if r > 0 else p
        for m in range(start, L_max + 1, p):
            covered[m] = 1
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"Covered length L: {L}")
    # Check budget
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
