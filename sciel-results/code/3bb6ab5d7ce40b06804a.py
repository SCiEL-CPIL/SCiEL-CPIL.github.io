"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def construct():
    # Generate all primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # We want to maximize L such that every integer 1..L is covered.
    # Budget: sum of 1/p <= 1
    # Target: L >= 999 ideally, but let's aim for max possible.
    # 
    # Strategy: Use a greedy/backtracking approach with bitsets.
    # For each prime p, choosing residue r covers {r, r+p, r+2p, ...} within [1, L_target].
    # 
    # Key insight: To cover [1, L] densely, we should use small primes with residues that
    # fill gaps. The classic covering systems use primes 2,3,5,7,11,... with specific residues.
    #
    # Let's try to build a covering system that covers [1, 1000] with sum 1/p <= 1.
    #
    # Known result: The Erdős-Ruzsa conjecture states that for any set of distinct primes
    # with sum 1/p <= 1, the covered integers have density at most... actually the conjecture
    # is about the covering problem. Let me think differently.
    #
    # We need to find residues r_p for each prime p in our set such that every integer
    # in [1, L] is congruent to r_p mod p for at least one chosen prime p.
    #
    # Greedy approach: Start with L_target = 1000. Use a bitset for uncovered integers.
    # For each prime (sorted by some heuristic), choose the residue that covers the most
    # currently uncovered integers, while maintaining budget feasibility.
    
    L_target = 1000
    
    # Bitset representation: bit i represents integer i+1 (1-indexed)
    # Use Python integers as bitsets
    def make_bitset(L):
        return (1 << L) - 1  # bits 0 to L-1 set, representing integers 1 to L
    
    # For a prime p and residue r, the covered integers in [1, L] are:
    # r, r+p, r+2p, ... where r is in [0, p-1], but we only count those in [1, L]
    # If r == 0, covered integers are p, 2p, 3p, ...
    # If r > 0, covered integers are r, r+p, r+2p, ...
    
    def covered_bits(p, r, L):
        """Return bitmask of integers in [1, L] congruent to r mod p."""
        bits = 0
        if r == 0:
            start = p
        else:
            start = r
        if start > L:
            return 0
        for x in range(start, L + 1, p):
            bits |= (1 << (x - 1))
        return bits
    
    # Precompute for all primes and all residues the bitmask
    # This might be expensive, so let's be smart.
    
    # Actually, let's use a different approach. We'll use a greedy algorithm:
    # 1. Start with all integers 1..L_target uncovered.
    # 2. For each step, try each available prime (not yet used) and each residue.
    # 3. Pick the (p, r) that covers the most uncovered integers, subject to budget.
    # 4. Repeat until no more primes can be added or budget is exhausted.
    
    # Budget tracking with exact rationals
    budget = Fraction(1, 1)
    used_primes = set()
    chosen = []  # list of [p, r]
    
    # Current uncovered bitmask
    uncovered = make_bitset(L_target)
    
    # For efficiency, precompute coverage counts
    # For each prime p, for each residue r, count how many integers in [1, L_target] are covered
    # and which ones are currently uncovered.
    
    # Let's do a smarter greedy: at each step, for each candidate (p, r),
    # compute the number of newly covered integers (intersection of covered_bits(p,r) with uncovered).
    # Pick the one with maximum new coverage, as long as adding 1/p doesn't exceed budget.
    
    # Sort primes by size - smaller primes cover more integers per residue on average
    # Actually, for a given prime p, each residue covers about L/p integers.
    # Smaller p means more integers per residue, so we should prioritize small primes.
    
    # But we need to be careful: using a small prime uses up budget (1/p is large)
    # but covers many integers.
    
    # Let's try a priority-based greedy:
    # At each step, consider all unused primes p where Fraction(1, p) <= remaining_budget.
    # For each such p, find the residue r that maximizes newly covered integers.
    # Among all (p, r) candidates, pick the one with the best ratio or just max coverage.
    
    # To make this efficient, let's limit the number of iterations.
    
    max_iterations = 50
    for _ in range(max_iterations):
        best_p = None
        best_r = None
        best_new_coverage = 0
        best_bits = 0
        
        remaining = budget - sum(Fraction(1, p) for p, _ in chosen)
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if cost > remaining:
                continue
            
            # For this prime, try all residues and find the best one
            for r in range(p):
                bits = covered_bits(p, r, L_target)
                new_cov = bits & uncovered
                count = bin(new_cov).count('1')
                if count > best_new_coverage:
                    best_new_coverage = count
                    best_p = p
                    best_r = r
                    best_bits = new_cov
        
        if best_p is None or best_new_coverage == 0:
            break
        
        # Add this prime-residue pair
        chosen.append([best_p, best_r])
        used_primes.add(best_p)
        uncovered &= ~best_bits
    
    # Now check what L we actually cover
    # Find the largest L such that all integers 1..L are covered
    # i.e., the first gap in the covered set
    covered = make_bitset(L_target) & ~uncovered
    
    # Find the first zero bit from the left (starting from bit 0)
    # Actually, we need the largest L such that bits 0..L-1 are all set in covered.
    L = 0
    for i in range(L_target):
        if covered & (1 << i):
            L = i + 1
        else:
            break
    
    # But wait, we might have done poorly with greedy. Let's also try a known-good construction.
    # 
    # Classic covering system: 
    # 2 | 0, 3 | 1, 5 | 4, 7 | 2, 11 | 5, 13 | 7, 17 | 10, 19 | 12, 23 | 14, 29 | 18, 31 | 20, 37 | 24, 41 | 28, 43 | 30, 47 | 34, 53 | 38, 59 | 42, 61 | 46, 67 | 50, 71 | 54, 73 | 58, 79 | 62, 83 | 66, 89 | 70, 97 | 76
    # Sum of reciprocals: 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + ... this is too large.
    
    # Let's check the sum for the greedy result
    total_cost = sum(Fraction(1, p) for p, _ in chosen)
    
    # If greedy didn't achieve L >= 999, let's try a more targeted approach.
    # We want to cover [1, 1000]. Let's use a constraint programming approach with backtracking.
    
    if L < 999:
        # Try a more systematic approach
        # We'll use the primes and try to assign residues to cover [1, 1000]
        # with the constraint that sum 1/p <= 1.
        
        # Start with a subset of small primes that sum to <= 1
        # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 = ?
        # Let's compute: 0.5 + 0.333 + 0.2 + 0.1429 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0435 = 1.4987 > 1
        # So we need to be selective.
        
        # Try: 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 = 0.5+0.333+0.2+0.1429+0.0909+0.0769+0.0588 = 1.4025 > 1
        # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 = 0.333+0.2+0.1429+0.0909+0.0769+0.0588+0.0526+0.0435+0.0345 = 1.0331 > 1
        # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 = 0.333+0.2+0.1429+0.0909+0.0769+0.0588+0.0526+0.0435 = 1.0026 > 1 (just barely)
        # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/29 = 0.333+0.2+0.1429+0.0909+0.0769+0.0588+0.0526+0.0345 = 0.9896 <= 1
        
        # Let's use these primes and try to find residues that cover [1, 1000]
        # This is a hard constraint satisfaction problem. Let's use a simple backtracking.
        
        candidate_primes = [3, 5, 7, 11, 13, 17, 19, 29]
        cost = sum(Fraction(1, p) for p in candidate_primes)
        assert cost <= 1, f"Cost {cost} exceeds 1"
        
        # Backtracking search for residues
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(idx, uncovered_mask, current_choices):
            if idx == len(candidate_primes):
                return None  # All primes assigned, check if fully covered
            
            p = candidate_primes[idx]
            
            # Try all residues for this prime
            # Heuristic: try residues that cover more uncovered integers first
            residues = list(range(p))
            residues.sort(key=lambda r: -bin(covered_bits(p, r, L_target) & uncovered_mask).count('1'))
            
            for r in residues:
                bits = covered_bits(p, r, L_target)
                new_uncovered = uncovered_mask & ~bits
                
                # Pruning: if new_uncovered has fewer bits than remaining primes can cover, skip
                # Each remaining prime can cover at most L_target / min_remaining_prime integers
                # This is a weak bound, but let's try.
                
                result = backtrack(idx + 1, new_uncovered, current_choices + [[p, r]])
                if result is not None:
                    return result
            
            return None
        
        # This backtracking might be too slow. Let's use a simpler approach:
        # Greedy with backtracking only when stuck.
        
        # Actually, let's just return the greedy result for now and see how it does.
        # The greedy approach might already give a good L.
    
    # Verify the result
    final_cost = sum(Fraction(1, p) for p, _ in chosen)
    assert final_cost <= 1, f"Final cost {final_cost} exceeds 1"
    for p, r in chosen:
        assert p <= 1000 and 0 <= r < p, f"Invalid class [{p}, {r}]"
    
    # Check that all primes are distinct
    prime_list = [p for p, _ in chosen]
    assert len(prime_list) == len(set(prime_list)), "Duplicate primes"
    
    return chosen
# EVOLVE-BLOCK-END
