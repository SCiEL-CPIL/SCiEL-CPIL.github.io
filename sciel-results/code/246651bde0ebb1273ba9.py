"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import sys

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
    
    L_target = 1000
    
    # Precompute coverage bitmasks for each prime and residue
    # covered_mask[p][r] = bitmask of integers in [1, L_target] congruent to r mod p
    covered_mask = {}
    for p in primes:
        covered_mask[p] = []
        for r in range(p):
            bits = 0
            if r == 0:
                start = p
            else:
                start = r
            if start <= L_target:
                for x in range(start, L_target + 1, p):
                    bits |= (1 << (x - 1))
            covered_mask[p].append(bits)
    
    # Full mask: all integers 1..L_target
    full_mask = (1 << L_target) - 1
    
    # Budget constraint: sum of 1/p <= 1
    # We want to maximize L such that [1, L] is fully covered.
    
    # Strategy: Backtracking with strong propagation
    # 1. Select primes in order (smaller primes first for better coverage)
    # 2. For each prime, try residues that maximize new coverage
    # 3. Prune branches where remaining budget can't cover remaining gaps
    
    # First, determine which primes we can afford.
    # We need sum(1/p) <= 1. Let's find a good subset of primes.
    
    # Greedy: add primes in order of increasing 1/p (i.e., decreasing p) 
    # until budget is exhausted. But we want small primes for coverage.
    
    # Better: try different subsets. Start with small primes and see if we can cover [1, L_target].
    
    # Let's try a systematic search:
    # - Consider primes in order: 2, 3, 5, 7, 11, ...
    # - At each step, decide whether to include the next prime
    # - If included, try all residues
    # - Use pruning: if current cost + minimum remaining cost > 1, prune
    
    # Actually, let's be smarter. We'll use a heuristic search:
    # 1. Start with a candidate set of primes (small ones that fit in budget)
    # 2. Use backtracking to assign residues
    # 3. Prune aggressively using coverage analysis
    
    # Let's find all subsets of primes with sum(1/p) <= 1
    # This is a knapsack-like problem. But we want small primes for coverage.
    
    # Key insight: For covering [1, L], we need primes that together can cover all residues.
    # The product of selected primes should be at least L for full coverage potential.
    
    # Let's try: use primes p where the cumulative sum 1/p <= 1.
    # Sort by prime value (ascending) and greedily add.
    
    affordable_primes = []
    current_cost = Fraction(0, 1)
    for p in primes:
        if current_cost + Fraction(1, p) <= Fraction(1, 1):
            affordable_primes.append(p)
            current_cost += Fraction(1, p)
        else:
            break
    
    # affordable_primes will be [2, 3, 5, 7, 11, 13, 17, ...] up to budget limit
    # Let's check: 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 = ?
    # = 0.5 + 0.333 + 0.2 + 0.1429 + 0.0909 + 0.0769 + 0.0588 = 1.4025 > 1
    # So we can't take all small primes.
    
    # Let's compute exactly:
    # 1/2 + 1/3 = 5/6
    # 5/6 + 1/5 = 25/30 + 6/30 = 31/30 > 1
    # So {2, 3} already exceeds... wait no, 5/6 < 1.
    # 5/6 + 1/5 = 25/30 + 6/30 = 31/30 > 1. So {2,3,5} exceeds budget.
    # {2, 3}: 5/6 <= 1. Can we add more?
    # 5/6 + 1/7 = 35/42 + 6/42 = 41/42 <= 1. 
    # 41/42 + 1/11 = 451/462 + 42/462 = 493/462 > 1.
    # So {2, 3, 7}: 41/42. Can we add 11? No.
    # What about {2, 3, 5}? 31/30 > 1. No.
    # {2, 3, 7, 13}: 41/42 + 1/13 = 533/546 + 42/546 = 575/546 > 1.
    # {2, 3, 7, 17}: 41/42 + 1/17 = 697/714 + 42/714 = 739/714 > 1.
    # {2, 3, 7, 19}: 41/42 + 1/19 = 779/798 + 42/798 = 821/798 > 1.
    # {2, 3, 7, 23}: 41/42 + 1/23 = 943/966 + 42/966 = 985/966 > 1.
    # {2, 3, 7, 29}: 41/42 + 1/29 = 1189/1218 + 42/1218 = 1231/1218 > 1.
    # {2, 3, 7, 31}: 41/42 + 1/31 = 1271/1302 + 42/1302 = 1313/1302 > 1.
    # {2, 3, 7, 37}: 41/42 + 1/37 = 1517/1554 + 42/1554 = 1559/1554 > 1.
    # {2, 3, 7, 41}: 41/42 + 1/41 = 1681/1722 + 42/1722 = 1723/1722 > 1.
    # {2, 3, 7, 43}: 41/42 + 1/43 = 1763/1806 + 42/1806 = 1805/1806 <= 1!
    # So {2, 3, 7, 43} has cost 1805/1806.
    
    # Let's verify: 1/2 + 1/3 + 1/7 + 1/43
    # = (1*3*7*43 + 2*7*43 + 2*3*43 + 2*3*7) / (2*3*7*43)
    # = (91 + 602 + 258 + 42) / 1806
    # = 1493 / 1806
    # Hmm, let me recompute.
    # 1/2 = 903/1806
    # 1/3 = 602/1806
    # 1/7 = 258/1806
    # 1/43 = 42/1806
    # Sum = (903 + 602 + 258 + 42) / 1806 = 1805/1806. Yes!
    
    # So {2, 3, 7, 43} fits in budget. Can we add more primes?
    # Remaining budget: 1/1806. No prime has 1/p <= 1/1806 for p <= 1000 (min p is 2, 1/2 > 1/1806).
    # So {2, 3, 7, 43} is maximal with these 4 primes.
    
    # But maybe other combinations give better coverage. Let's try different subsets.
    
    # Alternative: {3, 5, 7, 11, 13, 17, 19, 29}
    # 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/29
    # Let me compute this sum... it's approximately 0.333 + 0.2 + 0.1429 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0345 = 1.0331 > 1.
    # So this doesn't fit.
    
    # {3, 5, 7, 11, 13, 17, 19}: 0.333 + 0.2 + 0.1429 + 0.0909 + 0.0769 + 0.0588 + 0.0526 = 0.9551 <= 1.
    # Can we add 23? 0.9551 + 0.0435 = 0.9986 <= 1!
    # {3, 5, 7, 11, 13, 17, 19, 23}: sum ≈ 0.9986 <= 1.
    # Can we add 29? 0.9986 + 0.0345 = 1.0331 > 1. No.
    
    # So {3, 5, 7, 11, 13, 17, 19, 23} fits in budget with 8 primes!
    # This gives much more flexibility for coverage.
    
    # Let's verify exactly:
    # LCM of 3,5,7,11,13,17,19,23 is huge. Let's use Fraction.
    
    test_primes = [3, 5, 7, 11, 13, 17, 19, 23]
    test_cost = sum(Fraction(1, p) for p in test_primes)
    # Should be <= 1
    
    # Now, with 8 primes, we can try to assign residues to cover [1, 1000].
    # The total number of residue assignments is 3*5*7*11*13*17*19*23 = 3,233,230,005 which is too many.
    # We need backtracking with pruning.
    
    # Backtracking approach:
    # 1. Order primes by size (larger first for more branching, or smaller first for better pruning?)
    #    Let's try smaller primes first since they cover more integers per residue.
    # 2. For each prime, try residues in order of coverage (most new coverage first).
    # 3. Prune if:
    #    a. Current cost + min possible remaining cost > 1
    #    b. Remaining primes cannot cover the remaining uncovered integers
    
    # For (b), we need a bound on how many integers the remaining primes can cover.
    # Each prime p can cover at most ceil(L_target / p) integers per residue.
    # So the maximum additional coverage from remaining primes is sum of ceil(L_target / p) for each remaining prime.
    
    # Let's implement this.
    
    sys.setrecursionlimit(100000)
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    def search(prime_list, idx, uncovered, current_choices, current_cost):
        """
        prime_list: list of primes to consider (in order)
        idx: current index in prime_list
        uncovered: bitmask of uncovered integers in [1, L_target]
        current_choices: list of [p, r] chosen so far
        current_cost: Fraction, sum of 1/p for chosen primes
        """
        if idx == len(prime_list):
            # All primes assigned. Check if fully covered.
            if uncovered == 0:
                return current_choices
            return None
        
        p = prime_list[idx]
        
        # Try all residues, ordered by coverage (most new coverage first)
        residues = list(range(p))
        # Sort by number of new integers covered
        residues.sort(key=lambda r: -count_bits(covered_mask[p][r] & uncovered))
        
        for r in residues:
            new_cost = current_cost + Fraction(1, p)
            if new_cost > Fraction(1, 1):
                break  # Since we're going in order, all further will also exceed
            
            bits = covered_mask[p][r]
            new_uncovered = uncovered & ~bits
            
            # Pruning: check if remaining primes can cover new_uncovered
            if idx + 1 < len(prime_list):
                remaining_primes = prime_list[idx + 1:]
                max_cover = 0
                for qp in remaining_primes:
                    max_cover += (L_target + qp - 1) // qp  # ceil(L_target / qp)
                
                if count_bits(new_uncovered) > max_cover:
                    continue
            
            result = search(prime_list, idx + 1, new_uncovered, current_choices + [[p, r]], new_cost)
            if result is not None:
                return result
        
        return None
    
    # Try different prime sets
    # First try {3, 5, 7, 11, 13, 17, 19, 23}
    candidate_sets = [
        [3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 7, 43],
        [2, 3, 5, 11, 13, 17, 19],
        [2, 3, 7, 11, 13, 17],
        [3, 5, 7, 11, 13, 17, 23],
        [3, 5, 7, 11, 13, 19, 23],
        [3, 5, 7, 11, 17, 19, 23],
        [3, 5, 7, 13, 17, 19, 23],
        [3, 5, 11, 13, 17, 19, 23],
        [3, 7, 11, 13, 17, 19, 23],
        [5, 7, 11, 13, 17, 19, 23],
    ]
    
    best_result = None
    best_L = 0
    
    for prime_set in candidate_sets:
        cost = sum(Fraction(1, p) for p in prime_set)
        if cost > Fraction(1, 1):
            continue
        
        # Sort by prime value for search order
        sorted_primes = sorted(prime_set)
        
        result = search(sorted_primes, 0, full_mask, [], Fraction(0, 1))
        
        if result is not None:
            # Verify coverage
            covered = full_mask
            for p, r in result:
                covered &= ~covered_mask[p][r]
            
            if covered == 0:
                # Fully covered [1, L_target]
                if best_L < L_target:
                    best_L = L_target
                    best_result = result
    
    if best_result is not None:
        # Verify the result
        final_cost = sum(Fraction(1, p) for p, _ in best_result)
        assert final_cost <= 1, f"Final cost {final_cost} exceeds 1"
        for p, r in best_result:
            assert p <= 1000 and 0 <= r < p, f"Invalid class [{p}, {r}]"
        
        prime_list = [p for p, _ in best_result]
        assert len(prime_list) == len(set(prime_list)), "Duplicate primes"
        
        return best_result
    
    # If no full coverage found, try a greedy approach to maximize L
    # Use the best prime set and greedily assign residues
    
    # For now, return an empty list or a minimal valid solution
    # Let's try a simple greedy to get some coverage
    
    # Use {3, 5, 7, 11, 13, 17, 19, 23} and greedily assign residues
    prime_set = [3, 5, 7, 11, 13, 17, 19, 23]
    cost = sum(Fraction(1, p) for p in prime_set)
    assert cost <= 1
    
    uncovered = full_mask
    chosen = []
    
    for p in sorted(prime_set):
        best_r = 0
        best_new = 0
        for r in range(p):
            new_cov = covered_mask[p][r] & uncovered
            cnt = count_bits(new_cov)
            if cnt > best_new:
                best_new = cnt
                best_r = r
                best_bits = new_cov
        chosen.append([p, best_r])
        uncovered &= ~best_bits
    
    # Find the largest L such that [1, L] is covered
    L = 0
    for i in range(L_target):
        if not (uncovered & (1 << i)):
            L = i + 1
        else:
            break
    
    # Verify
    final_cost = sum(Fraction(1, p) for p, _ in chosen)
    assert final_cost <= 1
    for p, r in chosen:
        assert p <= 1000 and 0 <= r < p
    
    prime_list = [p for p, _ in chosen]
    assert len(prime_list) == len(set(prime_list))
    
    return chosen
# EVOLVE-BLOCK-END
