"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction

def _primes_upto(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    """
    Construct a set of prime residue classes covering 1..L for the largest possible L,
    subject to sum(1/p) <= 1 and distinct primes p <= 1000.
    
    Strategy: Use a greedy set-cover approach with exact rational arithmetic.
    We prioritize primes that cover many currently-uncovered integers in [1, CAP].
    We stop when adding any more prime would exceed the budget sum(1/p) <= 1.
    
    To maximize L, we focus coverage on the low end of the range.
    """
    CAP = 10**5
    primes = _primes_upto(1000)
    
    # We want to cover 1..L. Let's work with a target range.
    # The budget is sum(1/p) <= 1.
    # Greedy set cover: at each step, pick the (p, r) that maximizes newly covered integers in [1, CAP]
    # among those that fit in the remaining budget.
    
    # Precompute: for each prime p, the residue r that covers the most uncovered numbers in [1, CAP].
    # But "uncovered" changes as we add classes. So we need an iterative greedy.
    
    # For efficiency, we track which integers in [1, CAP] are covered.
    # CAP = 10^5, so a boolean array of size 10^5 is fine.
    
    covered = [False] * (CAP + 1)
    covered_count = 0
    
    used_primes = set()
    budget_used = Fraction(0, 1)
    classes = []
    
    # For each prime, precompute the count of multiples in [1, CAP] for each residue.
    # count[p][r] = number of n in [1, CAP] with n ≡ r (mod p)
    # This is roughly floor(CAP/p) or ceil(CAP/p) depending on r.
    
    # Actually, for greedy set cover with dynamic uncovered set, we need to know
    # how many *currently uncovered* integers would be covered by (p, r).
    # This is expensive to compute naively each step.
    
    # Alternative: since we want to maximize L (the prefix), we should focus on covering small integers.
    # Let's use a different approach: 
    # 1. Start with no classes.
    # 2. At each step, for each unused prime p, find the residue r that maximizes the number of
    #    currently-uncovered integers in [1, L_target] that would be covered, where L_target is the
    #    current maximum contiguous prefix we're trying to extend.
    
    # Actually, let's think about this differently. The key insight for covering 1..L is that
    # small primes are more valuable because they have smaller 1/p cost and cover more numbers.
    # But the constraint is sum(1/p) <= 1.
    
    # Let's use a greedy approach that prioritizes extending the covered prefix.
    # At each step:
    #   - Let L_cur be the largest L such that 1..L are all covered.
    #   - For each unused prime p that fits in the remaining budget, find the residue r
    #     that maximizes the extension of the covered prefix (i.e., the largest L' > L_cur
    #     such that 1..L' would be covered after adding (p, r)).
    #   - Pick the (p, r) that gives the largest L'. If ties, pick the one with smallest 1/p.
    #   - Add it and update.
    #   - Stop when no prime can be added within budget.
    
    # To compute the extension efficiently, we can check from L_cur+1 upward to find the first
    # uncovered integer, and see if adding (p, r) covers it and extends the prefix.
    
    # Let's implement this.
    
    def current_prefix(covered, CAP):
        """Return the largest L such that 1..L are all covered."""
        L = 0
        for i in range(1, CAP + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    # Initial prefix is 0.
    # We'll iterate: at each step, try to extend the prefix.
    
    # To make this efficient, let's precompute for each prime p the list of residues
    # and how they cover numbers. But the dynamic nature makes this tricky.
    
    # Let's just do the greedy step by step. The number of primes is ~168, and CAP is 10^5.
    # Each step, for each unused prime, we try all p residues and compute the new prefix.
    # That's 168 * 1000 * 10^5 = too slow.
    
    # Better: for each (p, r), compute how many new integers in [1, CAP] are covered,
    # and specifically whether it extends the prefix.
    
    # Let's use a smarter approach: 
    # For a given current covered set, and a candidate (p, r), the new covered set is
    # covered OR (n ≡ r mod p). The new prefix is the largest L such that all of 1..L
    # are in covered OR (n ≡ r mod p).
    
    # We can compute this by finding the first integer in [1, CAP] that is NOT in
    # (covered OR (n ≡ r mod p)). That's the first integer that is uncovered AND not ≡ r mod p.
    
    # Let's precompute for each prime p, for each residue r, the sorted list of numbers
    # in [1, CAP] that are ≡ r mod p. Then for a given covered set, the new prefix is
    # the minimum of (first uncovered not in residue class) - 1.
    
    # This is still complex. Let's try a simpler heuristic:
    # Use the greedy set cover maximizing total newly-covered integers, with a bias
    # toward small integers (to extend the prefix).
    
    # Weights: w[n] = 1 if n is in [1, CAP], with higher weight for smaller n.
    # Let w[n] = 1/n or something. Actually, to maximize the prefix, we want to cover
    # small n first. Let's use w[n] = 1 for all n, but prioritize coverage of the
    # current prefix boundary.
    
    # Let's try: at each step, for each unused prime p, find the residue r that covers
    # the most currently-uncovered integers in [1, CAP]. Among those, prefer the one
    # that also covers the current prefix boundary (L_cur + 1).
    
    # Let's implement a straightforward greedy:
    
    L_cur = 0
    
    while True:
        best_p = None
        best_r = None
        best_gain = 0
        best_new_prefix = 0
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > Fraction(1, 1):
                continue
            
            # Try all residues for this prime
            for r in range(p):
                # Count newly covered integers
                gain = 0
                # Check if this covers L_cur + 1
                covers_boundary = ((L_cur + 1) % p == r) if L_cur < CAP else False
                
                # Count how many currently-uncovered integers in [1, CAP] are ≡ r mod p
                # This is expensive. Let's approximate or compute efficiently.
                
                # For now, let's just count by iterating. This is O(CAP/p) per residue.
                # Total per step: sum over p of p * (CAP/p) = sum over p of CAP = 168 * 10^5 = 1.68 * 10^7
                # That's too slow for 168 steps.
                
                # Let's precompute something.
                
                # Alternative: for each prime p, precompute count[p][r] = number of n in [1, CAP] with n ≡ r mod p.
                # But we need *uncovered* count, which changes.
                
                # Let's just do a quick estimate: if covers_boundary, prioritize.
                # Otherwise, use count[p][r] as a proxy.
                
                pass
        
        # This is getting too complex. Let's use a simpler approach.
        
        # Simple approach: use the known result that the optimal covering for small L
        # uses small primes. Let's just try all primes in order of increasing 1/p
        # (i.e., decreasing p) and pick the residue that covers the most uncovered numbers.
        
        # Actually, let's just use a heuristic: for each prime, pick the residue r
        # such that r ≡ 0 mod p covers multiples, or r is chosen to cover small numbers.
        
        # Let's try: for each prime p, the residue r = (L_cur + 1) % p covers the boundary.
        # Use that if it fits in budget.
        
        added = False
        for p in sorted(primes, key=lambda x: x):  # Smallest primes first (most cost-effective per number covered)
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > Fraction(1, 1):
                continue
            
            # Pick residue to cover L_cur + 1
            r = (L_cur + 1) % p
            
            # Check how many new integers this covers in [1, CAP]
            new_covered = 0
            for n in range(1, CAP + 1, p):
                if (n + p - r - 1) % p == 0:  # n ≡ r mod p
                    # Actually, let's just check directly
                    pass
            
            # Let's just add it and see if it extends the prefix
            # Tentatively mark covered
            for n in range(1, CAP + 1):
                if not covered[n] and n % p == r:
                    covered[n] = True
                    new_covered += 1
            
            # Compute new prefix
            new_L = 0
            for i in range(1, CAP + 1):
                if covered[i]:
                    new_L = i
                else:
                    break
            
            if new_L > L_cur:
                # Commit
                used_primes.add(p)
                budget_used += cost
                classes.append([p, r])
                L_cur = new_L
                added = True
                break
            else:
                # Undo
                for n in range(1, CAP + 1):
                    if covered[n] and n % p == r:
                        covered[n] = False
        
        if not added:
            break
    
    # Compute final prefix
    L_final = 0
    for i in range(1, CAP + 1):
        if covered[i]:
            L_final = i
        else:
            break
    
    return classes
# EVOLVE-BLOCK-END
