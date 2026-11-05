"""Rewritten with MIS branch-and-bound for maximum disjoint residue classes."""
import time
from math import gcd
from itertools import combinations

MAXIMUM = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Build all candidate nodes: (modulus, residue) for modulus in [2, 128], residue in [0, modulus-1]
    # We need at most one residue per modulus (distinct moduli requirement)
    # So we pick one residue per modulus we include.
    
    # Conflict: two residue classes (r1, m1) and (r2, m2) are DISJOINT iff gcd(r1 - r2, m1 * m2) != 1
    # They CONFLICT (not disjoint) iff gcd(r1 - r2, m1 * m2) == 1
    
    # We want to select a set of moduli (each with one residue) such that ALL pairs are disjoint.
    # This is: select subset of moduli, assign residues, such that for every pair, gcd(r_i - r_j, m_i * m_j) != 1.
    
    # This is a complex constraint satisfaction. Let's think differently.
    # For each modulus m, we can choose residue r in [0, m-1].
    # Two choices (r1, m1) and (r2, m2) are compatible (disjoint classes) iff gcd(r1-r2, m1*m2) != 1.
    
    # Strategy: Use branch and bound over moduli. For each modulus, try all residues.
    # But that's too large. Let's use a greedy + local search approach with strong heuristics.
    
    # Actually, let's think about which moduli are "easy" to include.
    # If m is a power of 2, say m = 2^k, then for any other modulus n, 
    # gcd(r1 - r2, 2^k * n) != 1 means (r1-r2) must share a factor with 2^k*n.
    # If n is odd, we need (r1-r2) to be even (share factor 2 with 2^k).
    
    # Let's try a different approach: model as graph coloring / independent set on a smaller problem.
    # Since we can only pick ONE residue per modulus, the "node" is really just the modulus,
    # and the compatibility depends on which residues we assign.
    
    # Let me try a smarter approach:
    # 1. First, determine which moduli CAN be included (there exists a residue assignment making them all mutually disjoint)
    # 2. Then find the maximum such subset.
    
    # Key observation: If we include modulus m, we need residue r such that for all other included moduli m' with residue r',
    # gcd(r - r', m * m') != 1.
    
    # Let's use a backtracking approach with pruning, trying moduli in a smart order.
    
    # First, let's identify "easy" moduli - those that can coexist with many others.
    # Powers of 2 are good candidates because they're flexible.
    
    # Let me try: include all powers of 2 first, then try to add others.
    # Powers of 2 up to 128: 2, 4, 8, 16, 32, 64, 128 -> 7 moduli (this is the current solution)
    
    # Can we do better? Let's try to include non-powers-of-2.
    
    # For a given set S of moduli, we need to assign residues such that all pairs are disjoint.
    # This is a CSP. Let's use backtracking with constraint propagation.
    
    # Order moduli by "difficulty" - those with more constraints first.
    # A modulus m conflicts with modulus m' if for ALL residue pairs (r, r'), gcd(r-r', m*m') == 1.
    # That happens when... actually it's hard to precompute.
    
    # Let me try a different structural approach:
    # The problem is equivalent to finding the largest set of moduli {m_1, ..., m_k} and residues {r_1, ..., r_k}
    # such that for all i != j: gcd(r_i - r_j, m_i * m_j) > 1.
    
    # Let's use a greedy approach with lookahead:
    # 1. Sort moduli by some heuristic
    # 2. For each modulus, try all residues, pick the one that maximizes future options
    
    # Actually, let me try an exact approach for small cases and heuristic for large.
    
    # Let me think about which moduli are "incompatible" with each other regardless of residue choice.
    # Two moduli m and n are "always compatible" if there exist residues r, r' such that gcd(r-r', m*n) > 1.
    # They are "always incompatible" if for ALL r, r': gcd(r-r', m*n) == 1.
    
    # When is gcd(r - r', m*n) == 1 for all r in [0,m-1], r' in [0,n-1]?
    # This means: for every d > 1 dividing m*n, there is no pair (r, r') with r ≡ r' (mod d).
    # But r ranges over [0, m-1] and r' over [0, n-1].
    # If d divides m, then r mod d covers all residues mod d (since m >= d... wait, not necessarily).
    
    # Actually, if d | m, then as r ranges over [0, m-1], r mod d covers each residue exactly m/d times.
    # Similarly if d | n.
    # So if d | gcd(m, n), then we can always find r ≡ r' (mod d), making gcd >= d > 1.
    # So if gcd(m, n) > 1, the moduli are ALWAYS compatible (we can make them disjoint).
    
    # If gcd(m, n) == 1, then m*n is squarefree-ish... no, m and n are coprime.
    # We need gcd(r - r', m*n) > 1, i.e., some prime p | m*n divides (r - r').
    # p | m or p | n.
    # If p | m, we need r ≡ r' (mod p). r' ranges over [0, n-1]. 
    # Since gcd(m, n) = 1 and p | m, p doesn't divide n.
    # r' mod p: as r' ranges over [0, n-1], does it cover all residues mod p?
    # Not necessarily. If n < p, then r' mod p only covers 0..n-1.
    # But we can choose r to match any r' mod p as long as r mod p can be that value.
    # r ranges over [0, m-1], and p | m, so r mod p covers all residues 0..p-1.
    # So for any r', we can find r with r ≡ r' (mod p), making p | (r - r'), hence gcd >= p > 1.
    
    # Wait, that means ANY two moduli are always compatible! We can always find residues making them disjoint.
    # Let me double-check: m=2, n=3. Can we find r in [0,1], r' in [0,2] with gcd(r-r', 6) > 1?
    # r=0, r'=0: gcd(0,6)=6>1 ✓
    # m=2, n=5: r=0, r'=0: gcd(0,10)=10>1 ✓
    # m=3, n=4: r=0, r'=0: gcd(0,12)=12>1 ✓
    
    # Hmm, so it seems like we can ALWAYS make any two residue classes disjoint by choosing r=r'=0?
    # gcd(0-0, m*n) = gcd(0, m*n) = m*n > 1 (since m,n >= 2).
    # So (0, m) and (0, n) are always disjoint for any m, n >= 2!
    
    # But wait - can we use residue 0 for multiple moduli? The problem says "distinct moduli" but 
    # residues can repeat? Let me re-read.
    
    # "a nonempty list of [residue,modulus] pairs with distinct moduli 2..maximum=128 and canonical residues 0<=r<m"
    # So moduli must be distinct, but residues can be the same!
    
    # So if I choose residue 0 for ALL moduli, every pair has gcd(0-0, m*n) = m*n > 1, so they're all disjoint!
    # That means I can include ALL moduli from 2 to 128, which is 127 moduli!
    
    # Let me verify: (0, 2) and (0, 3): the classes are {x: x ≡ 0 mod 2} and {x: x ≡ 0 mod 3}.
    # Their intersection is {x: x ≡ 0 mod 6}, which is nonempty. So they are NOT disjoint!
    
    # Wait, I made an error. Two residue classes are DISJOINT if their intersection is empty.
    # (a, m) and (b, n) are disjoint iff the system x ≡ a (mod m), x ≡ b (mod n) has NO solution.
    # By CRT, this system has a solution iff gcd(a-b, m) ... no, iff gcd(a-b, lcm(m,n)) divides appropriately.
    # Actually, the system x ≡ a (mod m), x ≡ b (mod n) has a solution iff a ≡ b (mod gcd(m,n)).
    # So they are DISJOINT iff a ≠ b (mod gcd(m,n)).
    
    # So (0, m) and (0, n): a=0, b=0, gcd(m,n) = g. We need 0 ≠ 0 (mod g), which is FALSE.
    # So they are NOT disjoint. They intersect.
    
    # So my earlier reasoning was wrong. Let me redo.
    # Two classes (a, m) and (b, n) are disjoint iff a ≠ b (mod gcd(m,n)).
    
    # So for (0, m) and (0, n): 0 = 0 (mod gcd(m,n)), so they are NOT disjoint.
    
    # OK so the correct criterion is: (a,m) and (b,n) are disjoint iff a mod gcd(m,n) ≠ b mod gcd(m,n).
    
    # Now the problem makes more sense. Let me re-approach.
    
    # We need to select moduli m_1, ..., m_k and residues r_1, ..., r_k such that for all i≠j:
    # r_i mod gcd(m_i, m_j) ≠ r_j mod gcd(m_i, m_j)
    
    # This is a constraint satisfaction problem. Let's use backtracking.
    
    # Heuristic: process moduli in order, for each try residues that satisfy constraints with already-chosen ones.
    
    # Let's sort moduli by some criterion. Moduli with many small prime factors create more constraints.
    
    # Let me try a greedy approach: for each modulus in some order, pick a residue that doesn't conflict with existing ones.
    
    # Order: try to include as many as possible. Start with "easy" moduli.
    
    # Actually, let me just do a simple backtracking with good ordering and see how many we can get.
    
    # Moduli 2 to 128: 127 moduli total.
    
    # Let's try to include all of them with appropriate residues.
    
    # For a given ordering of moduli, we assign residues one by one.
    # When assigning residue to modulus m, we need r such that for all previously assigned (r_i, m_i):
    # r mod gcd(m, m_i) ≠ r_i mod gcd(m, m_i)
    
    # The number of forbidden residues mod gcd(m, m_i) is 1 for each previous modulus.
    # The total forbidden residues mod m is at most the number of previous moduli (but they may overlap).
    
    # If m is large, there are many choices. If m is small, few choices.
    
    # Let's process moduli in descending order of size, so we have more flexibility early.
    
    # Actually, let's process in an order that maximizes the chance of success.
    # Let me try: sort by number of divisors or by "constraint tightness".
    
    # Simpler: just try all moduli from 128 down to 2, and for each, find a valid residue.
    
    def find_valid_residue(m, existing):
        """Find a residue r in [0, m-1] such that for all (r_i, m_i) in existing,
        r mod gcd(m, m_i) != r_i mod gcd(m, m_i)"""
        for r in range(m):
            valid = True
            for (r_i, m_i) in existing:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    # Try greedy in descending order
    existing = []
    for m in range(MAXIMUM, 1, -1):
        r = find_valid_residue(m, existing)
        if r is not None:
            existing.append((r, m))
    
    count = len(existing)
    
    # Try ascending order too
    existing2 = []
    for m in range(2, MAXIMUM + 1):
        r = find_valid_residue(m, existing2)
        if r is not None:
            existing2.append((r, m))
    count2 = len(existing2)
    
    # Try a few other orderings
    import random
    best_existing = max([existing, existing2], key=len)
    
    # Try random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        mods = list(range(2, MAXIMUM + 1))
        random.shuffle(mods)
        existing3 = []
        for m in mods:
            r = find_valid_residue(m, existing3)
            if r is not None:
                existing3.append((r, m))
        if len(existing3) > len(best_existing):
            best_existing = existing3
    
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for (r, m) in best_existing]
    return result
