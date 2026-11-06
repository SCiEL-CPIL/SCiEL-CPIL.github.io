"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

MAX_MOD = 128
DEADLINE_SEC = 75  # safety margin under 90s wall-clock

def _modular_inverse(a, m):
    # returns x such that a*x % m == 1 (requires gcd(a,m)==1)
    # extended Euclidean
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m

def _disjoint(r1, m1, r2, m2):
    # Two infinite residue classes {r1 + k*m1} and {r2 + k*m2} are disjoint
    # iff r1 != r2 mod gcd(m1, m2)
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _build_candidates():
    """
    Build list of candidate (residue, modulus) pairs.
    For each modulus m in 2..MAX_MOD, we can pick any residue r in 0..m-1.
    But to maximize count, we want to pick one residue per modulus (since moduli must be distinct).
    So the choice is: for each m, which residue r to use.
    """
    # We'll represent a solution as a dict: modulus -> residue
    # Total moduli: 2..128, that's 127 moduli.
    # We want to select a subset of these moduli and assign residues such that all pairs are disjoint.
    # Since we want to maximize the number of classes, we want to use as many moduli as possible.
    
    # Strategy: use branch-and-bound / backtracking with pruning.
    # Order moduli in a smart way to prune early.
    # Moduli with more prime factors tend to conflict more, so maybe start with those?
    # Or start with small moduli since they have fewer residues?
    
    # Actually, let's think about it differently. The conflict between (r1,m1) and (r2,m2)
    # depends on gcd(m1,m2). If gcd(m1,m2)=1, then they're always disjoint (since r1%1==0==r2%1 always).
    # Wait: if gcd(m1,m2)=1, then r1 % 1 == 0 and r2 % 1 == 0, so they're NOT disjoint? 
    # Let me re-check: disjoint iff r1 != r2 mod gcd(m1,m2). If gcd=1, then r1%1=0, r2%1=0, so 0!=0 is false.
    # So if gcd(m1,m2)=1, they are NOT disjoint. That means coprime moduli always conflict!
    
    # Wait, that seems counterintuitive. Let me verify with an example.
    # Class A: numbers ≡ 0 mod 2 (even numbers)
    # Class B: numbers ≡ 0 mod 3 (multiples of 3)
    # These intersect at multiples of 6, so they're NOT disjoint. gcd(2,3)=1, and 0%1==0%1, so not disjoint. Correct.
    
    # Class A: numbers ≡ 1 mod 2 (odd numbers)
    # Class B: numbers ≡ 0 mod 3 (multiples of 3)
    # Do these intersect? 3, 9, 15... 3 is odd, so 3 is in both. Not disjoint. gcd(2,3)=1, 1%1==0%1==0, not disjoint.
    
    # So indeed, if gcd(m1,m2)=1, they're always not disjoint. This means we can only use moduli that are pairwise non-coprime?
    # No wait, that can't be right. Let me re-read the criterion.
    
    # "Every two infinite residue classes must be disjoint; the verifier uses the exact gcd criterion."
    # The criterion is: classes {r1 + k*m1} and {r2 + k*m2} are disjoint iff r1 != r2 mod gcd(m1,m2).
    
    # If gcd(m1,m2) = g, then the intersection is non-empty iff r1 ≡ r2 (mod g).
    # If g=1, then r1 ≡ r2 (mod 1) is always true, so intersection is always non-empty.
    # So yes, coprime moduli always have non-disjoint classes.
    
    # This is a strong constraint! We can only use moduli where every pair has gcd > 1.
    # In other words, we need a set of moduli where every pair shares a common factor > 1.
    # This is a "pairwise non-coprime" set.
    
    # For example, all even numbers: gcd of any two evens is at least 2.
    # Or all multiples of 6, etc.
    
    # So the problem reduces to:
    # 1. Find the largest subset S of {2,...,128} such that every pair in S has gcd > 1.
    # 2. For each m in S, choose a residue r_m in 0..m-1 such that for all m1,m2 in S, r_m1 != r_m2 mod gcd(m1,m2).
    
    # Step 1: Largest pairwise non-coprime subset of {2..128}.
    # All even numbers from 2 to 128: that's 64 numbers. Are they pairwise non-coprime? Yes, gcd of any two evens >= 2.
    # Can we do better? What about all multiples of 2 or 3? No, because 2 and 3 are coprime.
    # What about numbers divisible by 2, or numbers divisible by 3? No, same issue.
    # The largest pairwise non-coprime subset is likely all even numbers, which is 64.
    # Could we include some odd numbers? An odd number m must share a factor with all 64 evens.
    # The only way is if m shares a factor with every even number. But gcd(m, 2) must be > 1, which means m must be even. Contradiction.
    # So if we include 2 in our set, all other elements must be even.
    # If we don't include 2, we could potentially include odd numbers that share factors with all other elements.
    # But it's hard to beat 64.
    
    # Let's assume we use all 64 even moduli: 2,4,6,...,128.
    # Now step 2: assign residues.
    # For m1=2a, m2=2b, gcd(m1,m2) = 2*gcd(a,b).
    # We need r_{2a} != r_{2b} mod 2*gcd(a,b) for all a!=b.
    
    # This is still complex. Let me just do a backtracking search.
    
    # Actually, let's just use the even moduli and try to assign residues via backtracking.
    
    pass

def construct():
    start_time = time.time()
    
    # Use all even moduli from 2 to 128
    moduli = list(range(2, MAX_MOD + 1, 2))  # 64 moduli
    
    # Order: process in some order. Let's try processing smaller moduli first since they have fewer residues.
    # Or maybe process in order of increasing number of conflicts?
    
    # Backtracking: assign residues one by one.
    # For each modulus, try residues 0..m-1 and check compatibility with already-assigned moduli.
    
    n = len(moduli)
    assignment = [None] * n  # assignment[i] = residue for moduli[i]
    
    # Precompute gcd between all pairs
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcd_table[i][j] = math.gcd(moduli[i], moduli[j])
            gcd_table[j][i] = gcd_table[i][j]
    
    best_solution = None
    best_count = 0
    
    # Try to find a complete assignment using all 64 moduli
    def backtrack(idx):
        nonlocal best_solution, best_count
        if time.time() - start_time > DEADLINE_SEC:
            return False
        if idx == n:
            best_solution = assignment[:]
            best_count = n
            return True
        
        m = moduli[idx]
        # Try residues in some order. Maybe 0 first?
        for r in range(m):
            ok = True
            for j in range(idx):
                if assignment[j] is None:
                    continue
                g = gcd_table[idx][j]
                if (r % g) == (assignment[j] % g):
                    ok = False
                    break
            if ok:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = None
        
        return False
    
    # Try full assignment
    if not backtrack(0):
        # If full assignment fails, try to find the largest subset
        # This is more complex; for now, return what we have
        # Actually, let's try a greedy approach as fallback
        pass
    
    if best_solution is not None:
        result = [[best_solution[i], moduli[i]] for i in range(n) if best_solution[i] is not None]
        return result
    
    # Fallback: just return the seed solution
    return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

# EVOLVE-BLOCK-END
