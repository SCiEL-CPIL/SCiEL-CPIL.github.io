"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    MAX_M = 128
    
    # We want to select a subset of moduli m in [2, 128] and assign residues r_m in [0, m-1]
    # such that for any two selected moduli m1, m2 with residues r1, r2:
    #   gcd(m1, m2) divides (r1 - r2)  is FALSE
    # i.e., the classes are disjoint iff gcd(m1,m2) does NOT divide (r1-r2) for all pairs.
    # Wait, let's re-read the criterion.
    # Two residue classes a mod m and b mod n are disjoint iff gcd(m,n) does not divide (a-b).
    # So we need: for all pairs (m1,r1), (m2,r2) in our set, gcd(m1,m2) ∤ (r1 - r2).
    
    # Strategy: Backtracking search.
    # Variables: moduli m in [2, 128]. We can choose to include or exclude each modulus.
    # If included, we choose a residue r.
    # To maximize count, we prefer including more moduli.
    # Heuristic: Process moduli in an order that maximizes constraints early?
    # Or simply iterate through all moduli and try to include them if possible.
    # Since we want to MAXIMIZE the number of classes, we should try to include as many moduli as possible.
    # This is a maximum independent set problem on a graph where nodes are (m, r) pairs?
    # No, the nodes are moduli, but the compatibility depends on the chosen residue.
    
    # Let's use a greedy approach with backtracking or just a smart greedy.
    # Given the small size (127 moduli), a backtracking search with good heuristics should work.
    # However, the search space is huge.
    
    # Alternative insight:
    # If we pick moduli that are pairwise coprime, then gcd(m1,m2)=1, and 1 divides (r1-r2) always.
    # So if gcd(m1,m2)=1, the classes are NEVER disjoint?
    # Wait. gcd(m1,m2)=1 implies 1 divides (r1-r2). So the condition "gcd does not divide (r1-r2)" is FALSE.
    # This means if gcd(m1,m2)=1, the classes are NOT disjoint. They always intersect.
    # Let's verify: a ≡ r1 (mod m1), b ≡ r2 (mod m2).
    # If gcd(m1,m2)=1, CRT says there is always a solution for x ≡ r1 (mod m1) and x ≡ r2 (mod m2).
    # So yes, if gcd(m1,m2)=1, the classes intersect.
    # Therefore, to have disjoint classes, we MUST have gcd(m1,m2) > 1 for every pair.
    # AND, we must ensure that gcd(m1,m2) does NOT divide (r1-r2).
    
    # So the problem reduces to:
    # 1. Select a subset of moduli S ⊆ {2,...,128} such that for all m1,m2 in S, gcd(m1,m2) > 1.
    #    (i.e., no two moduli are coprime).
    # 2. Assign residues r_m to each m in S such that for all m1,m2 in S,
    #    gcd(m1,m2) ∤ (r1 - r2).
    
    # Step 1: Find a large subset of moduli with no coprime pairs.
    # This is equivalent to finding a large clique in the graph where edges connect non-coprime numbers.
    # Or, finding a large set of numbers such that every pair shares a prime factor.
    # By the Erdős–Ko–Rado type results or just simple observation:
    # If we pick all even numbers, they all share factor 2.
    # If we pick all multiples of 3, they all share factor 3.
    # We can mix them, but if we pick an even number and a multiple of 3 that is odd (e.g., 6 is even, 9 is odd),
    # gcd(6,9)=3 > 1. OK.
    # But if we pick 4 (even) and 9 (multiple of 3), gcd(4,9)=1. Not allowed.
    # So we cannot mix "pure" even numbers (powers of 2) with "pure" multiples of 3 (powers of 3) if they don't share other factors.
    
    # The largest such set is likely all numbers divisible by a specific prime p, or a combination.
    # Let's check the size of {m in [2,128] : 2|m}. Count = 63.
    # Let's check {m in [2,128] : 3|m}. Count = 42.
    # Can we do better?
    # If we take all numbers divisible by 2 or 3?
    # 4 (div by 2) and 9 (div by 3) -> gcd(4,9)=1. Fail.
    # So we can't just take the union.
    # What if we take all numbers divisible by 2? Size 63.
    # Can we add any number not divisible by 2?
    # If we add an odd number m, it must share a factor with ALL even numbers in the set.
    # The only factor an odd number can share with an even number is an odd factor.
    # For m to share a factor with 2, it must be divisible by 2 (impossible for odd).
    # Wait, gcd(2, m) must be > 1. If m is odd, gcd(2,m)=1.
    # So if 2 is in the set, NO odd number can be in the set.
    # So if we include 2, we can only include even numbers.
    # Max size including 2 is 63 (all evens).
    
    # What if we exclude 2?
    # We can include 3. Then we can include multiples of 3.
    # Can we include 4? gcd(3,4)=1. No.
    # Can we include 6? gcd(3,6)=3. Yes.
    # So if we include 3 and 6, we can include other multiples of 3.
    # Can we include 4? No, gcd(3,4)=1.
    # Can we include 5? No, gcd(3,5)=1.
    # Can we include 9? Yes.
    # So the set must be a subset of multiples of 3?
    # If we pick a set where every number is divisible by 3, size is 42.
    # Is there a larger set?
    # Consider primes. If we pick a set of numbers all divisible by 2, size 63.
    # If we pick a set of numbers all divisible by 3, size 42.
    # If we pick a set of numbers all divisible by 5, size 25.
    # 
    # What if we don't fix a single prime?
    # Example: {6, 10, 15}.
    # gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. All > 1.
    # Can we add 30? Yes.
    # Can we add 2? gcd(2,15)=1. No.
    # Can we add 3? gcd(3,10)=1. No.
    # Can we add 5? gcd(5,6)=1. No.
    # Can we add 4? gcd(4,15)=1. No.
    # Can we add 9? gcd(9,10)=1. No.
    # Can we add 12? gcd(12,10)=2, gcd(12,15)=3. Yes.
    # Can we add 18? gcd(18,10)=2, gcd(18,15)=3. Yes.
    # Can we add 20? gcd(20,15)=5, gcd(20,6)=2. Yes.
    # Can we add 24? gcd(24,15)=3, gcd(24,10)=2. Yes.
    # Can we add 25? gcd(25,6)=1. No.
    # 
    # It seems the set of all even numbers (size 63) is a strong candidate.
    # Let's verify if we can get more than 63.
    # Total numbers 2..128 is 127.
    # If we exclude all odds, we have 63 evens.
    # If we exclude 2, we can potentially include some odds.
    # But any odd number m must have gcd(m, n) > 1 for all n in the set.
    # If the set contains any number not divisible by m's prime factors, it fails.
    # Specifically, if the set contains 2, no odd can be included.
    # If the set does NOT contain 2, can we include more?
    # Suppose we include 3, 5, 7...
    # If we include 3 and 5, we can include 15.
    # But we cannot include 2 (gcd(2,3)=1).
    # We cannot include 4 (gcd(4,3)=1).
    # We cannot include 6? gcd(6,5)=1. No.
    # So if we include 3 and 5, we can only include numbers divisible by 3 or 5?
    # No, 6 is div by 3, but gcd(6,5)=1. So 6 cannot be included if 5 is included.
    # So the set must be numbers divisible by 3 AND 5? i.e., 15, 30...
    # Or numbers divisible by 3, but then 5 cannot be included if any non-multiple of 5 is included.
    # This suggests that mixing primes reduces the set size significantly.
    # The "star" configurations (all multiples of p) are likely optimal.
    # Max size is 63 (multiples of 2).
    
    # So, Step 1: Select S = {m in [2,128] : m % 2 == 0}. Size 63.
    # Step 2: Assign residues r_m for each m in S such that for all m1, m2 in S,
    #         gcd(m1, m2) ∤ (r1 - r2).
    
    # Let's analyze the constraints for S = evens.
    # For any m1, m2 in S, let g = gcd(m1, m2).
    # Since m1, m2 are even, g is at least 2.
    # We need r1 - r2 NOT to be a multiple of g.
    
    # Consider m1 = 2, m2 = 4. g = 2.
    # r1 in [0,1], r2 in [0,3].
    # r1 - r2 must not be divisible by 2.
    # So r1 and r2 must have different parity.
    # 
    # Consider m1 = 4, m2 = 6. g = 2.
    # r1 (mod 4) and r2 (mod 6) must have different parity.
    # 
    # Consider m1 = 4, m2 = 8. g = 4.
    # r1 (mod 4) and r2 (mod 8) must NOT satisfy r1 ≡ r2 (mod 4).
    # 
    # Consider m1 = 6, m2 = 10. g = 2.
    # Parity must differ.
    
    # It seems the parity constraint is very strong.
    # If we set r_m = 0 for all m, then r1-r2=0, which is divisible by any g. Fail.
    # If we set r_m = 1 for all m, same.
    # 
    # Let's try a systematic assignment.
    # For each m, we can choose r_m.
    # Notice that if we choose r_m such that r_m is always odd?
    # Then r1 - r2 is even.
    # If g is even (which it is, since m1,m2 even), then r1-r2 being even means it IS divisible by 2.
    # But we need it NOT divisible by g.
    # If g=2, r1-r2 even means r1-r2 is a multiple of 2. So r1 ≡ r2 (mod 2).
    # This violates the condition for g=2.
    # So r1 and r2 must have DIFFERENT parity if gcd(m1,m2)=2.
    
    # This implies we cannot have all residues with the same parity.
    # In fact, the graph of constraints "r1 != r2 (mod g)" is complex.
    
    # Let's try a greedy assignment with backtracking for the residues.
    # Order moduli by descending value? Or ascending?
    # Ascending might be better as small moduli have fewer residues.
    
    # Let's implement a backtracking search for residues given the fixed set S = evens.
    # If that fails, we can try other sets (e.g., remove some evens to allow odds, or use multiples of 3).
    
    # Optimization: Precompute gcds for the set S.
    
    S = [m for m in range(2, MAX_M + 1) if m % 2 == 0]
    # Size 63.
    
    # Sort S by number of residues (ascending) to constrain early?
    # All have m residues. Small m has fewer choices.
    # So sort by m ascending.
    S_sorted = sorted(S)
    
    # Precompute gcd matrix
    n = len(S_sorted)
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(S_sorted[i], S_sorted[j])
            gcds[j][i] = gcds[i][j]
    
    # Backtracking
    # r[i] is the residue for S_sorted[i]
    r = [-1] * n
    best_solution = None
    best_count = 0
    
    # Heuristic: Try residues that are "different" from previous ones.
    # Since the constraint is r_i - r_j NOT divisible by g_ij,
    # we want r_i mod g_ij != r_j mod g_ij.
    
    def is_valid(i, val):
        for j in range(i):
            if r[j] != -1:
                g = gcds[i][j]
                if (val - r[j]) % g == 0:
                    return False
        return True
    
    def backtrack(i):
        nonlocal best_solution, best_count
        if time.time() > deadline:
            return False
        
        if i == n:
            best_solution = r[:]
            best_count = n
            return True
        
        # Try all residues for S_sorted[i]
        m = S_sorted[i]
        # To find a valid residue quickly, we can shuffle or use a heuristic.
        # Let's just iterate 0..m-1.
        for val in range(m):
            if is_valid(i, val):
                r[i] = val
                if backtrack(i+1):
                    return True
        r[i] = -1
        return False
    
    # Run backtracking
    # This might be slow if it backtracks a lot.
    # Let's add a randomization or better heuristic.
    # Actually, for S=evens, is a solution always possible?
    # Let's try.
    
    success = backtrack(0)
    
    if success:
        result = [[r[i], S_sorted[i]] for i in range(n)]
        return result
    else:
        # If backtracking fails or times out, return best found or fallback
        # Fallback: The seed program.
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
# EVOLVE-BLOCK-END
