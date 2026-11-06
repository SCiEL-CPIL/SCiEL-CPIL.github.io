"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Generate candidate pairs (m, r)
    # We want to maximize sum(1/m). Small m are high value.
    # However, we must satisfy pairwise disjointness.
    # Disjointness condition: for any two pairs (m1, r1) and (m2, r2),
    # r1 % gcd(m1, m2) != r2 % gcd(m1, m2).
    
    # Let's consider all moduli in [20, 200].
    # For each modulus m, we can choose any residue r in [0, m-1].
    # There are too many combinations to brute force all.
    # However, we can reason about which moduli are "compatible".
    
    # Strategy:
    # 1. Identify a set of moduli M that are pairwise "compatible" in the sense that
    #    there exist residues r_m for each m in M such that the set is disjoint.
    # 2. Among compatible sets, pick the one with max sum(1/m).
    # 3. Once M is fixed, finding the optimal residues is a constraint satisfaction problem (CSP).
    #    Since the graph of constraints is likely sparse or structured, we can solve it.
    
    # Actually, a simpler approach:
    # If we pick moduli that are all multiples of a common number K, and K is large,
    # the constraint is r_i != r_j mod K.
    # If we pick moduli that are pairwise coprime, gcd=1, so r_i != r_j mod 1 is always true (0!=0 is false? No, 0 mod 1 is 0. 0 != 0 is False).
    # Wait. If gcd(m1, m2) = 1, then the condition is r1 != r2 mod 1.
    # Since any integer is 0 mod 1, r1 mod 1 = 0 and r2 mod 1 = 0.
    # So 0 != 0 is False.
    # Therefore, pairwise coprime moduli are NEVER disjoint!
    # This is a crucial insight. We MUST share factors.
    
    # If gcd(m1, m2) = g > 1, we need r1 % g != r2 % g.
    # To make this easy, we can choose residues such that they are distinct modulo the LCM of the gcds?
    # Or simpler: If we choose all moduli to be multiples of a specific number G,
    # then gcd(m_i, m_j) is a multiple of G? No, gcd(m_i, m_j) is at least G.
    # If m_i = G * k_i, then gcd(m_i, m_j) = G * gcd(k_i, k_j).
    # The constraint is r_i % (G * gcd(k_i, k_j)) != r_j % (G * gcd(k_i, k_j)).
    
    # Let's try a specific structure:
    # Choose moduli that are all multiples of 20.
    # m = 20 * k.
    # gcd(20k1, 20k2) = 20 * gcd(k1, k2).
    # We need r1 % (20*gcd(k1,k2)) != r2 % (20*gcd(k1,k2)).
    # If we set r_k = k * 20? No, r < m.
    
    # Let's look at the "seed" solution: [[20, 0], [40, 1], [60, 2], ...]
    # m_j = 20*j, r_j = j-1.
    # gcd(20j, 20k) = 20 * gcd(j, k).
    # Check disjointness for j=1, k=2:
    # gcd(20, 40) = 20.
    # r1 = 0, r2 = 1.
    # 0 % 20 = 0, 1 % 20 = 1. 0 != 1. OK.
    # Check j=2, k=4:
    # gcd(40, 80) = 40.
    # r2 = 1, r4 = 3.
    # 1 % 40 = 1, 3 % 40 = 3. 1 != 3. OK.
    # Check j=2, k=6:
    # gcd(40, 120) = 40.
    # r2 = 1, r6 = 5.
    # 1 % 40 = 1, 5 % 40 = 5. OK.
    
    # It seems choosing m_j = 20j and r_j = j-1 works for j=1..10.
    # Why?
    # Let m_i = 20i, m_j = 20j.
    # gcd(m_i, m_j) = 20 * gcd(i, j).
    # r_i = i-1, r_j = j-1.
    # We need (i-1) % (20*gcd(i,j)) != (j-1) % (20*gcd(i,j)).
    # Let g = gcd(i, j). i = g*i', j = g*j', gcd(i', j')=1.
    # Modulus is 20g.
    # i-1 = g*i' - 1.
    # j-1 = g*j' - 1.
    # (g*i' - 1) % (20g) vs (g*j' - 1) % (20g).
    # Note that g*i' - 1 is not necessarily small.
    # However, if 20g is large enough, and i-1, j-1 are distinct and less than 20g, they are distinct.
    # Max i=10, j=10. 20*10 = 200.
    # r_i = i-1 is at most 9.
    # So r_i < 20g always holds for g>=1.
    # Thus the condition reduces to i-1 != j-1, which is true for i != j.
    # So ANY set of distinct j's with m_j = 20j and r_j = j-1 is a valid disjoint set!
    
    # The score for the seed is sum_{j=1}^{10} 1/(20j) = (1/20) * H_10.
    # H_10 ≈ 2.928968. Score ≈ 0.1464.
    
    # Can we do better?
    # We are restricted to m in [20, 200].
    # The smallest moduli give the largest 1/m.
    # We should pick the smallest possible moduli.
    # The seed picks 20, 40, 60, ..., 200.
    # Are there smaller moduli we can add?
    # We can pick m=20, 21, 22, ...?
    # If we pick m=20 and m=21.
    # gcd(20, 21) = 1.
    # Condition: r20 % 1 != r21 % 1 => 0 != 0 => False.
    # So 20 and 21 CANNOT both be in the set.
    # In general, if gcd(m1, m2) = 1, they cannot coexist.
    # This implies we cannot have two coprime moduli in our set.
    # This is a very strong constraint.
    # All moduli in our set must share a common factor?
    # Not necessarily a common factor to ALL, but pairwise gcd > 1.
    # Example: 6, 10, 15.
    # gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5.
    # Can we find residues?
    # m1=6, r1. m2=10, r2. m3=15, r3.
    # r1 % 2 != r2 % 2
    # r1 % 3 != r3 % 3
    # r2 % 5 != r3 % 5
    # Let r1=0.
    # r2 must be odd (1,3,5,7,9).
    # r3 must be != 0 mod 3 (1,2,4,5,7,8,10,11,13,14).
    # r2 % 5 != r3 % 5.
    # Try r2=1.
    # r3 % 5 != 1. So r3 not in {1, 6, 11}.
    # r3 also not in {0, 3, 9, 12} (mod 3 = 0).
    # Available r3 in [0,14]:
    # 0 (no, mod3), 1 (no, mod5), 2 (ok), 3 (no, mod3), 4 (ok), 5 (ok), 6 (no, mod5), 7 (ok), 8 (ok), 9 (no, mod3), 10 (ok), 11 (no, mod5), 12 (no, mod3), 13 (ok), 14 (ok).
    # So r3=2 works.
    # Set: (6,0), (10,1), (15,2).
    # Score: 1/6 + 1/10 + 1/15 = 5/30 + 3/30 + 2/30 = 10/30 = 1/3 ≈ 0.333.
    # This is much better than 0.146.
    
    # So the strategy is to find a set of moduli in [20, 200] such that:
    # 1. Pairwise gcd > 1 (necessary condition for disjointness, as gcd=1 implies impossible).
    #    Wait, is gcd=1 strictly impossible?
    #    If gcd(m1, m2)=1, condition is r1 % 1 != r2 % 1 => 0 != 0 => False. Yes, strictly impossible.
    # 2. There exist residues r_i such that r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j) for all i != j.
    # 3. Maximize sum(1/m_i).
    
    # Since we want to maximize sum(1/m), we want small m.
    # Let's try to include as many small numbers as possible.
    # The set of "small" numbers in [20, 200] that are pairwise non-coprime?
    # Actually, we don't need pairwise non-coprime, we just need the residue system to exist.
    # But gcd=1 makes it impossible. So we MUST avoid coprime pairs.
    # This means the set of moduli must be a "clique" in the graph where edges connect non-coprime numbers?
    # No, we just need the set to be such that for EVERY pair, gcd > 1.
    # This is equivalent to saying the set is a clique in the "non-coprime" graph.
    # Or simply: the set of moduli must be such that no two are coprime.
    
    # Let's find a large clique of non-coprime numbers in [20, 200] with high sum(1/m).
    # Numbers with many small factors are good candidates.
    # Multiples of 2: 20, 22, 24, 26, ...
    # Multiples of 3: 21, 24, 27, ...
    # If we pick all even numbers in [20, 200], they are all pairwise non-coprime (gcd >= 2).
    # Can we assign residues to all even numbers?
    # Let M be the set of even numbers in [20, 200].
    # Size: (200-20)/2 + 1 = 91 numbers.
    # If we can assign residues to all 91, the score will be huge.
    # Score ≈ sum_{k=10}^{100} 1/(2k) = 0.5 * H_100 ≈ 0.5 * 5.18 = 2.59.
    # This is much larger than 0.146 or 0.333.
    
    # Is it possible to assign residues to all even numbers?
    # Let m_i = 2 * k_i, where k_i are integers from 10 to 100.
    # gcd(m_i, m_j) = 2 * gcd(k_i, k_j).
    # Condition: r_i % (2*gcd(k_i, k_j)) != r_j % (2*gcd(k_i, k_j)).
    # Let's try r_i = k_i - 10? Or r_i = k_i?
    # If r_i = k_i, then r_i < m_i = 2k_i is true for k_i > 0.
    # Check condition:
    # k_i % (2*gcd(k_i, k_j)) != k_j % (2*gcd(k_i, k_j)).
    # Let g = gcd(k_i, k_j). k_i = g*a, k_j = g*b, gcd(a,b)=1.
    # Modulus is 2g.
    # k_i % 2g = g*a % 2g.
    # k_j % 2g = g*b % 2g.
    # We need (g*a % 2g) != (g*b % 2g).
    # Note that g*a % 2g is either 0 (if a even) or g (if a odd).
    # Similarly for b.
    # Case 1: a even, b even. Impossible since gcd(a,b)=1.
    # Case 2: a odd, b odd.
    #   g*a % 2g = g.
    #   g*b % 2g = g.
    #   g != g is False.
    #   So if both a and b are odd, this assignment fails.
    #   a, b are odd iff k_i/g and k_j/g are odd.
    #   This happens if v_2(k_i) = v_2(k_j). (Both have the same power of 2).
    #   Example: k_i=10 (2*5), k_j=14 (2*7). g=2. a=5, b=7. Both odd.
    #   m_i=20, m_j=28.
    #   r_i=10, r_j=14.
    #   gcd(20,28)=4.
    #   10 % 4 = 2. 14 % 4 = 2. 2!=2 False.
    #   So r_i = k_i fails for pairs with same 2-adic valuation.
    
    # We need a better residue assignment.
    # We need r_i such that for any i,j with same 2-adic valuation,
    # r_i % (2*gcd(k_i,k_j)) != r_j % (2*gcd(k_i,k_j)).
    # Actually, the constraint is local to the gcd.
    # Let's use a greedy approach or backtracking to assign residues.
    # Since the number of variables is ~90, and constraints are pairwise,
    # we can use a simple backtracking search.
    # Order variables by m (smallest first) to fail fast? Or largest?
    # Smallest m have highest weight, so we want to ensure they are included.
    # But they also have more constraints?
    # Let's just try to include ALL even numbers in [20, 200].
    # If that fails, we can remove some.
    
    # Even better: Can we include ODD numbers too?
    # If we include an odd number, it must share a factor with all even numbers.
    # An odd number cannot share factor 2.
    # So it must share factor 3, 5, etc.
    # If we pick 21 (3*7), it must share factor with 20 (2^2*5). gcd(20,21)=1. Impossible.
    # So we cannot mix even and odd numbers if the even number is not a multiple of the odd number's factors?
    # Actually, if we pick a set of numbers all divisible by 2, we cannot add any odd number,
    # because any odd number is coprime to some even number?
    # No. If we pick 6 (even) and 9 (odd). gcd(6,9)=3. OK.
    # But if we pick 20 (even) and 9 (odd). gcd(20,9)=1. Fail.
    # So if our set contains 20, we cannot contain 9.
    # To maximize sum(1/m), we want small numbers.
    # 20 is small. 9 is not in range [20,200].
    # Smallest odd in range: 21.
    # Can we have 20 and 21? gcd(20,21)=1. No.
    # Can we have 22 and 21? gcd(22,21)=1. No.
    # Can we have 24 and 21? gcd(24,21)=3. Yes.
    # So we could potentially have a mix.
    # However, the set of all even numbers is very dense and gives a high score.
    # Let's stick to even numbers first.
    
    # Algorithm:
    # 1. Start with all even numbers in [20, 200].
    # 2. Try to assign residues.
    # 3. If assignment fails, remove the "least useful" number (largest m) and retry.
    # 4. Repeat until a valid assignment is found or time runs out.
    
    # Residue Assignment Strategy:
    # We need r_i for each m_i.
    # Constraint: r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j).
    # This is a graph coloring problem on a constraint graph?
    # The domain of r_i is [0, m_i-1].
    # We can use backtracking.
    # To make it fast, we can precompute the constraints.
    
    # Let's implement a backtracking solver.
    # Variables: indices of the selected moduli.
    # Order: sort by m ascending.
    
    moduli = list(range(20, 201, 2)) # All evens
    
    # Precompute gcds?
    # gcd(m_i, m_j)
    
    def check_disjointness(selected_pairs):
        """Check if a list of (m, r) is pairwise disjoint."""
        for i in range(len(selected_pairs)):
            m1, r1 = selected_pairs[i]
            for j in range(i+1, len(selected_pairs)):
                m2, r2 = selected_pairs[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def solve_residues(moduli_list):
        """Find residues for the given moduli list such that they are disjoint.
        Returns list of (m, r) or None if no solution found within time limit."""
        n = len(moduli_list)
        if n == 0:
            return []
        
        # Sort by modulus size to prioritize small ones?
        # Actually, for backtracking, we want to fail fast.
        # Variables with smaller domains or more constraints should go first.
        # Here domain size is m. Smaller m has smaller domain.
        # So sort by m ascending.
        sorted_indices = sorted(range(n), key=lambda i: moduli_list[i])
        
        # We will assign residues in this order.
        assignments = [None] * n
        current_time = time.time()
        
        def backtrack(idx):
            nonlocal current_time
            if time.time() > current_time + 5: # 5s per backtrack attempt? No, global deadline.
                return False
            
            if idx == n:
                return True
            
            i = sorted_indices[idx]
            m_i = moduli_list[i]
            
            # Try residues in some order.
            # Heuristic: Try residues that are "different" from already assigned ones modulo gcd.
            # Simple: Iterate 0 to m_i-1.
            for r in range(m_i):
                # Check against all previously assigned
                valid = True
                for k in range(idx):
                    j = sorted_indices[k]
                    m_j = moduli_list[j]
                    r_j = assignments[j]
                    g = gcd(m_i, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                
                if valid:
                    assignments[i] = r
                    if backtrack(idx + 1):
                        return True
                    assignments[i] = None
            return False
        
        if backtrack(0):
            return [(moduli_list[i], assignments[i]) for i in range(n)]
        else:
            return None

    # Try the full set of evens
    best_result = None
    best_score = 0.0
    
    # Start with all evens
    current_moduli = moduli[:]
    
    # It might be that not all evens can be assigned.
    # We will iteratively remove the largest modulus if it fails.
    # This is a greedy heuristic.
    
    while current_moduli:
        if time.time() > deadline:
            break
            
        result = solve_residues(current_moduli)
        if result:
            # Calculate score
            score = sum(1.0 / m for m, r in result)
            if score > best_score:
                best_score = score
                best_result = result
            
            # If we found a valid set, can we add more?
            # We started with all evens. If it failed, we removed one.
            # If it succeeded, we have a valid set.
            # Can we try to add odd numbers?
            # Probably not worth the complexity for the time limit.
            # The score of all evens is ~2.5.
            # Adding an odd number m requires it to be non-coprime with all existing.
            # Given the density of evens, this is hard.
            break
        else:
            # Remove the largest modulus (lowest weight)
            current_moduli.pop()
            
    # Fallback: If the greedy removal didn't find a good set, or if we ran out of time,
    # we should return the best found so far.
    # If best_result is None, we might need a simpler fallback.
    # The seed solution is always valid.
    if best_result is None:
        # Use seed solution
        best_result = [[20 * j, j - 1] for j in range(1, 11)]
        best_score = sum(1.0 / (20 * j) for j in range(1, 11))
        
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_result]

# EVOLVE-BLOCK-END
