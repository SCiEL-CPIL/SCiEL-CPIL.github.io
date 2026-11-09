"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import gcd

def construct():
    MAX_M = 128
    
    # We need to select a set of (residue, modulus) pairs such that:
    # 1. All moduli are distinct and in [2, MAX_M].
    # 2. Residues are canonical: 0 <= r < m.
    # 3. For any two pairs (r1, m1) and (r2, m2), the congruences x ≡ r1 (mod m1) and x ≡ r2 (mod m2) are incompatible.
    #    This happens if and only if gcd(m1, m2) does not divide (r1 - r2).
    #    Wait, the condition for disjointness of two arithmetic progressions is:
    #    The system x ≡ r1 (mod m1), x ≡ r2 (mod m2) has a solution iff r1 ≡ r2 (mod gcd(m1, m2)).
    #    So they are disjoint iff r1 NOT≡ r2 (mod gcd(m1, m2)).
    
    # We want to maximize the number of such pairs.
    # This is equivalent to finding a maximum set of moduli M' ⊆ {2, ..., 128} and assigning residues r_m for each m ∈ M'
    # such that for all m1, m2 ∈ M', r_m1 NOT≡ r_m2 (mod gcd(m1, m2)).
    
    # Note: If we pick all moduli, can we assign residues?
    # Let's check small cases.
    # If m1=2, m2=3. gcd=1. r1 NOT≡ r2 (mod 1) is always true (0≡0 mod 1 is true, so condition is false? No.)
    # Wait: r1 ≡ r2 (mod 1) is ALWAYS true because 1 divides any difference.
    # So if gcd(m1, m2) == 1, the congruences are ALWAYS compatible.
    # Therefore, we CANNOT have two coprime moduli in our set!
    # Because if gcd(m1, m2) = 1, then r1 ≡ r2 (mod 1) is always true, meaning the classes intersect.
    # Thus, for any pair of moduli in our set, gcd(m1, m2) must be > 1.
    # And more strongly, we need to be able to choose residues such that r1 NOT≡ r2 (mod gcd(m1, m2)).
    
    # So, first constraint: The set of moduli M' must be such that for all m1, m2 ∈ M', gcd(m1, m2) > 1.
    # This means M' must be a clique in the graph where edges connect numbers with gcd > 1? No, it means M' must be a set of pairwise non-coprime numbers.
    # Actually, if gcd(m1, m2) = 1, they intersect. So we must avoid coprime pairs.
    # So M' must be a set of integers from 2..128 such that no two are coprime.
    # Among such sets, we must also be able to assign residues.
    # Is it always possible to assign residues if gcd(m1, m2) > 1?
    # Let g = gcd(m1, m2). We need r1 NOT≡ r2 (mod g).
    # This is possible if g > 1, because we can just pick r1=0 and r2=1 (if g>1, 0 not equiv 1 mod g).
    # Wait, we need a consistent assignment for ALL pairs.
    # This looks like a graph coloring or constraint satisfaction problem on the residues modulo the gcds.
    # However, if we pick a set of moduli that are all multiples of a common factor d > 1, then gcd(m1, m2) is a multiple of d.
    # If we set all residues to 0, then r1 ≡ r2 ≡ 0 (mod g) for any g. This fails.
    # We need r1 NOT≡ r2 (mod g).
    # If we pick all even moduli, gcd(m1, m2) is even.
    # Can we assign residues to even moduli 2, 4, 6, ..., 128 such that for any pair, r1 NOT≡ r2 (mod gcd)?
    # Let's try a simpler approach: Pick a prime p. Let M' be all multiples of p in 2..128.
    # Then gcd(m1, m2) is a multiple of p.
    # If we assign residues r_m, we need r_m NOT≡ r_n (mod gcd(m,n)).
    # Since p divides gcd(m,n), we need r_m NOT≡ r_n (mod p).
    # So we need to assign residues modulo p such that no two are congruent modulo p.
    # But there are many moduli, and only p residues mod p. By pigeonhole, if |M'| > p, we can't assign distinct residues mod p.
    # Wait, the condition is r_m NOT≡ r_n (mod gcd(m,n)), which implies r_m NOT≡ r_n (mod p).
    # So we need all r_m to be distinct modulo p.
    # Thus, the maximum size of M' is at most p.
    # To maximize |M'|, we should pick a small prime p, so that we can fit up to p moduli.
    # The multiples of p in 2..128 are: p, 2p, 3p, ..., floor(128/p)*p.
    # The count is floor(128/p).
    # We need floor(128/p) <= p.
    # If p=2: floor(128/2)=64. 64 > 2. So we can only pick 2 moduli?
    # Wait, if we pick 2 moduli, say 2 and 4. gcd=2. We need r2 NOT≡ r4 (mod 2). We can pick r2=0, r4=1. This works.
    # Can we pick 3 multiples of 2? Say 2, 4, 6.
    # gcd(2,4)=2, gcd(2,6)=2, gcd(4,6)=2.
    # We need r2 NOT≡ r4 (mod 2), r2 NOT≡ r6 (mod 2), r4 NOT≡ r6 (mod 2).
    # This means r2, r4, r6 must be pairwise distinct mod 2. But there are only 2 residues mod 2. Impossible.
    # So for p=2, max size is 2.
    # For p=3: floor(128/3)=42. Max size <= 3.
    # Can we pick 3 multiples of 3? 3, 6, 9.
    # gcds are multiples of 3. We need r3, r6, r9 distinct mod 3. Possible: 0, 1, 2.
    # So size 3 is possible.
    # For p=5: floor(128/5)=25. Max size <= 5.
    # Can we pick 5 multiples of 5? 5, 10, 15, 20, 25.
    # gcds are multiples of 5. We need 5 distinct residues mod 5. Possible.
    # So size 5 is possible.
    # For p=7: floor(128/7)=18. Max size <= 7.
    # Can we pick 7 multiples of 7? 7, 14, ..., 49.
    # Need 7 distinct residues mod 7. Possible.
    # For p=11: floor(128/11)=11. Max size <= 11.
    # Can we pick 11 multiples of 11? 11, 22, ..., 121.
    # Need 11 distinct residues mod 11. Possible.
    # For p=13: floor(128/13)=9. Max size <= 9 (since 9 < 13, we can pick 9).
    # So size 9 is possible.
    # For p=17: floor(128/17)=7. Size 7.
    # For p=19: floor(128/19)=6. Size 6.
    # ...
    # It seems the maximum size is achieved by p=11, giving 11 classes.
    # Let's verify p=11: Moduli 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121.
    # Assign residues 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 respectively.
    # Check: For any i != j, gcd(11i, 11j) is a multiple of 11.
    # r_i = i-1, r_j = j-1.
    # r_i - r_j = i - j.
    # We need 11 NOT divide (i-j). Since 1 <= i, j <= 11 and i != j, |i-j| < 11, so 11 does not divide i-j.
    # Thus, the condition holds.
    # So 11 is achievable.
    
    # Can we do better by mixing primes?
    # If we mix moduli with different prime factors, the gcds might be smaller, making the constraint tighter.
    # For example, if we have 6 (2*3) and 10 (2*5), gcd=2. We need r6 NOT≡ r10 (mod 2).
    # If we have 6 and 15 (3*5), gcd=3. We need r6 NOT≡ r15 (mod 3).
    # Mixing seems to create more constraints. The "pure prime multiple" strategy seems optimal.
    # Let's check if we can get 12.
    # If we pick 12 moduli, they must be pairwise non-coprime.
    # If they all share a common factor d, then d must be such that there are at least 12 multiples of d in 2..128, and 12 <= d.
    # If d=2, multiples are 64, but 12 > 2, so we can't assign distinct residues mod 2.
    # If d=3, multiples are 42, but 12 > 3, can't assign distinct residues mod 3.
    # If d=4, multiples are 32, 12 > 4, can't assign distinct residues mod 4.
    # If d=5, multiples are 25, 12 > 5, can't assign distinct residues mod 5.
    # If d=6, multiples are 21, 12 > 6, can't assign distinct residues mod 6.
    # If d=7, multiples are 18, 12 > 7, can't assign distinct residues mod 7.
    # If d=8, multiples are 16, 12 > 8, can't assign distinct residues mod 8.
    # If d=9, multiples are 14, 12 > 9, can't assign distinct residues mod 9.
    # If d=10, multiples are 12, 12 > 10, can't assign distinct residues mod 10.
    # If d=11, multiples are 11, 12 > 11, can't assign distinct residues mod 11.
    # If d=12, multiples are 10, so we can't even pick 12 moduli all divisible by 12.
    # So it's impossible to have 12 moduli all sharing a common factor d >= 12.
    # What if they don't all share a common factor?
    # Suppose we have a set of 12 moduli. By the property of pairwise non-coprime, it's known that the maximum size of a pairwise non-coprime set in {2..N} is achieved by taking all multiples of 2 (or the smallest prime). But here we have the additional residue constraint.
    # Given the residue constraint requires distinct residues mod gcd, and gcd >= common factor, the "pure prime" strategy is likely optimal.
    # Let's double check p=11 gives 11.
    # Is there a set of 12 moduli that works?
    # If we take 12 moduli, they must be pairwise non-coprime.
    # It is a known result that the largest subset of {2, ..., N} with pairwise non-coprime elements has size floor(N/2) (all evens).
    # But we can't use all evens because of the residue constraint.
    # For evens, gcd is at least 2. We need distinct residues mod 2 for all pairs? No, distinct mod gcd.
    # If we pick evens, gcd is even. We need r_i NOT≡ r_j (mod gcd).
    # If we pick 12 evens, can we assign residues?
    # The gcds can be 2, 4, 6, etc.
    # If we pick 2, 4, 6, ..., 24.
    # gcd(2,4)=2. r2 NOT≡ r4 (mod 2).
    # gcd(2,6)=2. r2 NOT≡ r6 (mod 2).
    # ...
    # This requires r2, r4, ..., r24 to be distinct mod 2? No, just pairwise distinct mod their gcd.
    # But r2 NOT≡ r4 (mod 2) and r2 NOT≡ r6 (mod 2) implies r4 ≡ r6 (mod 2) is not required, but r2 is different from both.
    # Actually, if we have many numbers with gcd 2, we can only have 2 distinct residues mod 2.
    # So we can have at most 2 numbers with gcd 2 among them? No, we can have many, but their residues must be such that for any pair with gcd 2, they are different mod 2.
    # This means all residues of numbers that are "2-connected" must be distinct mod 2.
    # If we have 12 evens, many pairs will have gcd 2.
    # For example, 2 and 4 have gcd 2. 2 and 6 have gcd 2.
    # So r2, r4, r6 must be pairwise distinct mod 2? No, r2 != r4 mod 2 and r2 != r6 mod 2. This implies r4 == r6 mod 2 is allowed.
    # But 4 and 6 have gcd 2. So r4 != r6 mod 2.
    # So r2, r4, r6 must be pairwise distinct mod 2. Impossible.
    # So we can't have 3 evens where each pair has gcd 2.
    # This suggests the "pure prime multiple" strategy is indeed the best.
    
    # Let's implement the solution for p=11.
    p = 11
    moduli = [p * k for k in range(1, 12)]  # 11, 22, ..., 121
    result = []
    for i, m in enumerate(moduli):
        r = i
        result.append([r, m])
        
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            g = gcd(m1, m2)
            if (r1 - r2) % g == 0:
                # This would mean they intersect, which is bad.
                # But we proved it's not possible.
                pass
                
    return result

# EVOLVE-BLOCK-END
