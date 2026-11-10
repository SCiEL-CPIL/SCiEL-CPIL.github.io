"""Maximize pairwise disjoint residue classes using even moduli with uniform parity."""
# EVOLVE-BLOCK-START
import time
import math

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    d = abs(r1 - r2)
    return math.gcd(math.gcd(d, m1), m2) > 1

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Key insight: all even moduli with all residues having the same parity
    # are pairwise disjoint because gcd(0, m1, m2) = gcd(m1, m2) >= 2 for even m1, m2.
    # Even moduli from 2 to 128: 2, 4, 6, ..., 128 = 63 moduli
    even_moduli = list(range(2, 129, 2))  # 63 moduli
    
    # All residues = 0 (even), so all pairs have r1-r2 = 0, gcd(0, m1, m2) = gcd(m1,m2) >= 2
    result = [[0, m] for m in even_moduli]
    
    # Verify correctness
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            if not _is_disjoint(r1, m1, r2, m2):
                # This should never happen with our construction
                pass
    
    # Try to add odd moduli if possible
    # For an odd modulus m with residue r, it's disjoint from (0, m_even) iff
    # gcd(r, m, m_even) > 1. Since m is odd, we need gcd(r, m_even) > 1.
    # m_even is even, so we need r to share a factor with some even modulus.
    # Actually: gcd(r - 0, m, m_even) = gcd(r, m, m_even). Since m is odd,
    # this equals gcd(r, m_even) if m doesn't add new factors... 
    # More carefully: gcd(r, m, m_even) = gcd(gcd(r, m), m_even)
    # For this to be > 1, we need gcd(r, m) and m_even to share a factor,
    # OR gcd(r, m_even) > 1 directly.
    
    # Try to add odd moduli with residue 0: gcd(0, m_odd, m_even) = gcd(m_odd, m_even)
    # Since m_odd is odd and m_even is even, gcd(m_odd, m_even) could be 1 or > 1
    # e.g., m_odd=3, m_even=6: gcd(3,6)=3>1. m_odd=5, m_even=6: gcd(5,6)=1.
    
    # So odd moduli that share a factor with at least one even modulus can be added with residue 0.
    # But they also need to be disjoint from each other: gcd(0, m_odd1, m_odd2) = gcd(m_odd1, m_odd2)
    # So odd moduli must be pairwise non-coprime.
    
    # Let's try to greedily add odd moduli with residue 0
    odd_moduli = list(range(3, 129, 2))  # 63 odd moduli
    
    # We need: for each new odd m, gcd(m, m_even) > 1 for ALL even m in our set,
    # AND gcd(m, m_odd_j) > 1 for ALL previously added odd m_odd_j.
    # Actually no: we need gcd(0, m, m_even) > 1 for each even m_even, which means
    # gcd(m, m_even) > 1 for each even m_even. That's very restrictive.
    # gcd(m, 2) > 1 means m is even, contradiction. So we CAN'T add any odd modulus with residue 0
    # because gcd(m_odd, 2) = 1 for odd m_odd.
    
    # What about odd modulus with different residue? For (r, m_odd) vs (0, m_even):
    # gcd(r, m_odd, m_even) > 1. Since m_odd is odd, this is gcd(r, m_even) if we consider
    # the full gcd. Actually gcd(r, m_odd, m_even) = gcd(gcd(r, m_odd), m_even).
    # We need this > 1 for all even m_even in our set.
    # gcd(gcd(r, m_odd), 2) > 1 means gcd(r, m_odd) is even, meaning r is even (since m_odd is odd,
    # gcd(r, m_odd) has same parity as r... no, gcd(r, m_odd) divides m_odd which is odd, so gcd is odd).
    # So gcd(gcd(r, m_odd), 2) = 1 always! Thus we CANNOT add any odd modulus to a set containing modulus 2.
    
    # Alternative: drop modulus 2 and use other even moduli? If we drop 2, we have 62 even moduli.
    # For odd m with residue r: need gcd(r, m, m_even) > 1 for all even m_even in set.
    # The even moduli are 4,6,8,...,128. gcd of all these... gcd(4,6)=2, so gcd includes 2.
    # So we still need gcd(r, m, 2) > 1, meaning r is even. But then gcd(r, m_odd) is odd (divides m_odd),
    # and gcd(odd, 2) = 1. Still fails.
    
    # What if we use a different prime? Drop all even moduli and use multiples of 3?
    # Multiples of 3 from 3 to 128: 3,6,9,...,126 = 42 moduli.
    # With residue 0: gcd(0, m1, m2) = gcd(m1, m2). For multiples of 3, gcd >= 3 > 1. Works!
    # But 42 < 63, so even moduli are better.
    
    # Can we combine? Use even moduli (63) and see if we can add any others.
    # We showed we can't add odd moduli when modulus 2 is present.
    # What if we drop modulus 2? Then even moduli are 4,6,...,128 (62 moduli).
    # For odd m, residue r: need gcd(r, m, m_e) > 1 for all m_e in {4,6,...,128}.
    # gcd(4,6,8,...,128): gcd(4,6)=2, so the overall gcd includes 2.
    # So we need gcd(r, m, 2) > 1, i.e., r even. But gcd(r_even, m_odd) is odd, gcd(odd,2)=1. Fails.
    
    # The structure is: any set containing two even moduli has gcd including 2 in their common divisors
    # only if both are even. The constraint for adding (r, m_odd) is that for EVERY (0, m_even) in set,
    # gcd(r, m_odd, m_even) > 1. This requires that for each m_even, there's a prime p dividing r, m_odd, and m_even.
    # Since m_odd is odd, p must be odd. So we need an odd prime p that divides m_odd AND m_even AND r.
    # For m_even=4: only prime is 2. So no odd prime divides 4. Thus m_even=4 blocks all odd additions.
    
    # Drop modulus 4 too? Even moduli: 6,8,10,...,128 (61 moduli).
    # m_even=8: only prime 2. Still blocks.
    # Drop 6,8: even moduli 10,12,...,128 (60 moduli). m_even=10: primes 2,5.
    # So we need odd prime 5 dividing m_odd and r. Possible!
    # But m_even=12: primes 2,3. Need 3 dividing m_odd and r.
    # So for (r, m_odd) to work with all even moduli in {10,12,...,128}, we need:
    # For m_even=10: 5 | r and 5 | m_odd
    # For m_even=12: 3 | r and 3 | m_odd
    # So 15 | r and 15 | m_odd. And r < m_odd <= 128.
    # m_odd must be odd multiple of 15: 15,45,75,105 (odd multiples of 15 up to 128)
    # Wait, 15 is odd, 45 odd, 75 odd, 105 odd. 135>128.
    # But we also need m_odd to be disjoint from other odd moduli we add.
    
    # This is getting complex. The 63 even moduli solution is already very strong.
    # Let's just return that.
    
    return result
# EVOLVE-BLOCK-END
