"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _is_disjoint(r1, m1, r2, m2):
    """Check if residue class r1 mod m1 is disjoint from r2 mod m2."""
    # Two classes {x : x ≡ r1 (mod m1)} and {x : x ≡ r2 (mod m2)} are disjoint
    # iff r1 - r2 is NOT divisible by gcd(m1, m2)
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    
    Two residue classes (r1, m1) and (r2, m2) are disjoint iff r1 - r2 is not divisible by gcd(m1, m2).
    
    Strategy: Use a greedy approach with backtracking-like heuristics.
    We want to pick as many (r, m) pairs as possible where all moduli are distinct (in 2..128)
    and the classes are pairwise disjoint.
    
    Key insight: If we choose moduli that are pairwise coprime, then gcd(m_i, m_j) = 1 for all i≠j,
    which means (r_i - r_j) % 1 = 0 always, so they would NEVER be disjoint. Wait, that's wrong.
    If gcd(m1,m2)=1, then the classes are disjoint iff r1 ≠ r2 mod 1, which is always true since
    there's only one residue class mod 1. So actually if gcd=1, the classes are ALWAYS disjoint
    (since any two classes mod coprime moduli intersect by CRT... wait no).
    
    Let me re-check: Classes A = {x : x ≡ r1 mod m1} and B = {x : x ≡ r2 mod m2}.
    They intersect iff there exists x such that x ≡ r1 (mod m1) and x ≡ r2 (mod m2).
    By CRT, this system has a solution iff r1 ≡ r2 (mod gcd(m1,m2)).
    So they are DISJOINT iff r1 ≠ r2 (mod gcd(m1,m2)).
    
    So if gcd(m1,m2) = 1, then r1 ≡ r2 (mod 1) is always true (everything is ≡ 0 mod 1),
    so the classes ALWAYS intersect. So pairwise coprime moduli are BAD.
    
    We want gcd(m1,m2) > 1 and r1 ≠ r2 (mod gcd(m1,m2)).
    
    Best approach: Choose moduli that share factors, and assign residues to be different mod the shared gcd.
    
    Let's try: pick moduli, then for each pair, ensure the residue difference is not 0 mod gcd.
    
    A strong heuristic: pick moduli that are all multiples of 2 (so gcd ≥ 2), and assign residues
    such that for any pair, their residues differ mod the gcd.
    
    Actually, a simpler powerful strategy: use moduli that are all even. Then gcd(m_i, m_j) ≥ 2.
    If we assign residues r_i = i (or some pattern) such that for any pair i,j, r_i ≠ r_j mod gcd(m_i,m_j).
    
    Even simpler: if all moduli are multiples of some prime p, and we assign residues r_i such that
    r_i mod p are all distinct, then for any pair, gcd(m_i,m_j) is a multiple of p, so
    r_i - r_j mod gcd is 0 only if r_i ≡ r_j mod p. If all r_i mod p are distinct, then
    r_i ≠ r_j mod p, hence r_i ≠ r_j mod any multiple of p, hence the classes are disjoint.
    
    So: pick a prime p, choose moduli all divisible by p, and assign residues with distinct values mod p.
    Maximize the count.
    
    For p=2: moduli even, residues alternate 0,1,0,1,... but we need distinct mod 2, so only 2 residues.
    That limits us to 2 classes. Not good.
    
    For p=3: moduli divisible by 3, residues 0,1,2 mod 3. Only 3 classes.
    
    Hmm, that's limited. We need a different approach.
    
    Let me think differently. We can use a greedy algorithm:
    1. Start with all moduli 2..128.
    2. For each modulus, try all residues 0..m-1.
    3. Greedily add (r,m) if it's disjoint from all already-chosen classes.
    4. To maximize, try different orderings or use a smarter heuristic.
    
    Actually, let's just do a greedy with a good ordering. Sort moduli in some order, and for each,
    pick the first residue that works.
    
    Let me implement a greedy that tries to maximize the count.
    """
    # Greedy approach: try all moduli in order, pick a valid residue if one exists
    # We can try multiple orderings and keep the best
    
    best_result = []
    
    def greedy(order):
        chosen = []  # list of (r, m)
        for m in order:
            # Try all residues 0..m-1
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if not _is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    break
        return chosen
    
    # Try different orderings
    moduli = list(range(2, 129))
    
    # Order 1: natural
    r1 = greedy(moduli)
    
    # Order 2: reversed
    r2 = greedy(moduli[::-1])
    
    # Order 3: sort by number of prime factors (desc) - more shared factors might be easier
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    r3 = greedy(sorted(moduli, key=lambda x: -num_prime_factors(x)))
    
    # Order 4: sort by value of smallest prime factor (asc) - share small primes
    def smallest_prime_factor(n):
        if n % 2 == 0:
            return 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return d
            d += 2
        return n
    
    r4 = greedy(sorted(moduli, key=lambda x: smallest_prime_factor(x)))
    
    # Pick the best
    candidates = [r1, r2, r3, r4]
    best = max(candidates, key=len)
    
    # Convert to list of [residue, modulus]
    return [[r, m] for (r, m) in best]

# EVOLVE-BLOCK-END
