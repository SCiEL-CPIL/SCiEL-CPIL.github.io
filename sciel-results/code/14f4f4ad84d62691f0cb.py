<<<<<<< SEARCH
def _disjoint(r1, m1, r2, m2):
    diff = r1 - r2
    if diff == 0:
        return False  # same residue and same modulus would be identical; but moduli are distinct
    # Actually if diff==0 but m1 != m2, gcd(0, m1*m2) = m1*m2 > 1, so they ARE disjoint
    # Wait: gcd(0, n) = n. So if r1 == r2, gcd(0, m1*m2) = m1*m2 > 1. Disjoint!
    # Let me recheck: two classes {r mod m1} and {r mod m2}. Do they intersect?
    # x ≡ r (mod m1) and x ≡ r (mod m2) has a solution iff gcd(r-r, m1*m2) | (r-r), i.e., gcd(0, m1*m2) | 0, which is always true.
    # So they DO intersect! The CRT condition: x ≡ a (mod m), x ≡ b (mod n) has a solution iff gcd(a-b, m*n) divides... 
    # Actually the standard CRT: x ≡ a (mod m), x ≡ b (mod n) has a solution iff a ≡ b (mod gcd(m,n)).
    # So if r1 == r2, then r1 ≡ r2 (mod gcd(m1,m2)) always holds, so they DO intersect.
    # So gcd(r1-r2, m1*m2) > 1 is NOT the correct criterion!
    # The correct criterion: they are disjoint iff r1 ≢ r2 (mod gcd(m1,m2)).
    # Equivalently: gcd(m1,m2) does not divide (r1-r2).
    # Let me re-read the task: "the verifier uses the exact gcd criterion"
    # I think the criterion is: disjoint iff gcd(r1-r2, lcm(m1,m2)) > 1? No...
    # Standard: {r1 mod m1} ∩ {r2 mod m2} ≠ ∅ iff r1 ≡ r2 (mod gcd(m1,m2)).
    # So disjoint iff r1 ≢ r2 (mod gcd(m1,m2)), i.e., gcd(m1,m2) ∤ (r1-r2).
    # Equivalently: gcd(r1-r2, m1) != gcd(m1,m2) or gcd(r1-r2, m2) != gcd(m1,m2)...
    # Simpler: let g = gcd(m1,m2). Disjoint iff (r1-r2) % g != 0.
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0
=======
def _disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0
>>>>>>> REPLACE

Now let me also add a local search / improvement step and better heuristics:

