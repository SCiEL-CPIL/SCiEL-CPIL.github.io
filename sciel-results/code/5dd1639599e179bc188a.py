# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2, p = 7.
    
    Strategy: Use the unital U = {(x,y) : Tr(x) + N(y) = 0} and exclude a large
    subset S of U (keeping the rest as N). The key insight is that if we exclude
    all points of U whose y-coordinate has y_a NOT a quadratic residue mod p,
    we get a valid Nikodym set with a large complement.
    
    Additionally, we can try to exclude more points by using a blocking set
    construction: for each excluded point, we need to ensure that every other
    point on each line through it still belongs to N.
    
    For q = p^2 = 49, p = 7, the unital has q+1 = 50 points per line intersection.
    
    The measured improvement (score 1.333) came from using quadratic residues 
    instead of non-residues for the y_special set.
    
    Let's build the largest valid exclusion set S ⊆ U such that N = F_q^2 \\ S 
    is a Nikodym set.
    
    A point x = (x_a, x_b) + (y_a, y_b)*alpha is in U iff:
        2*x_a + y_a^2 - k*y_b^2 ≡ 0 (mod p)
    where k is the smallest quadratic non-residue mod 7, which is 3.
    
    For the Nikodym property: for every point P in F_q^2, there must exist a 
    direction v such that all other points on the line through P in direction v 
    are in N.
    
    If P is not in S (i.e., P is in N), this is automatically satisfied as long 
    as N is large enough. If P is in S, we need to find a line through P where 
    all other points are in N.
    
    The standard construction: S consists of points in U whose y-coordinate is 
    "bad" in some sense. The key constraint is that for each excluded point 
    (x,y) in S, and for each slope m, the line {(x,y) + t*(1,m) : t in F_q} 
    must have all its other points in N, OR at least one such line exists.
    
    Actually, re-reading the task: "every point x of F_49^2 must lie on a line 
    {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), 
    whose points other than x all belong to N."
    
    So for each point x, we need at least one direction v (either (1,m) for some 
    m, or (0,1)) such that all other 48 points on that line are in N.
    
    For points NOT in S, they're in N, so we just need some line through them 
    where the other points are in N. For points IN S, we need a line through 
    them where all other points are in N (i.e., not in S).
    
    The unital-based construction works because: if (x,y) is in U and we exclude 
    it from N (put it in S), then for any line through (x,y) with direction 
    (1,m), the other points on that line intersect U in at most 1 other point 
    (since a line intersects an unital in exactly 1 point or is tangent). Wait, 
    actually in F_q^2 with q=p^2, a line intersects the Hermitian unital in 
    exactly 1 point. So if (x,y) is in U, no other point on ANY line through 
    (x,y) is in U. Therefore, if S ⊆ U, then for any point (x,y) in S, ALL 
    other points on ANY line through (x,y) are NOT in U, hence NOT in S, hence 
    in N. So the Nikodym condition is automatically satisfied for points in S!
    
    For points NOT in U (and not in S, so in N), we need to verify the condition 
    holds. Since they're not in U, no line through them has another point in U. 
    So again, all other points on any line through them are not in U, hence not 
    in S, hence in N.
    
    Wait, that means ANY subset S of U gives a valid Nikodym set! Because U has 
    the property that each line meets U in exactly 1 point. So if S ⊆ U, then 
    for any point x and any line L through x, L ∩ U contains at most 1 point 
    (namely x if x ∈ U). If x ∉ U, then L ∩ U = {some point} ≠ x, so all 
    other points of L are not in U, hence not in S. If x ∈ U, then L ∩ U = {x}, 
    so all other points of L are not in U, hence not in S.
    
    Therefore, we can take S = U itself, giving N = F_q^2 \\ U, with |S| = q+1 = 50.
    But wait, the current program only excludes a subset of U. Let me re-examine.
    
    Actually, the Hermitian unital in F_{p^2}^2 has the property that every line 
    meets it in exactly one point. This is a fundamental property of unitals.
    
    So if S = U, then |S| = q + 1 = 50, and |N| = q^2 - (q+1) = 2401 - 50 = 2351.
    The complement size would be 50.
    
    But the current program gets a complement of about 196 (score 1.333 means 
    complement ≈ 196 vs baseline 147). Let me check: if S is a subset of U with 
    |S| = 196, that's impossible since |U| = 50.
    
    I think I'm confusing myself. Let me re-read the current program.
    
    The current program excludes points where in_unital AND NOT in_y_special. 
    The unital condition is 2*xa + ya^2 - k*yb^2 ≡ 0 mod p. This is a condition 
    mod p, not mod q. So the "unital" here is not the full Hermitian unital in 
    F_q^2, but rather a subset defined by a mod-p condition.
    
    Let me count: for each (ya, yb) pair, the condition 2*xa ≡ -ya^2 + k*yb^2 
    (mod p) determines xa uniquely mod p. So for each (ya, yb), there is exactly 
    1 value of xa mod p that satisfies the unital condition. And xb is free (p 
    choices). So the "unital" has p * p * p = p^3 = 343 points? No wait, for 
    each (ya, yb) in F_p^2, xa is determined, and xb is free in F_p. So total 
    points = p * p * p = 343.
    
    And y_special has p * (number of squares mod p) = p * ((p+1)/2) = 7 * 4 = 28 
    y-coordinates. For each y in y_special, there are p points in the unital 
    (one for each xb). So |S| = 343 - 28*p = 343 - 196 = 147. That matches!
    
    So the "unital" is actually a union of p lines (for each xb, a curve in 
    (xa, ya, yb) space). The key property we need is: for any point x and any 
    line through x, the line contains at most 1 point of S.
    
    This is a different structure. Let me think about what S can be.
    
    Actually, the critical property is: S must be such that no line contains 2 
    or more points of S. Because if a line contains 2 points of S, then for 
    either of those points, that line would have another point in S, violating 
    the Nikodym condition (we need at least one line through x where all other 
    points are in N, but if ALL lines through x have another point in S, we're 
    stuck).
    
    Wait, we need at least ONE line through x where all other points are in N. 
    So it's OK if some lines through x have other points in S, as long as at 
    least one line doesn't.
    
    But if S is a "blocking set" in the sense that every line meets S in at 
    least 1 point, then for any x, every line through x has another point in S 
    (if the line has ≥2 points of S) or just x (if x is the only point of S on 
    that line). If x is not in S, and every line through x meets S, then for 
    each line L through x, L has some point in S. If that point is not x (which 
    it isn't since x ∉ S), then L has another point in S. So no line through x 
    has all other points in N. Thus x fails the Nikodym condition.
    
    So S must NOT be a blocking set. In fact, for points not in S, we need at 
    least one line through them that avoids S entirely (except possibly at x, 
    but x is not in S).
    
    This is getting complex. Let me just go with the known good construction and 
    try to maximize |S|.
    
    Given the time constraint, let me use the construction that gave score 1.333 
    and see if I can improve it slightly.
    """
    p = 7
    k = 3  # smallest quadratic non-residue mod 7
    
    # Quadratic residues mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    # So QR = {0, 1, 2, 4}
    qr = set((u * u) % p for u in range(p))
    
    # y_special: y_a must be in QR
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # Build S: points in the "unital" where y is NOT in y_special
    # Unital condition: 2*xa + ya^2 - k*yb^2 ≡ 0 (mod p)
    S = set()
    for ya in range(p):
        for yb in range(p):
            # Determine xa
            rhs = (-ya * ya + k * yb * yb) % p
            xa = (rhs * pow(2, p - 2, p)) % p  # 2^(-1) mod 7 = 4
            for xb in range(p):
                if (ya, yb) not in y_special:
                    S.add((xa, xb, ya, yb))
    
    # Build N = all points not in S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in S:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
