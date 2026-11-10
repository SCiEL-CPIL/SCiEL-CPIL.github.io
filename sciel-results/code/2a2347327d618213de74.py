import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    We define N as the complement of S, where S is a subset of the Hermitian unital U.
    To maximize the score (size of S), we want S to be as large as possible while keeping N valid.
    The current best uses S = { (x,y) in U : y_a not in {1,2,4} }.
    We will try to expand S by reducing the "Special" set (y-values kept in N).
    We test progressively smaller Special sets until we find one that yields a valid Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3 is a quadratic non-residue mod 7

    # Precompute field operations
    # Elements are (a, b) representing a + b*alpha
    # Add: (a1+b1, a2+b2) mod p
    # Mult: (a1+b1*alpha)(a2+b2*alpha) = a1a2 + b1b2*alpha^2 + (a1b2+a2b1)alpha
    #      = (a1a2 + k*b1b2, a1b2+a2b1) mod p

    def mul(v1, v2):
        a1, b1 = v1
        a2, b2 = v2
        a = (a1 * a2 + k * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)

    def add(v1, v2):
        return ((v1[0] + v2[0]) % p, (v1[1] + v2[1]) % p)

    def sub(v1, v2):
        return ((v1[0] - v2[0]) % p, (v1[1] - v2[1]) % p)

    def neg(v):
        return ((-v[0]) % p, (-v[1]) % p)

    def inv(v):
        # Inverse of a + b*alpha is (a - b*alpha) / (a^2 - k*b^2)
        a, b = v
        norm = (a * a - k * b * b) % p
        if norm == 0:
            return None
        inv_norm = pow(norm, p - 2, p)
        return ((a * inv_norm) % p, (-b * inv_norm) % p)

    # List all elements of F_49
    F = [(a, b) for a in range(p) for b in range(p)]
    zero = (0, 0)
    one = (1, 0)

    # Identify the Unital U: Tr(x) + Nm(y) = 0
    # Tr(x) = x + x^7 = 2*xa (since x^7 = xa + xb*alpha^7 = xa + xb*(alpha^2)^3*alpha = xa + xb*27*alpha = xa + xb*6*alpha? No.
    # alpha^2=3. alpha^3=3alpha. alpha^4=9=2. alpha^5=2alpha. alpha^6=6. alpha^7=6alpha.
    # Wait. alpha^7 = alpha*(alpha^6) = alpha*6 = 6alpha.
    # So x^7 = xa + xb*6alpha.
    # Tr(x) = x + x^7 = (xa+xa) + (xb+6xb)alpha = 2xa + 0xb alpha = 2xa.
    # Nm(y) = y * y^7 = (ya+yb alpha)(ya+6yb alpha) = ya^2 + 6ya yb alpha + ya yb alpha + 6 yb^2 alpha^2
    # = ya^2 + 7 ya yb alpha + 18 yb^2 = ya^2 + 0 + 4 yb^2 (since 18 mod 7 = 4).
    # So Nm(y) = ya^2 + 4 yb^2.
    # Condition: 2*xa + ya^2 + 4*yb^2 == 0 (mod p)

    U_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    val = (2 * xa + ya * ya + 4 * yb * yb) % p
                    if val == 0:
                        U_points.append(((xa, xb), (ya, yb)))

    # Candidate Special sets (y-values to keep in N)
    # We want to minimize Special to maximize S = U \ Special.
    # S = { (x,y) in U : y not in Special }
    # N = F^2 \ S
    # Validity: For every x in N, there is a line L through x with L\{x} subset N.
    # Lines are defined by v=(1,m) or v=(0,1).
    
    # Let's try different Special sets.
    # 1. Empty Special set. S = U. N = F^2 \ U.
    # 2. Small Special sets.

    def is_valid_nikodym(N_set, U_points, F, zero, one, add, sub, mul, inv, p, k, deadline):
        """
        Check if N_set is a valid Nikodym set.
        N_set is a set of points (x, y) where x, y are tuples (a, b).
        """
        # Precompute slopes m in F
        # For each x in N, we need to find a direction v such that x + t*v in N for all t != 0.
        # v can be (0,1) or (1,m).
        
        # To speed up, we can precompute for each point in N, the set of points on lines through it?
        # No, we check existence.
        
        # Optimized check:
        # For each x in N:
        #   Check if there exists m in F such that for all t in F \ {0}, x + t*(1,m) in N.
        #   OR if for all t in F \ {0}, x + t*(0,1) in N.
        
        # We can iterate x in N.
        # For each x, iterate m.
        # For each m, check t.
        
        # This is O(|N| * |F| * |F|) = O(2401 * 49 * 49) ~ 5.7M operations.
        # In Python, this might be slow. 5.7M * (cost of check).
        # Cost of check: tuple add, set lookup.
        # 5.7M * 1us = 5.7s. Acceptable.
        
        N_set = set(N_set)
        
        for x in N_set:
            found = False
            # Check vertical line v=(0,1)
            ok = True
            for t in F:
                if t == zero: continue
                y_pt = add((0,0), t) # t*(0,1) = (0, t)
                pt = add(x, y_pt)
                if pt not in N_set:
                    ok = False
                    break
            if ok:
                found = True
            else:
                # Check lines with slope m
                for m in F:
                    ok = True
                    for t in F:
                        if t == zero: continue
                        # v = (1, m)
                        # t*v = t*1, t*m
                        tx = t
                        ty = mul(t, m)
                        pt = add(x, add(tx, ty))
                        if pt not in N_set:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                    if time.time() > deadline - 5:
                        return False # Timeout safety
            if found:
                continue
            else:
                return False
        return True

    # Try Special sets
    # Special set is a set of y-values (tuples)
    
    # Candidate 1: Special = {}
    # S = U
    # N = F^2 \ U
    
    # Candidate 2: Special = { y : ya == 0 }
    # Candidate 3: Special = { y : ya == 1 }
    
    # Let's start with the most aggressive (empty Special) and see if it's valid.
    # If valid, score is max.
    
    # Note: The current best uses Special = { y : ya in {1,2,4} }.
    # This implies S = { y : ya not in {1,2,4} } = { y : ya in {0,3,5,6} }.
    # Size of S = 4/7 * 343 = 196.
    
    # If we use Special = {}, S = U, size 343.
    # Is N = F^2 \ U a Nikodym set?
    # This is a known question. For q=49, is the complement of a Hermitian unital a Nikodym set?
    # Generally, the complement of an unital is NOT a Nikodym set because the unital blocks all lines?
    # No, a line intersects a unital in 1 or q points.
    # If a line intersects U in q points, it contains no points of N (except maybe 1 if tangent? No, q points means secant).
    # If a line intersects U in 1 point, it contains q-1 points of N.
    # For x in N, we need a line through x with NO other points in U?
    # i.e., a line through x that intersects U in exactly 1 point (which is x? No, x is in N, so x not in U).
    # So the line must intersect U in 0 points? Impossible, unital intersects every line in 1 or q points.
    # Wait. Unital U: every line meets U in 1 or q points.
    # If a line L meets U in q points, then L is contained in U? No, U is not a line.
    # If L meets U in q points, then L \ U is empty. So L is not a valid line for any x in N.
    # If L meets U in 1 point, say P. Then L \ {P} is in N.
    # So for x in N, we need a line L through x such that L \ {x} subset N.
    # This is equivalent to L \ U subset N.
    # If L meets U in 1 point P != x, then L \ {x} contains P? No.
    # L \ {x} = (L \ U) U (U \ {x}).
    # If P is in U and P != x, then P is in L \ {x}. But P is not in N.
    # So L \ {x} is NOT subset of N.
    # Therefore, we need a line L through x such that L \ U is empty? No.
    # We need L \ {x} subset N.
    # This implies L \ U subset N (true) and (L \ {x}) \ (L \ U) = L \ (U U {x}) is empty?
    # Basically, L can contain at most one point of U, and if it does, it must be x?
    # But x is in N, so x not in U.
    # So L must contain NO points of U.
    # But every line meets U in at least 1 point.
    # Thus, NO line L through x in N can have L \ {x} subset N.
    # So N = F^2 \ U is NEVER a Nikodym set.
    
    # So Special cannot be empty.
    
    # We need S to be such that for every x in N, there is a line L through x with L \ S subset N.
    # i.e., L \ S does not contain any points of S?
    # L \ {x} subset N means L \ {x} is disjoint from S.
    # So L \ {x} subset F^2 \ S = N.
    # This means L \ {x} contains no points of S.
    # So L \ S contains no points of S? Trivial.
    # It means L \ {x} is a subset of N.
    # So L \ S = (L \ {x}) \ S U (S \ {x})? No.
    # L \ {x} subset N means for all y in L, y != x => y not in S.
    # So L \ {x} is disjoint from S.
    # This implies L \ S = L \ {x} U (S \ {x})? No.
    # It implies that S \ L is empty? No.
    # It implies that S \ (L \ {x}) = S \ L U (S \ {x})?
    # Simply: S \ L must be empty? No.
    # S \ (L \ {x}) = S \ L U (S \ {x})?
    # Condition: (L \ {x}) \ S = empty.
    # This is equivalent to L \ {x} subset S^c = N.
    # So L can intersect S only at x? But x not in S.
    # So L must be disjoint from S.
    # Wait. If L is disjoint from S, then L subset N.
    # Then for any x in L, L \ {x} subset N.
    # So if there exists a line L disjoint from S, then every point in L is "covered" by L.
    # But we need EVERY x in N to be covered by SOME line.
    # So for every x in N, there must be a line L_x through x such that L_x is disjoint from S.
    
    # So the condition is: The set of points not in N (which is S) must not "block" any point x in N.
    # A point x is blocked if every line through x intersects S at some point other than x.
    # Since x not in S, "intersects S" means intersects S non-trivially.
    # So x is blocked if for every line L through x, L \ {x} intersects S.
    
    # We want to maximize S such that no point in N is blocked.
    
    # Let's try Special = { y : ya == 0 }.
    # S = { (x,y) in U : ya != 0 }.
    
    # Let's test this.
    
    special_candidates = [
        set(), # Should fail
        { (0, b) for b in range(p) }, # ya == 0
        { (1, b) for b in range(p) }, # ya == 1
        { (ya, b) for ya in [1,2] for b in range(p) }, # ya in {1,2}
    ]
    
    # We know ya in {1,2,4} works.
    # Let's try ya in {1,2}.
    # If that works, try ya in {1}.
    
    best_N = None
    best_S_size = -1
    
    for special in special_candidates:
        # Construct S
        S = set()
        for pt in U_points:
            x, y = pt
            if y not in special:
                S.add(pt)
        
        # Construct N
        all_points = { ((xa, xb), (ya, yb)) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p) }
        N = all_points - S
        
        # Check validity
        # To save time, we can do a quick check or just run the full check if time permits.
        # We have ~590s.
        # Each check is ~6s.
        # We can run a few.
        
        if time.time() > deadline - 30:
            break
            
        if is_valid_nikodym(N, U_points, F, zero, one, add, sub, mul, inv, p, k, deadline):
            if len(S) > best_S_size:
                best_S_size = len(S)
                best_N = N
        
        # If we find a valid one with larger S, we can stop if we are confident?
        # No, we want max S.
        # But the list is ordered by increasing Special size (decreasing S size? No).
        # Special = {} -> S = U (Max S).
        # Special = {ya=0} -> S = U \ {ya=0} (Smaller S).
        # So we are testing from Max S to Smaller S.
        # The first valid one is the best!
        
        if best_N is not None:
            break

    # If no valid N found (should not happen, as current best works), fallback to current best
    if best_N is None:
        # Fallback to known good: Special = { y : ya in {1,2,4} }
        special = { (ya, b) for ya in [1,2,4] for b in range(p) }
        S = set()
        for pt in U_points:
            x, y = pt
            if y not in special:
                S.add(pt)
        all_points = { ((xa, xb), (ya, yb)) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p) }
        best_N = all_points - S

    # Convert best_N to array
    points = []
    for pt in best_N:
        x, y = pt
        points.append([[x[0], x[1]], [y[0], y[1]]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
