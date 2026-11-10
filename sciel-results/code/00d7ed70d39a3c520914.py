import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 500.0  # Safety margin
    
    p = 7
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element represented as (a, b) for a + b*alpha
    
    # Precompute field operations
    # a + b*alpha
    # (a1 + b1*a) * (a2 + b2*a) = a1*a2 + (a1*b2 + a2*b1)*a + b1*b2*a^2
    # = a1*a2 + 3*b1*b2 + (a1*b2 + a2*b1)*a
    def field_mul(a1, b1, a2, b2):
        ra = (a1 * a2 + 3 * b1 * b2) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra, rb
    
    def field_add(a1, b1, a2, b2):
        return (a1 + a2) % p, (b1 + b2) % p
    
    def field_neg(a, b):
        return (-a) % p, (-b) % p
    
    # Squares in F_49
    squares = set()
    for a in range(p):
        for b in range(p):
            sa, sb = field_mul(a, b, a, b)
            squares.add((sa, sb))
            
    is_square = lambda a, b: (a, b) in squares

    # Parabola S0: y = x^2
    # x = (xa, xb), y = (ya, yb)
    S0 = set()
    for xa in range(p):
        for xb in range(p):
            ya, yb = field_mul(xa, xb, xa, xb)
            S0.add(((xa, xb), (ya, yb)))
            
    # Current S
    S = set(S0)
    
    # Private lines for points in S
    # For x in S0, private line is tangent.
    # Tangent at x: y = 2*x*X - x^2
    # We don't need to store lines for S0 explicitly if we just check "not on tangent"
    # Condition for P=(u,v) to not be on any tangent of S0:
    # v = 2*x*u - x^2 => x^2 + 2*u*x - v = 0
    # Discriminant D = (2u)^2 + 4v = 4(u^2+v)
    # P is on a tangent iff u^2+v is a square.
    # So P is a candidate iff u^2+v is NOT a square.
    
    candidates = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    P = ((xa, xb), (ya, yb))
                    if P in S:
                        continue
                    # Check if P is on any tangent of S0
                    u = (xa, xb)
                    v = (ya, yb)
                    u2a, u2b = field_mul(xa, xb, xa, xb)
                    sum_a, sum_b = field_add(u2a, u2b, ya, yb)
                    if not is_square(sum_a, sum_b):
                        candidates.append(P)
                        
    # Store private lines for added points
    # Line represented as (slope_m, intercept_c) for y = m*x + c
    # Or vertical line x = k
    # We will store the line points to check disjointness? No, too big.
    # Store (type, params)
    # type 0: vertical x = k (k is (a,b))
    # type 1: non-vertical y = m*x + c
    
    private_lines = []
    
    # Sort candidates? Maybe by "density" of non-squares?
    # Just iterate.
    
    for P in candidates:
        if time.time() > deadline:
            break
            
        u = P[0]
        v = P[1]
        
        # Check if P is on any existing private line
        conflict = False
        for line in private_lines:
            if line[0] == 0: # vertical x = k
                if u == line[1]:
                    conflict = True
                    break
            else: # y = m*x + c
                # Check if v = m*u + c
                m = line[1]
                c = line[2]
                mu_a, mu_b = field_mul(m[0], m[1], u[0], u[1])
                res_a, res_b = field_add(mu_a, mu_b, c[0], c[1])
                if res_a == v[0] and res_b == v[1]:
                    conflict = True
                    break
        
        if conflict:
            continue
            
        # Find a line through P disjoint from S
        # Try all slopes m
        found_line = None
        for ma in range(p):
            for mb in range(p):
                m = (ma, mb)
                # Line: y - v = m * (x - u) => y = m*x + (v - m*u)
                # c = v - m*u
                mu_a, mu_b = field_mul(ma, mb, u[0], u[1])
                c_a, c_b = field_add(v[0], v[1], field_neg(mu_a, mu_b))
                c = (c_a, c_b)
                
                # Check if this line is disjoint from S
                # 1. Disjoint from S0?
                # Intersection with y=x^2: x^2 = m*x + c => x^2 - m*x - c = 0
                # D = m^2 + 4c
                m2a, m2b = field_mul(ma, mb, ma, mb)
                # 4c = 2c + 2c
                c2a, c2b = field_add(c_a, c_b, c_a, c_b)
                c4a, c4b = field_add(c2a, c2b, c2a, c2b)
                Da, Db = field_add(m2a, m2b, c4a, c4b)
                if is_square(Da, Db):
                    continue # Intersects S0
                
                # 2. Disjoint from added points?
                # Check if any added point Q is on this line
                # We only need to check if the line passes through any Q in S \ S0
                # But we don't have a list of Qs easily accessible in a set for fast lookup?
                # We can store S_added = set(S) - S0
                # Actually, we can just check if the line contains any point in S_added
                # But S_added is small.
                # Let's maintain a set of added points.
                
                # Optimization: Store added points in a list
                # Check line against all added points
                ok = True
                for Q in S_added_points:
                    # Check if Q is on y = m*x + c
                    qx = Q[0]
                    qy = Q[1]
                    mq_a, mq_b = field_mul(ma, mb, qx[0], qx[1])
                    res_a, res_b = field_add(mq_a, mq_b, c_a, c_b)
                    if res_a == qy[0] and res_b == qy[1]:
                        ok = False
                        break
                if ok:
                    found_line = (1, m, c)
                    break
            if found_line:
                break
                
        if found_line:
            S.add(P)
            S_added_points.append(P)
            private_lines.append(found_line)

    # Construct N
    N = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    P = ((xa, xb), (ya, yb))
                    if P not in S:
                        N.append([[xa, xb], [ya, yb]])
                        
    return np.array(N, dtype=np.int64)

# Helper to hold added points
S_added_points = []
