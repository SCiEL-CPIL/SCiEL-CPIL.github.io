# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2. We aim to maximize the complement size
    (points NOT in N) while ensuring N is a valid Nikodym set.
    
    Strategy:
    1. Start with the baseline construction (unital-based) which gives a valid Nikodym set.
    2. Use a greedy heuristic to expand the complement S (excluded points) while maintaining
       the Nikodym property.
    
    The Nikodym property: For every point x in F_q^2, there exists a direction v (either
    (1, m) for some m in F_q, or (0, 1) for the vertical direction) such that all points
    on the line {x + t*v : t in F_q, t != 0} are in N.
    
    Equivalently: For every point x, there exists a line L through x (with one of the
    allowed directions) such that L \ {x} is contained in N, i.e., L \ {x} is disjoint
    from S.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element represented as (a, b) = a + b*alpha
    # alpha^2 = 3, so a^2 + 2ab*alpha + b^2*alpha^2 = a^2 + 3b^2 + 2ab*alpha
    # So (a,b)*(c,d) = (ac + 3bd, ad + bc) mod 7
    
    # Precompute field multiplication table
    # idx = a + b*p, where a, b in 0..6
    def to_idx(a, b):
        return a + b * p
    
    def from_idx(idx):
        a = idx % p
        b = idx // p
        return (a, b)
    
    # Multiplication table: mul[i][j] = i * j in F_49, result as idx
    mul = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        a1, b1 = from_idx(i)
        for j in range(q):
            a2, b2 = from_idx(j)
            # (a1 + b1*alpha) * (a2 + b2*alpha) = a1*a2 + (a1*b2 + b1*a2)*alpha + b1*b2*alpha^2
            # = (a1*a2 + 3*b1*b2) + (a1*b2 + b1*a2)*alpha
            res_a = (a1 * a2 + 3 * b1 * b2) % p
            res_b = (a1 * b2 + b1 * a2) % p
            mul[i, j] = to_idx(res_a, res_b)
    
    # Addition is component-wise mod p
    def add_f(i, j):
        a1, b1 = from_idx(i)
        a2, b2 = from_idx(j)
        return to_idx((a1 + a2) % p, (b1 + b2) % p)
    
    # Negation
    def neg_f(i):
        a, b = from_idx(i)
        return to_idx((-a) % p, (-b) % p)
    
    # Subtraction
    def sub_f(i, j):
        return add_f(i, neg_f(j))
    
    # Precompute all points in F_49^2
    # Point (x, y) where x, y in F_49, each represented as (a, b)
    # Total points: 49 * 49 = 2401
    # Index a point as x_idx * q + y_idx
    
    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    # Total directions: 49 + 1 = 50
    
    # For each point and each direction, precompute the line (set of points)
    # Line through point p0 = (x0, y0) with direction v = (vx, vy):
    #   points are (x0 + t*vx, y0 + t*vy) for t in F_49
    
    # Precompute line membership: for each direction d, for each point p,
    #   line[d][p] = set of points on the line through p with direction d
    
    # Actually, let's think differently. For the Nikodym condition:
    # For every point x, there exists a direction d such that all other points on
    # the line through x with direction d are in N (not in S).
    
    # So x is "covered" by direction d if (line_d(x) \ {x}) ∩ S = ∅
    
    # We want to maximize |S| such that every point is covered by at least one direction.
    
    # This is a hitting set / domination problem. Let's use a greedy approach:
    # Start with S = empty, then greedily add points to S as long as the Nikodym
    # property is maintained.
    
    # Actually, let's start with the baseline S and try to expand it.
    
    # Baseline: unital U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = a + (-a) = 2a (since char 7, -a = 6a, so a + 6a = 7a = 0... wait)
    # In F_49 = F_7[alpha], the conjugate of a + b*alpha is a - b*alpha = a + 4b*alpha (since -1 = 6 mod 7)
    # Trace(a + b*alpha) = (a + b*alpha) + (a - b*alpha) = 2a
    # Norm(a + b*alpha) = (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - 3b^2
    
    # So U = {(x, y) : 2*x_a + (y_a^2 - 3*y_b^2) = 0 mod 7}
    
    # The baseline excluded set S is U \ B where B is the blocking set.
    # Let's just start with S = U (the whole unital) and see if it's valid.
    
    # Check: is U a valid excluded set? For every point x, is there a direction d
    # such that line_d(x) \ {x} is disjoint from U?
    
    # Let's first build the baseline S and verify it.
    
    # Build unital U
    in_unital = np.zeros((q, q), dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x_idx = to_idx(xa, xb)
            for ya in range(p):
                for yb in range(p):
                    y_idx = to_idx(ya, yb)
                    # 2*xa + ya^2 - 3*yb^2 = 0 mod 7
                    if (2 * xa + ya * ya - 3 * yb * yb) % p == 0:
                        in_unital[x_idx, y_idx] = True
    
    # Baseline S = U
    S = in_unital.copy()
    
    # Verify the Nikodym property for this S
    def is_valid_nikodym(S):
        """Check if S is a valid excluded set (N = complement is a Nikodym set)."""
        # For each point x, check if there exists a direction d such that
        # all other points on line_d(x) are not in S.
        for x_idx in range(q):
            for y_idx in range(q):
                point_idx = x_idx * q + y_idx
                # Check all 50 directions
                found = False
                for d in range(50):
                    if d < q:
                        # Direction (1, m) where m = d
                        vx = 1
                        vy = d
                    else:
                        # Direction (0, 1)
                        vx = 0
                        vy = 1
                    
                    # Check all t != 0
                    ok = True
                    for t in range(1, q):
                        # Point: (x + t*vx, y + t*vy)
                        tx_idx = add_f(x_idx, mul[t, vx])
                        ty_idx = add_f(y_idx, mul[t, vy])
                        if S[tx_idx, ty_idx]:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if not found:
                    return False
        return True
    
    # The baseline should be valid. Let's try to expand S greedily.
    # For each point not in S, try adding it to S and check if the property still holds.
    # This is expensive, so let's be smart.
    
    # Key insight: When we add a point p to S, we only need to check points x for which
    # p lies on a line through x with some direction d, AND for which d was the ONLY
    # direction that worked for x.
    
    # Let's precompute for each point x, which directions work.
    def compute_working_directions(S):
        """For each point, return the set of directions that work."""
        working = [set() for _ in range(q * q)]
        for x_idx in range(q):
            for y_idx in range(q):
                point_idx = x_idx * q + y_idx
                for d in range(50):
                    if d < q:
                        vx = 1
                        vy = d
                    else:
                        vx = 0
                        vy = 1
                    
                    ok = True
                    for t in range(1, q):
                        tx_idx = add_f(x_idx, mul[t, vx])
                        ty_idx = add_f(y_idx, mul[t, vy])
                        if S[tx_idx, ty_idx]:
                            ok = False
                            break
                    if ok:
                        working[point_idx].add(d)
        return working
    
    # Initial working directions
    working = compute_working_directions(S)
    
    # Now greedily add points to S
    # A point p can be added if for every x such that p is on line_d(x) for some d,
    # x has at least one other working direction d' != d.
    
    # For efficiency, let's precompute for each point p, which (x, d) pairs have p on line_d(x)
    # p is on line_d(x) iff p = x + t*v_d for some t != 0
    # i.e., x = p - t*v_d = p + (-t)*v_d
    
    # Let's try a simpler greedy: for each candidate point, check if adding it breaks
    # the Nikodym property.
    
    candidates = []
    for x_idx in range(q):
        for y_idx in range(q):
            if not S[x_idx, y_idx]:
                candidates.append((x_idx, y_idx))
    
    # Shuffle candidates for randomness
    np.random.shuffle(candidates)
    
    improved = False
    for (cx, cy) in candidates:
        if time.time() > deadline - 30:
            break
        
        # Try adding (cx, cy) to S
        S[cx, cy] = True
        
        # Check if any point now has no working direction
        # Only need to check points x for which (cx, cy) is on some line_d(x)
        # i.e., x = (cx, cy) + t*(vx, vy) for some t != 0 and direction d
        affected_points = set()
        for d in range(50):
            if d < q:
                vx = 1
                vy = d
            else:
                vx = 0
                vy = 1
            
            for t in range(1, q):
                # x = (cx, cy) + t*(vx, vy)
                tx_idx = add_f(cx, mul[t, vx])
                ty_idx = add_f(cy, mul[t, vy])
                affected_points.add(tx_idx * q + ty_idx)
        
        # For each affected point, check if it still has a working direction
        valid = True
        for point_idx in affected_points:
            if len(working[point_idx]) == 0:
                valid = False
                break
            # Check if the working directions are still valid
            # A direction d works for x if no point on line_d(x) \ {x} is in S
            # Since we just added (cx, cy), we need to check if (cx, cy) is on
            # line_d(x) for any working direction d
            for d in list(working[point_idx]):
                if d < q:
                    vx = 1
                    vy = d
                else:
                    vx = 0
                    vy = 1
                
                # Check if (cx, cy) is on line_d(x) \ {x}
                # x + t*(vx, vy) = (cx, cy) for some t != 0
                # t*(vx, vy) = (cx - x, cy - y)
                dx = sub_f(cx, point_idx % q)
                dy = sub_f(cy, point_idx // q)
                
                # Check if (dx, dy) = t*(vx, vy) for some t != 0
                if vx == 0:
                    # Direction (0, 1): need dx = 0 and dy != 0
                    if dx == 0 and dy != 0:
                        working[point_idx].discard(d)
                        if len(working[point_idx]) == 0:
                            valid = False
                            break
                else:
                    # Direction (1, m): need dx != 0 and dy = dx * m
                    if dx != 0:
                        # t = dx (since vx = 1)
                        # Check if dy = dx * vy
                        expected_dy = mul[dx, vy]
                        if dy == expected_dy:
                            working[point_idx].discard(d)
                            if len(working[point_idx]) == 0:
                                valid = False
                                break
        
        if not valid:
            # Revert
            S[cx, cy] = False
        else:
            improved = True
    
    # Build the Nikodym set N = complement of S
    points = []
    for x_idx in range(q):
        for y_idx in range(q):
            if not S[x_idx, y_idx]:
                xa, xb = from_idx(x_idx)
                ya, yb = from_idx(y_idx)
                points.append([[xa, xb], [ya, yb]])
    
    if len(points) == 0:
        # Fallback: return all points
        for x_idx in range(q):
            for y_idx in range(q):
                xa, xb = from_idx(x_idx)
                ya, yb = from_idx(y_idx)
                points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
