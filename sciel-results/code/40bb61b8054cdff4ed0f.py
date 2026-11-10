# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    The baseline uses the Hermitian unital construction. Here, we use a
    much smaller excluded set S: the union of two vertical lines x=0 and x=1.
    
    A point x is in S iff x_a == 0 or x_a == 1 (where x_a is the a-component of x).
    
    We need to verify: for every point x in F_49^2, there exists a direction v
    (either v=(1,m) for some m, or v=(0,1)) such that the punctured line
    {x + t*v : t != 0} is fully contained in N (i.e., no other point on that
    line is in S).
    
    Key insight: If x is in S (x_a in {0,1}), we can choose v=(1,0). Then
    x + t*v has a-component x_a + t. For t != 0, x_a + t != x_a, so the
    a-component is not in {0,1} (since t is nonzero and x_a is 0 or 1,
    x_a+t is 1 or 0... wait, that's wrong).
    
    Let me reconsider. If x_a = 0 and v=(1,m), then x+t*v has a-component t.
    For t != 0, a-component = t. This is in {0,1} only if t=0 or t=1.
    So for t=1, the point is in S. So v=(1,m) doesn't work if t=1 gives a
    point in S.
    
    Actually, we need ALL t != 0 to give points NOT in S. So for x in S with
    x_a=0, using v=(1,m), the points are (t, y_a + t*m_a, y_b + t*m_b) for
    t in F_49^*. The a-component is t, which is in {0,1} only when t=0 or t=1.
    Since t != 0, t=1 gives a-component 1, which is in S. So this fails.
    
    Let me try v=(0,1). If x_a=0, then x+t*v has a-component 0 for all t.
    So all points are in S. This fails.
    
    So two vertical lines don't work! Let me think more carefully.
    
    For a set S to be a valid "blocking" set (complement of Nikodym set),
    we need: for every x in S and every direction v, the line through x in
    direction v contains at least one other point of S.
    
    With S = {x_a=0} union {x_a=1}, take x=(0, y). Direction v=(1,m).
    The line is {(t, y + t*m) : t in F_49}. Points in S on this line:
    - t=0: (0, y) - this is x itself
    - t=1: (1, y+m) - a-component 1, so in S
    
    So there IS another point in S on this line. Good.
    
    Take x=(0, y). Direction v=(0,1). Line is {(0, y+t) : t in F_49}.
    All points have a-component 0, so all are in S. Good.
    
    Take x=(1, y). Direction v=(1,m). Line is {(1+t, y+t*m) : t in F_49}.
    - t=0: (1, y) - x itself
    - t=-1: (0, y-m) - a-component 0, so in S
    
    Good.
    
    Take x=(1, y). Direction v=(0,1). Line is {(1, y+t) : t in F_49}.
    All have a-component 1, so all in S. Good.
    
    Now for x NOT in S (x_a not in {0,1}), we need SOME direction v such that
    the entire punctured line is in N. Since x is not in S, we just need to
    find a direction where no other point on the line is in S.
    
    For x with x_a = c where c not in {0,1}, use v=(1,m). The line is
    {(c+t, y+t*m) : t in F_49}. A point is in S iff its a-component is 0 or 1.
    So c+t = 0 or c+t = 1, i.e., t = -c or t = 1-c.
    For t=0: a-component is c (not in {0,1}), so x is in N. Good.
    For t=-c: a-component 0, in S.
    For t=1-c: a-component 1, in S.
    
    So there are 2 other points in S on this line. This doesn't work for
    making the punctured line fully in N.
    
    Hmm, so two vertical lines DON'T give a valid Nikodym set. The issue is
    that for points NOT in S, every non-vertical line through them will hit
    both x=0 and x=1.
    
    What if S is a single line? S = {x_a = 0}. |S| = 49.
    For x in S (x_a=0), direction v=(1,m): line hits x_a=1 at t=1, which is
    not in S. So only x itself is in S on this line. This fails the condition
    for x in S.
    
    So a single line doesn't work either.
    
    Let me think about what sets S work. The condition is:
    1. For x in S, every line through x in any direction has another point in S.
    2. For x not in S, there exists a direction v such that the punctured
       line through x in direction v has no points in S.
    
    Condition 1 means S is a "blocking set" in the sense that every line
    through a point of S meets S in at least 2 points.
    
    Actually, the standard result is that the complement of a line works if
    we're careful. Let me reconsider.
    
    Actually, I think the simplest valid construction is:
    S = a line (e.g., y=0, i.e., y_a=0 and y_b=0).
    |S| = 49.
    
    For x in S: x = (x_a, x_b, 0, 0).
    Direction v=(0,1): line is {(x_a, x_b, t, 0) : t in F_49}. Wait, no.
    v=(0,1) means the y-component changes by t*1. So line is
    {(x_a, x_b, 0 + t, 0 + 0*t) : t} = {(x_a, x_b, t, 0) : t}.
    Points in S on this line: those with y_a=0 and y_b=0. Only t=0 gives
    that. So only x itself. Fails.
    
    I'm struggling with the construction. Let me just use the baseline
    Hermitian unital approach which is known to work, and try to optimize
    the computation.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    p = 7
    q = p * p  # 49
    
    # Find smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3
    
    # Special y-coordinate set Y = {(u^2 mod p, v)}: y_a must be a square mod p
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # Build the excluded set S: points in unital U with y NOT in y_special
    # N = complement of S
    S_set = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S_set.add((xa, xb, ya, yb))
    
    # Build N: all points NOT in S
    N_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in S_set:
                        N_points.append([[xa, xb], [ya, yb]])
    
    if len(N_points) == 0:
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])
    
    result = np.array(N_points, dtype=np.int64)
    
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
