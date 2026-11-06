import random
import math
import time

N = 64

# Precompute all points
all_points = [(x, y) for x in range(N) for y in range(N)]

def get_conflicts(p, a):
    """
    Returns a set of points q such that {p, a, q} forms an isosceles triangle.
    p, a are tuples (x, y).
    """
    res = set()
    px, py = p
    ax, ay = a
    
    dx = ax - px
    dy = ay - py
    r2 = dx*dx + dy*dy
    
    if r2 == 0:
        return res
        
    # Case 1: q is on the perpendicular bisector of pa
    # 2*dx*x + 2*dy*y = (ax^2 - px^2) + (ay^2 - py^2)
    A = 2 * dx
    B = 2 * dy
    C = (ax*ax - px*px) + (ay*ay - py*py)
    
    if A == 0 and B == 0:
        pass # Should not happen if r2 > 0
    elif A == 0:
        # B * y = C
        if C % B == 0:
            y_val = C // B
            if 0 <= y_val < N:
                for x in range(N):
                    res.add((x, y_val))
    elif B == 0:
        # A * x = C
        if C % A == 0:
            x_val = C // A
            if 0 <= x_val < N:
                for y in range(N):
                    res.add((x_val, y))
    else:
        # A * x + B * y = C
        # Iterate x, solve for y
        # y = (C - A*x) / B
        for x in range(N):
            val = C - A * x
            if val % B == 0:
                y = val // B
                if 0 <= y < N:
                    res.add((x, y))

    # Case 2: q is on circle centered at p with radius sqrt(r2)
    for qx in range(N):
        dx_q = qx - px
        rem = r2 - dx_q * dx_q
        if rem < 0:
            continue
        dy_q = int(math.sqrt(rem))
        if dy_q * dy_q == rem:
            qy1 = py + dy_q
            qy2 = py - dy_q
            if 0 <= qy1 < N:
                res.add((qx, qy1))
            if 0 <= qy2 < N and qy2 != qy1:
                res.add((qx, qy2))

    # Case 3: q is on circle centered at a with radius sqrt(r2)
    for qx in range(N):
        dx_q = qx - ax
        rem = r2 - dx_q * dx_q
        if rem < 0:
            continue
        dy_q = int(math.sqrt(rem))
        if dy_q * dy_q == rem:
            qy1 = ay + dy_q
            qy2 = ay - dy_q
            if 0 <= qy1 < N:
                res.add((qx, qy1))
            if 0 <= qy2 < N and qy2 != qy1:
                res.add((qx, qy2))

    # Remove p and a themselves, as they are already in S
    res.discard(p)
    res.discard(a)
    
    return res

def construct():
    best_set = []
    deadline = time.time() + 250.0 # Leave 50s margin
    
    # We can also try to start with a small random set to speed up convergence?
    # No, starting empty is safer for greedy.
    
    while time.time() < deadline:
        S = set()
        F = set() # Forbidden points
        
        # Random order of candidates
        candidates = all_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if p in F:
                continue
                
            # Check if p is already in S (should not be, but safe)
            if p in S:
                continue
            
            # Try to add p
            # If we add p, we must forbid all q that form isosceles with p and any a in S
            # We can do this check before adding, or add then rollback if it fails?
            # Since F contains points that *would* form isosceles with *existing* S,
            # if p is not in F, it means p does not form an isosceles triangle with any pair in S.
            # So we can safely add p.
            
            S.add(p)
            
            # Update F with conflicts involving p and all existing a in S
            for a in S:
                if a == p:
                    continue
                conflicts = get_conflicts(p, a)
                F.update(conflicts)
        
        if len(S) > len(best_set):
            best_set = list(S)
            
    return best_set
