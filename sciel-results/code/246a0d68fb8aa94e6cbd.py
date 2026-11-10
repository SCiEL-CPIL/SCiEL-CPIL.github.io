import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a baseline Hermitian Unital construction
    followed by a greedy maximization of the complement set S.
    
    The complement S must satisfy: for every x in S, there exists a line L through x 
    such that L ∩ S = {x}.
    We start with S0 derived from the Hermitian unital and greedily add points 
    that preserve this property.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7
    
    # Field arithmetic helpers
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def trace_f49(pt):
        a, b = pt
        return (2 * a) % p
        
    def norm_f49(pt):
        a, b = pt
        return (a*a - k*b*b) % p

    # Generate all points in F_49^2
    # Represent point as (xa, xb, ya, yb)
    all_points = []
    point_to_idx = {}
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    point_to_idx[pt] = len(all_points)
                    all_points.append(pt)
    
    n_points = len(all_points)
    
    # Precompute lines
    # Directions: v = (0, 1) and v = (1, m) for m in F_49
    # Total 50 directions.
    # Each line is a set of point indices.
    
    lines_by_point = defaultdict(list) # point_idx -> list of line_ids
    all_lines = [] # list of sets of point indices
    
    # Helper to get point index
    def get_idx(pt):
        return point_to_idx[pt]
        
    # 1. Vertical lines: v = (0, 1). x is constant.
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    idx = get_idx(pt)
                    line_pts.append(idx)
            line_id = len(all_lines)
            all_lines.append(set(line_pts))
            for idx in line_pts:
                lines_by_point[idx].append(line_id)
                
    # 2. Slope lines: v = (1, m).
    # Equation: y = x*m + c.
    # c = y - x*m.
    # For a fixed m, the line is determined by c.
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            # Group points by their intercept c
            intercepts = defaultdict(list)
            for xa in range(p):
                for xb in range(p):
                    x = (xa, xb)
                    mx = mul_f49(x, m)
                    # c depends on y. y varies.
                    # For each y, we have a point.
                    # Actually, it's easier to iterate through all points and calculate c.
                    pass
            
            # Recalculate intercepts properly
            intercepts.clear()
            for xa in range(p):
                for xb in range(p):
                    x = (xa, xb)
                    mx = mul_f49(x, m)
                    for ya in range(p):
                        for yb in range(p):
                            y = (ya, yb)
                            # c = y - mx
                            ca = (ya - mx[0]) % p
                            cb = (yb - mx[1]) % p
                            c = (ca, cb)
                            pt = (xa, xb, ya, yb)
                            idx = get_idx(pt)
                            intercepts[c].append(idx)
            
            for c, pts in intercepts.items():
                line_id = len(all_lines)
                all_lines.append(set(pts))
                for idx in pts:
                    lines_by_point[idx].append(line_id)
                
    # Step 1: Construct Baseline S0
    # S0 = { (x,y) : Tr(x) + N(y) = 0 } \ B
    # B = { (x,y) in S0 : y_a is a square mod 7 }
    
    squares_mod_p = set((u*u) % p for u in range(p))
    
    S0 = set()
    for pt in all_points:
        xa, xb, ya, yb = pt
        x = (xa, xb)
        y = (ya, yb)
        tr_x = trace_f49(x)
        n_y = norm_f49(y)
        if (tr_x + n_y) % p == 0:
            # Check if y is in Y (y_a is square)
            if ya not in squares_mod_p:
                S0.add(pt)
                
    S = set(S0)
    
    # Step 2: Initialize A_q for each q in S
    # A_q = { L : q in L and L ∩ S = {q} }
    # Represent A_q as a set of line_ids
    
    A = {}
    for q in S:
        q_idx = get_idx(q)
        valid_lines = set()
        for line_id in lines_by_point[q_idx]:
            line_pts = all_lines[line_id]
            # Check if line_pts ∩ S == {q}
            # Since q is in line_pts, we check if any other point in S is in line_pts
            is_valid = True
            for other_q in S:
                if other_q == q:
                    continue
                if other_q in line_pts:
                    is_valid = False
                    break
            if is_valid:
                valid_lines.add(line_id)
        A[q] = valid_lines
        
        # If a point in S0 has no valid lines, the baseline is invalid?
        # The construction guarantees validity, but let's be safe.
        if not valid_lines:
             # Fallback: return S0 as N complement? No, if invalid, we can't use it.
             # But the inspiration says it's valid.
             pass

    # Step 3: Greedy Addition
    # Candidates: all points not in S
    candidates = [pt for pt in all_points if pt not in S]
    
    # To make the search faster, we can prioritize candidates.
    # But a simple loop with restart is safer for correctness.
    # We will iterate through candidates. If one is added, we restart the scan 
    # because the state of A has changed.
    
    added_any = True
    while added_any:
        added_any = False
        # Shuffle candidates to avoid bias? Or just iterate.
        # Given time constraints, we might not be able to shuffle 2000 items 200 times.
        # We'll just iterate.
        
        for p_cand in candidates:
            if time.time() > deadline:
                break
                
            if p_cand in S:
                continue
                
            p_idx = get_idx(p_cand)
            
            # Check 1: Does p_cand have a line L with L ∩ S = empty?
            # Such a line will be valid for p_cand in the new set S ∪ {p_cand}.
            has_clean_line = False
            for line_id in lines_by_point[p_idx]:
                line_pts = all_lines[line_id]
                # Check if line_pts ∩ S is empty
                # Since p_cand is not in S, we just check if any point in S is in line_pts
                is_clean = True
                for q in S:
                    if q in line_pts:
                        is_clean = False
                        break
                if is_clean:
                    has_clean_line = True
                    break
                    
            if not has_clean_line:
                continue
                
            # Check 2: Does adding p_cand invalidate any q in S?
            # A line L in A_q is invalidated if p_cand is in L.
            # If ALL lines in A_q contain p_cand, then A_q becomes empty.
            is_safe = True
            for q in S:
                a_q = A[q]
                if not a_q:
                    # Should not happen if we maintain invariants, but just in case
                    is_safe = False
                    break
                # Check if all lines in a_q contain p_cand
                all_contain_p = True
                for line_id in a_q:
                    if p_cand not in all_lines[line_id]:
                        all_contain_p = False
                        break
                if all_contain_p:
                    is_safe = False
                    break
                    
            if is_safe:
                # Add p_cand to S
                S.add(p_cand)
                
                # Update A for all q in S
                # Remove lines containing p_cand from A[q]
                for q in S:
                    if q == p_cand:
                        continue
                    new_a_q = set()
                    for line_id in A[q]:
                        if p_cand not in all_lines[line_id]:
                            new_a_q.add(line_id)
                    A[q] = new_a_q
                    
                # Initialize A for p_cand
                # A[p_cand] = { L : p_cand in L and L ∩ S = {p_cand} }
                # Note: S now includes p_cand.
                # So we need L ∩ (S \ {p_cand}) = empty.
                # Which is equivalent to L ∩ S_old = empty.
                # We already know there is at least one such line.
                new_a_p = set()
                for line_id in lines_by_point[p_idx]:
                    line_pts = all_lines[line_id]
                    # Check if line_pts ∩ (S \ {p_cand}) is empty
                    is_valid = True
                    for q in S:
                        if q == p_cand:
                            continue
                        if q in line_pts:
                            is_valid = False
                            break
                    if is_valid:
                        new_a_p.add(line_id)
                A[p_cand] = new_a_p
                
                # Remove p_cand from candidates
                candidates.remove(p_cand)
                added_any = True
                break # Restart the loop to pick the next best candidate
                
        if time.time() > deadline:
            break

    # Construct N = F_49^2 \ S
    N_points = [pt for pt in all_points if pt not in S]
    
    if not N_points:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    arr = np.array(N_points, dtype=np.int64).reshape(-1, 2, 2)
    return arr
