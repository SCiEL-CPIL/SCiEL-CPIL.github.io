import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Precompute lines
    # Points are (xa, xb, ya, yb)
    # We map point to index 0..2400
    # idx = ((xa * p + xb) * p + ya) * p + yb
    
    # Generate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append((xa, xb, ya, yb))
    
    n_points = len(points)
    
    # Map point tuple to index
    pt_to_idx = {pt: i for i, pt in enumerate(points)}
    
    # Precompute lines
    # A line is defined by a point and a direction
    # Directions: (1, m) for m in F_q, and (0, 1)
    # We represent direction as (dx_a, dx_b, dy_a, dy_b)
    # F_q elements are (a, b)
    
    # Helper to add points in F_q
    def add_pt(pt1, pt2):
        return ((pt1[0] + pt2[0]) % p, (pt1[1] + pt2[1]) % p, 
                (pt1[2] + pt2[2]) % p, (pt1[3] + pt2[3]) % p)
        
    def mul_scalar(pt, s_a, s_b):
        # pt * (s_a + s_b * alpha)
        # (x_a + x_b a) * (s_a + s_b a) = x_a s_a + x_b s_b a^2 + (x_a s_b + x_b s_a) a
        # a^2 = 3
        x_a, x_b, y_a, y_b = pt
        # Wait, pt is (xa, xb, ya, yb). Scalar is (sa, sb).
        # We need to multiply coordinate-wise? No, scalar multiplication is in F_q.
        # x' = x * s.
        # x = (xa, xb), s = (sa, sb).
        # x' = (xa*sa + xb*sb*3, xa*sb + xb*sa) mod 7.
        
        # This function is for generating lines.
        # We need to generate all lines.
        pass

    # Better line generation:
    # For each point P, and each direction D, the line is { P + t*D : t in F_q }
    # Directions D are (1, m) for m in F_q, and (0, 1).
    # m is (m_a, m_b).
    # D = (1, 0, m_a, m_b) ? No.
    # Vector v = (1, m).
    # 1 is (1, 0). m is (m_a, m_b).
    # So v = ((1,0), (m_a, m_b)).
    # And v = (0, 1) = ((0,0), (1,0)).
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    
    lines = []
    line_points = [] # List of sets of point indices
    
    # To avoid duplicates, we can just generate all and use a set of frozensets.
    # But there are only 2450 lines.
    # Total lines = (q^2 - 1) / (q - 1) = q + 1 = 50 lines per point.
    # Total unique lines = q^2 (q+1) / (q+1) ? No.
    # Number of lines in PG(1, q^2) ? No, affine plane.
    # Number of lines = q(q+1) ?
    # In F_q^2, number of lines is q(q+1).
    # For q=49, 49 * 50 = 2450 lines.
    
    line_set = {}
    
    for i, pt in enumerate(points):
        for d in directions:
            # Generate line points
            line_pts = []
            for t_a in range(p):
                for t_b in range(p):
                    # t = (t_a, t_b)
                    # pt + t * d
                    # t * d:
                    # dx = (1,0) * t or (0,0) * t
                    # dy = m * t or (1,0) * t
                    
                    # t * (1,0) = (t_a, t_b)
                    # t * (m_a, m_b) = (t_a*m_a + t_b*m_b*3, t_a*m_b + t_b*m_a) mod 7
                    
                    dx_a, dx_b = d[0]
                    dy_a, dy_b = d[1]
                    
                    # t * dx
                    if dx_a == 1 and dx_b == 0:
                        tdx_a = t_a
                        tdx_b = t_b
                    else: # dx is 0
                        tdx_a = 0
                        tdx_b = 0
                        
                    # t * dy
                    tdy_a = (t_a * dy_a + t_b * dy_b * 3) % p
                    tdy_b = (t_a * dy_b + t_b * dy_a) % p
                    
                    new_pt = (
                        (pt[0] + tdx_a) % p,
                        (pt[1] + tdx_b) % p,
                        (pt[2] + tdy_a) % p,
                        (pt[3] + tdy_b) % p
                    )
                    line_pts.append(pt_to_idx[new_pt])
            
            # Sort to create unique key
            key = tuple(sorted(line_pts))
            if key not in line_set:
                line_set[key] = set(line_pts)
                
    # Convert to list of lists for fast access
    lines_list = [list(s) for s in line_set.values()]
    num_lines = len(lines_list)
    
    # Map point index to list of line indices containing it
    pt_to_lines = [[] for _ in range(n_points)]
    for l_idx, l_pts in enumerate(lines_list):
        for pt_idx in l_pts:
            pt_to_lines[pt_idx].append(l_idx)
            
    # Greedy Algorithm
    best_C = set()
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 550 # Safety margin
    
    for seed in range(100):
        random.seed(seed)
        C = set()
        c_L = [0] * num_lines
        valid_count = [50] * n_points # Initially all lines have c_L=0, so all are valid witnesses for N-points
        
        # Order of points
        order = list(range(n_points))
        random.shuffle(order)
        
        for pt_idx in order:
            if time.time() > deadline:
                break
                
            # Tentatively add pt_idx to C
            C.add(pt_idx)
            
            # Update c_L for lines containing pt_idx
            affected_lines = pt_to_lines[pt_idx]
            for l_idx in affected_lines:
                c_L[l_idx] += 1
                
            # Check validity for points on affected lines
            # We need to update valid_count for all points on these lines
            # And check if any becomes 0
            
            # To be efficient, we can just recompute valid_count for points on affected lines
            # But valid_count depends on c_L for ALL lines through the point.
            # So we must update valid_count for points on affected lines.
            
            invalid_found = False
            # Points to update
            pts_to_update = set()
            for l_idx in affected_lines:
                pts_to_update.update(lines_list[l_idx])
                
            for pt_u in pts_to_update:
                # Recompute valid_count for pt_u
                vc = 0
                is_in_C = pt_u in C
                for l_idx_u in pt_to_lines[pt_u]:
                    if is_in_C:
                        if c_L[l_idx_u] == 1:
                            vc += 1
                    else:
                        if c_L[l_idx_u] == 0:
                            vc += 1
                            
                # We don't store valid_count globally in this simple check, 
                # but we need to know if vc > 0.
                if vc == 0:
                    invalid_found = True
                    break
                    
            if invalid_found:
                # Revert
                C.remove(pt_idx)
                for l_idx in affected_lines:
                    c_L[l_idx] -= 1
            else:
                # Keep
                pass
                
        if len(C) > best_size:
            best_size = len(C)
            best_C = set(C)
            
        if time.time() > deadline:
            break
            
    # Construct result array
    # N = All \ C
    # We need to return points of N.
    
    result = []
    for i, pt in enumerate(points):
        if i not in best_C:
            result.append([[pt[0], pt[1]], [pt[2], pt[3]]])
            
    return np.array(result, dtype=np.int64)
