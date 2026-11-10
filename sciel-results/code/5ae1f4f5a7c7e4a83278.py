# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using algebraic structure.
    
    Strategy: Use quadratic residues in F_7 to define which points to exclude.
    A point (x, y) is excluded from N (i.e., in S) if the y-coordinate's
    'a' component (real part) is a quadratic residue in F_7.
    
    This gives a structured complement that should be valid as a Nikodym set.
    """
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    p = 7
    k = 3  # alpha^2 = 3
    
    # F_49 multiplication
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    
    # Precompute lines for all 50 directions
    direction_lines = []
    
    # Vertical lines (v = (0, 1))
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[pt][dir] = line_idx
    point_line = [[0] * 50 for _ in range(n)]
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Quadratic residues in F_7
    # QR = {x^2 mod 7 : x in 0..6} = {0, 1, 2, 4}
    QR = {0, 1, 2, 4}
    
    # Define S: points where ya (real part of y) is in QR
    # This is a structured exclusion criterion
    S = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if ya in QR:
            S[i] = True
    
    s_size = sum(S)
    
    # Verify N is a valid Nikodym set
    # For every point x, there must exist a direction d such that 
    # L(x, d) \ {x} contains no points from S
    
    def is_valid_Nikodym(S_set):
        """Check if complement of S_set is a valid Nikodym set."""
        for x_idx in range(n):
            has_safe_dir = False
            for d in range(50):
                li = point_line[x_idx][d]
                line = direction_lines[d][li]
                # Check if any point in line (other than x_idx) is in S
                blocked = False
                for q in line:
                    if q != x_idx and S_set[q]:
                        blocked = True
                        break
                if not blocked:
                    has_safe_dir = True
                    break
            if not has_safe_dir:
                return False
        return True
    
    # Try different structured criteria to maximize |S|
    best_S = S[:]
    best_size = s_size
    
    # Try: ya in QR (current)
    if is_valid_Nikodym(best_S):
        pass  # Keep as is
    else:
        # Fallback: empty S (N = all points, always valid)
        best_S = [False] * n
        best_size = 0
    
    # Try: ya not in QR
    S2 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if ya not in QR:
            S2[i] = True
    s2_size = sum(S2)
    if s2_size > best_size and is_valid_Nikodym(S2):
        best_S = S2[:]
        best_size = s2_size
    
    # Try: xb in QR
    S3 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if xb in QR:
            S3[i] = True
    s3_size = sum(S3)
    if s3_size > best_size and is_valid_Nikodym(S3):
        best_S = S3[:]
        best_size = s3_size
    
    # Try: xa in QR
    S4 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if xa in QR:
            S4[i] = True
    s4_size = sum(S4)
    if s4_size > best_size and is_valid_Nikodym(S4):
        best_S = S4[:]
        best_size = s4_size
    
    # Try: yb in QR
    S5 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if yb in QR:
            S5[i] = True
    s5_size = sum(S5)
    if s5_size > best_size and is_valid_Nikodym(S5):
        best_S = S5[:]
        best_size = s5_size
    
    # Try combined criteria
    # ya in QR and xa in QR
    S6 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if ya in QR and xa in QR:
            S6[i] = True
    s6_size = sum(S6)
    if s6_size > best_size and is_valid_Nikodym(S6):
        best_S = S6[:]
        best_size = s6_size
    
    # ya in QR and xb in QR
    S7 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if ya in QR and xb in QR:
            S7[i] = True
    s7_size = sum(S7)
    if s7_size > best_size and is_valid_Nikodym(S7):
        best_S = S7[:]
        best_size = s7_size
    
    # ya in QR and yb in QR
    S8 = [False] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if ya in QR and yb in QR:
            S8[i] = True
    s8_size = sum(S8)
    if s8_size > best_size and is_valid_Nikodym(S8):
        best_S = S8[:]
        best_size = s8_size
    
    # Now try greedy refinement: try to add more points to best_S
    if time.time() < deadline - 30:
        # Use greedy addition to try to grow S further
        in_S = best_S[:]
        safe_mask = [(1 << 50) - 1] * n
        
        # Initialize safe masks based on current S
        for i in range(n):
            if not in_S[i]:
                continue
            for d in range(50):
                li = point_line[i][d]
                line = direction_lines[d][li]
                for q in line:
                    if q != i:
                        if (safe_mask[q] >> d) & 1:
                            safe_mask[q] &= ~(1 << d)
        
        # Try adding points not in S
        candidates = [i for i in range(n) if not in_S[i]]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline - 15:
                break
            
            # Check if we can add idx to S
            can_add = True
            for d in range(50):
                li = point_line[idx][d]
                line = direction_lines[d][li]
                for q in line:
                    if q == idx:
                        continue
                    if safe_mask[q] == (1 << d):
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                in_S[idx] = True
                for d in range(50):
                    li = point_line[idx][d]
                    line = direction_lines[d][li]
                    for q in line:
                        if q == idx:
                            continue
                        if (safe_mask[q] >> d) & 1:
                            safe_mask[q] &= ~(1 << d)
        
        if sum(in_S) > best_size:
            best_S = in_S[:]
            best_size = sum(best_S)
    
    # Build result N = complement of S
    pts = [all_points[i] for i in range(n) if not best_S[i]]
    if not pts:
        pts = all_points
        
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
