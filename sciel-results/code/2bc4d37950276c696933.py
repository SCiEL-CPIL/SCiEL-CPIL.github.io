# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy: Start with a baseline construction (Hermitian unital with quadratic-residue
    y-exclusion), then greedily add more excluded points. A point can be added to the
    excluded set S (removed from N) if, for every point P on every line through that
    point, P still has at least one valid witness direction after the removal.
    
    Additionally, try removing points from N that are NOT on the unital, and try
    different initial excluded sets to find a better starting point.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    def sub_f49(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)
    
    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines for each direction
    # Directions: v = (1, m) for m in F_49 (49 directions), and v = (0, 1) (1 direction)
    # Total 50 directions
    
    direction_lines = []
    
    # Vertical direction: v = (0, 1)
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)
    direction_lines.append(list(vert_lines.values()))
    
    # Slope directions: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                # Line through (xa,xb, ya,yb) with direction (1, m):
                # intercept c = y - x*m
                mx = mul_f49((xa, xb), (ma, mb))
                c = sub_f49((ya, yb), mx)
                lines_dict[c].append(i)
            direction_lines.append(list(lines_dict.values()))
    
    # point_line[point_idx][dir_idx] = line_idx within that direction
    point_line = [[None] * 50 for _ in range(n)]
    
    for dir_idx in range(50):
        for line_idx, line in enumerate(direction_lines[dir_idx]):
            for pt_idx in line:
                point_line[pt_idx][dir_idx] = line_idx
    
    # Helper: compute not_in_N_count and witness_count
    def compute_counts(in_N):
        not_in_N_count = [[0] * len(direction_lines[d]) for d in range(50)]
        for dir_idx in range(50):
            for line_idx, line in enumerate(direction_lines[dir_idx]):
                count = 0
                for pt_idx in line:
                    if not in_N[pt_idx]:
                        count += 1
                not_in_N_count[dir_idx][line_idx] = count
        
        witness_count = [0] * n
        for pt_idx in range(n):
            for dir_idx in range(50):
                line_idx = point_line[pt_idx][dir_idx]
                count = not_in_N_count[dir_idx][line_idx]
                # pt has dir as witness iff count == 0 or (count == 1 and not in_N[pt_idx])
                if count == 0 or (count == 1 and not in_N[pt_idx]):
                    witness_count[pt_idx] += 1
        
        return not_in_N_count, witness_count
    
    # Helper: check if a point can be excluded (removed from N)
    # Returns True if exclusion is valid
    def can_exclude(pt_idx, in_N, not_in_N_count, witness_count):
        if in_N[pt_idx]:
            return False
        
        for dir_idx in range(50):
            line_idx = point_line[pt_idx][dir_idx]
            line = direction_lines[dir_idx][line_idx]
            old_count = not_in_N_count[dir_idx][line_idx]
            new_count = old_count + 1
            
            for Q in line:
                if Q == pt_idx:
                    continue
                had_witness = (old_count == 0) or (old_count == 1 and not in_N[Q])
                has_witness = (new_count == 0) or (new_count == 1 and not in_N[Q])
                
                if had_witness and not has_witness:
                    # Q loses this direction
                    if witness_count[Q] <= 1:
                        return False
        
        return True
    
    # Helper: perform greedy improvement
    def greedy_improve(in_N, not_in_N_count, witness_count, candidates, deadline):
        improved = True
        while improved:
            improved = False
            for cand in candidates:
                if time.time() > deadline - 30:
                    return in_N, not_in_N_count, witness_count
                
                if not in_N[cand]:
                    continue
                
                if can_exclude(cand, in_N, not_in_N_count, witness_count):
                    # Exclude cand
                    in_N[cand] = False
                    
                    # Update counts
                    for dir_idx in range(50):
                        line_idx = point_line[cand][dir_idx]
                        not_in_N_count[dir_idx][line_idx] += 1
                        
                        # Update witness counts for points on this line
                        line = direction_lines[dir_idx][line_idx]
                        new_count = not_in_N_count[dir_idx][line_idx]
                        for Q in line:
                            had = (not_in_N_count[dir_idx][line_idx] - 1 == 0) or \
                                  (not_in_N_count[dir_idx][line_idx] - 1 == 1 and not in_N[Q])
                            has = (new_count == 0) or (new_count == 1 and not in_N[Q])
                            if had and not has:
                                witness_count[Q] -= 1
                            elif not had and has:
                                witness_count[Q] += 1
                    
                    improved = True
        
        return in_N, not_in_N_count, witness_count
    
    # Quadratic residues mod 7
    QR = set((u*u) % p for u in range(p))
    
    # Try multiple baseline constructions
    best_in_N = None
    best_excluded = -1
    
    # Baseline 1: y_a in QR (quadratic residue)
    y_special_1 = set((ya, yb) for ya in QR for yb in range(p))
    in_N_1 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special_1:
            in_N_1[i] = False
    excluded_1 = sum(1 for x in in_N_1 if not x)
    
    # Baseline 2: y_a NOT in QR (non-quadratic residue)
    NQR = set(range(p)) - QR
    y_special_2 = set((ya, yb) for ya in NQR for yb in range(p))
    in_N_2 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special_2:
            in_N_2[i] = False
    excluded_2 = sum(1 for x in in_N_2 if not x)
    
    # Baseline 3: all y
    y_special_3 = set((ya, yb) for ya in range(p) for yb in range(p))
    in_N_3 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0 and (ya, yb) not in y_special_3:
            in_N_3[i] = False
    excluded_3 = sum(1 for x in in_N_3 if not x)
    
    # Baseline 4: no y exclusion (exclude all unital points)
    in_N_4 = [True] * n
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            in_N_4[i] = False
    excluded_4 = sum(1 for x in in_N_4 if not x)
    
    # Choose the best baseline
    baselines = [(in_N_1, excluded_1), (in_N_2, excluded_2), (in_N_3, excluded_3), (in_N_4, excluded_4)]
    baselines.sort(key=lambda x: -x[1])
    
    # Try greedy improvement on the best baseline
    in_N, excluded = baselines[0]
    in_N = in_N[:]  # copy
    
    not_in_N_count, witness_count = compute_counts(in_N)
    
    # Candidates: all points currently in N
    candidates = list(range(n))
    
    # Sort candidates: try points with higher witness count first (safer to exclude)
    candidates.sort(key=lambda i: -witness_count[i])
    
    in_N, not_in_N_count, witness_count = greedy_improve(in_N, not_in_N_count, witness_count, candidates, deadline)
    
    excluded = sum(1 for x in in_N if not x)
    
    # If time remains, try other baselines
    if time.time() < deadline - 60:
        for in_N_base, excl_base in baselines[1:]:
            if time.time() > deadline - 60:
                break
            
            in_N_try = in_N_base[:]
            not_in_N_count_try, witness_count_try = compute_counts(in_N_try)
            
            candidates_try = list(range(n))
            candidates_try.sort(key=lambda i: -witness_count_try[i])
            
            in_N_try, not_in_N_count_try, witness_count_try = greedy_improve(
                in_N_try, not_in_N_count_try, witness_count_try, candidates_try, deadline)
            
            excl_try = sum(1 for x in in_N_try if not x)
            
            if excl_try > excluded:
                in_N = in_N_try
                excluded = excl_try
    
    # Convert to array
    points = []
    for i in range(n):
        if in_N[i]:
            points.append(list(all_points[i]))
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
